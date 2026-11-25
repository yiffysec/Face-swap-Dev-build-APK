package com.faceswap.studio.video

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import com.faceswap.core.FaceMapping
import com.faceswap.core.FaceSwapEngine
import com.faceswap.core.FaceTracker
import com.faceswap.core.ImageOps
import com.faceswap.core.RgbImage
import com.faceswap.core.SwapConfig
import com.faceswap.studio.util.Bitmaps
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.max
import kotlin.math.roundToInt

data class VideoProgress(
    val frame: Int,
    val totalFrames: Int,
    val fps: Float,
    val etaSeconds: Int,
    val facesSwapped: Int,
    val preview: Bitmap?,
) {
    val fraction get() = if (totalFrames > 0) (frame.toFloat() / totalFrames).coerceIn(0f, 1f) else 0f
}

data class VideoResult(val file: File, val frames: Int, val seconds: Float, val audioCopied: Boolean)

/**
 * Decode (MediaCodec) → upright RGB → face swap → H.264 encode (MediaCodec, YUV420 flexible) → MP4 mux,
 * with the original audio track copied through untouched.
 */
class VideoSwapper(private val context: Context) {

    suspend fun run(
        engine: FaceSwapEngine,
        input: Uri,
        output: File,
        mappings: List<FaceMapping>,
        config: SwapConfig,
        maxSide: Int,
        keepAudio: Boolean,
        workers: Int = 2,
        perf: com.faceswap.studio.engine.PerformanceManager? = null,
        onProgress: (VideoProgress) -> Unit,
    ): VideoResult = withContext(Dispatchers.Default) {
        // Render workers: swapping/enhancing N frames in parallel while this thread keeps decoding + detecting.
        val tids = java.util.Collections.synchronizedList(ArrayList<Int>())
        val pool = java.util.concurrent.Executors.newFixedThreadPool(workers.coerceIn(1, 4)) { r ->
            Thread({ tids += android.os.Process.myTid(); android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY); r.run() }, "swap-worker")
        }
        val pending = ArrayDeque<Triple<Long, java.util.concurrent.Future<RgbImage>, Int>>()
        val t0 = System.currentTimeMillis()
        val extractor = MediaExtractor().apply { setDataSource(context, input, null) }
        var decoder: MediaCodec? = null
        var encoder: MediaCodec? = null
        var muxer: MediaMuxer? = null
        var muxerStarted = false
        var audioCopied = false
        try {
            val vIdx = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).mime().startsWith("video/") }
                ?: error("No video track found")
            val aIdx = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).mime().startsWith("audio/") }
            val inFormat = extractor.getTrackFormat(vIdx)
            val srcW = inFormat.getInteger(MediaFormat.KEY_WIDTH)
            val srcH = inFormat.getInteger(MediaFormat.KEY_HEIGHT)
            val rotation = readRotation(input, inFormat)
            val fps = if (inFormat.containsKey(MediaFormat.KEY_FRAME_RATE)) runCatching { inFormat.getInteger(MediaFormat.KEY_FRAME_RATE) }.getOrElse { inFormat.getFloat(MediaFormat.KEY_FRAME_RATE).roundToInt() } else 30
            val durationUs = if (inFormat.containsKey(MediaFormat.KEY_DURATION)) inFormat.getLong(MediaFormat.KEY_DURATION) else 0L
            val totalFrames = max(1, (durationUs / 1_000_000.0 * fps).roundToInt())

            // output size: fit within maxSide, multiples of 16 for encoder compatibility
            val scale = minOf(1f, maxSide.toFloat() / max(srcW, srcH))
            val outW = max(16, ((srcW * scale).toInt() / 16) * 16)
            val outH = max(16, ((srcH * scale).toInt() / 16) * 16)

            extractor.selectTrack(vIdx)
            inFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            decoder = MediaCodec.createDecoderByType(inFormat.mime()).apply { configure(inFormat, null, null, 0); start() }

            val encFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, outW, outH).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                setInteger(MediaFormat.KEY_BIT_RATE, (outW * outH * fps * 0.2f).toInt().coerceIn(2_000_000, 40_000_000))
                setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }
            encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).apply {
                configure(encFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE); start()
            }
            output.parentFile?.mkdirs(); output.delete()
            muxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4).apply { setOrientationHint(rotation) }
            val audioFormat = if (keepAudio && aIdx != null) extractor.getTrackFormat(aIdx) else null
            var outVideoTrack = -1
            var outAudioTrack = -1

            val info = MediaCodec.BufferInfo()
            val encInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var decoderDone = false
            var encoderDone = false
            var frameNo = 0
            var swappedTotal = 0
            var explicit = false
            val checkEvery = max(1, fps / 2)
            val tracker = FaceTracker()
            val dec = decoder!!
            val enc = encoder!!
            val mux = muxer!!

            fun drainEncoder(untilEos: Boolean) {
                while (true) {
                    val oi = enc.dequeueOutputBuffer(encInfo, if (untilEos) 10_000 else 0)
                    when {
                        oi == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!untilEos) return
                        oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            outVideoTrack = mux.addTrack(enc.outputFormat)
                            if (audioFormat != null) outAudioTrack = runCatching { mux.addTrack(audioFormat) }.getOrDefault(-1)
                            mux.start(); muxerStarted = true
                        }
                        oi >= 0 -> {
                            val buf = enc.getOutputBuffer(oi)!!
                            if (encInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) encInfo.size = 0
                            if (encInfo.size > 0 && muxerStarted) {
                                buf.position(encInfo.offset); buf.limit(encInfo.offset + encInfo.size)
                                mux.writeSampleData(outVideoTrack, buf, encInfo)
                            }
                            enc.releaseOutputBuffer(oi, false)
                            if (encInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) { encoderDone = true; return }
                        }
                    }
                }
            }

            fun queueToEncoder(frame: RgbImage?, ptsUs: Long, eos: Boolean) {
                while (true) {
                    val ii = enc.dequeueInputBuffer(10_000)
                    if (ii >= 0) {
                        if (eos) { enc.queueInputBuffer(ii, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM); return }
                        val img = enc.getInputImage(ii)!!
                        Yuv.fromRgb(frame!!, img)
                        enc.queueInputBuffer(ii, 0, outW * outH * 3 / 2, ptsUs, 0)
                        return
                    }
                    drainEncoder(false)
                }
            }

            var lastPts = 0L
            var perfStarted = false
            var lastOut: RgbImage? = null
            fun flush(all: Boolean) {
                while (pending.isNotEmpty() && (all || pending.size > workers || pending.first().second.isDone)) {
                    val (pts, fut, swapped) = pending.removeFirst()
                    val rendered = fut.get()
                    lastOut = rendered
                    queueToEncoder(Yuv.rotate(rendered, (360 - rotation) % 360), pts, false)
                    swappedTotal += swapped
                    perf?.frameDone()
                }
            }
            while (!encoderDone) {
                coroutineContext.ensureActive()
                if (!inputDone) {
                    val ii = dec.dequeueInputBuffer(10_000)
                    if (ii >= 0) {
                        val n = extractor.readSampleData(dec.getInputBuffer(ii)!!, 0)
                        if (n < 0) { dec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true }
                        else { dec.queueInputBuffer(ii, 0, n, extractor.sampleTime, 0); extractor.advance() }
                    }
                }
                if (!decoderDone) {
                    val oi = dec.dequeueOutputBuffer(info, 10_000)
                    if (oi >= 0) {
                        val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        var frame: RgbImage? = null
                        if (info.size > 0) dec.getOutputImage(oi)?.let { frame = Yuv.toRgb(it) }
                        val pts = info.presentationTimeUs
                        dec.releaseOutputBuffer(oi, false)
                        frame?.let { decoded ->
                            var f = if (decoded.width != outW || decoded.height != outH) ImageOps.resize(decoded, outW, outH) else decoded
                            f = Yuv.rotate(f, rotation) // upright for detection
                            if (frameNo % checkEvery == 0) explicit = engine.policy.isExplicit(f)
                            val upright = f
                            val plan = engine.analyze(upright, mappings, config, tracker, explicitOverride = explicit)
                            val fut = pool.submit<RgbImage> { engine.render(upright, plan, config) }
                            pending.addLast(Triple(pts, fut, plan.assignments.size))
                            if (!perfStarted && tids.size >= minOf(workers.coerceIn(1, 4), frameNo + 1)) {
                                perf?.hint(tids.toList() + android.os.Process.myTid(), 1000L / fps.coerceAtLeast(1)); perfStarted = true
                            }
                            flush(false)
                            lastPts = pts
                            frameNo++
                            if (frameNo % 10 == 1) {
                                val el = (System.currentTimeMillis() - t0) / 1000f
                                val rate = frameNo / max(0.001f, el)
                                // newest finished frame – never block the pipeline on the frame just submitted
                                val preview = lastOut ?: upright
                                onProgress(VideoProgress(frameNo, totalFrames, rate, ((totalFrames - frameNo) / max(0.01f, rate)).roundToInt(), swappedTotal,
                                    Bitmaps.toBitmap(ImageOps.resize(preview, preview.width / 2, preview.height / 2))))
                            }
                        }
                        if (eos) { flush(true); decoderDone = true; queueToEncoder(null, lastPts, true) }
                    }
                }
                drainEncoder(decoderDone)
            }

            // copy audio track through untouched
            if (audioFormat != null && outAudioTrack >= 0 && muxerStarted) {
                val ax = MediaExtractor().apply { setDataSource(context, input, null); selectTrack(aIdx!!) }
                val cap = if (audioFormat.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) audioFormat.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE) else 1 shl 20
                val buf = ByteBuffer.allocate(max(cap, 64 * 1024))
                val ai = MediaCodec.BufferInfo()
                while (true) {
                    coroutineContext.ensureActive()
                    val n = ax.readSampleData(buf, 0)
                    if (n < 0) break
                    ai.set(0, n, ax.sampleTime, if (ax.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0)
                    mux.writeSampleData(outAudioTrack, buf, ai)
                    ax.advance()
                }
                ax.release()
                audioCopied = true
            }
            onProgress(VideoProgress(frameNo, frameNo, 0f, 0, swappedTotal, null))
            VideoResult(output, frameNo, (System.currentTimeMillis() - t0) / 1000f, audioCopied)
        } finally {
            pool.shutdownNow()
            runCatching { decoder?.stop() }; runCatching { decoder?.release() }
            runCatching { encoder?.stop() }; runCatching { encoder?.release() }
            runCatching { if (muxerStarted) muxer?.stop() }; runCatching { muxer?.release() }
            extractor.release()
        }
    }

    private fun readRotation(uri: Uri, format: MediaFormat): Int {
        if (format.containsKey(MediaFormat.KEY_ROTATION)) return format.getInteger(MediaFormat.KEY_ROTATION)
        return runCatching {
            MediaMetadataRetriever().use { r ->
                r.setDataSource(context, uri)
                r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
            }
        }.getOrDefault(0)
    }

    private fun MediaFormat.mime() = getString(MediaFormat.KEY_MIME) ?: ""
}

/** YUV420 (any plane layout) <-> ARGB conversion (BT.601 limited range) and 90° rotations. */
object Yuv {
    fun toRgb(image: Image): RgbImage {
        val crop = image.cropRect
        val w = crop.width(); val h = crop.height()
        val out = RgbImage(w, h)
        val y = image.planes[0]; val u = image.planes[1]; val v = image.planes[2]
        // bulk-copy planes once (per-pixel ByteBuffer.get(index) is several times slower)
        fun bytes(b: ByteBuffer) = b.duplicate().let { d -> d.position(0); ByteArray(d.limit()).also { d.get(it) } }
        val yb = bytes(y.buffer); val ub = bytes(u.buffer); val vb = bytes(v.buffer)
        val yRow = y.rowStride; val yPix = y.pixelStride
        val uRow = u.rowStride; val uPix = u.pixelStride
        val vRow = v.rowStride; val vPix = v.pixelStride
        val op = out.pixels
        com.faceswap.core.Par.rows(h, w) { from, to ->
            for (j in from until to) {
                val sy = j + crop.top
                val yo = sy * yRow; val uo = (sy / 2) * uRow; val vo = (sy / 2) * vRow
                for (i in 0 until w) {
                    val sx = i + crop.left
                    val yy = (yb[yo + sx * yPix].toInt() and 0xFF) - 16
                    val cu = (ub[uo + (sx / 2) * uPix].toInt() and 0xFF) - 128
                    val cv = (vb[vo + (sx / 2) * vPix].toInt() and 0xFF) - 128
                    val c = 1.164f * yy
                    val r = (c + 1.596f * cv).toInt().coerceIn(0, 255)
                    val g = (c - 0.392f * cu - 0.813f * cv).toInt().coerceIn(0, 255)
                    val b = (c + 2.017f * cu).toInt().coerceIn(0, 255)
                    op[j * w + i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        return out
    }

    fun fromRgb(src: RgbImage, image: Image) {
        val w = minOf(src.width, image.width); val h = minOf(src.height, image.height)
        val y = image.planes[0]; val u = image.planes[1]; val v = image.planes[2]
        val yb = y.buffer; val ub = u.buffer; val vb = v.buffer
        val yRow = y.rowStride; val yPix = y.pixelStride
        val uRow = u.rowStride; val uPix = u.pixelStride
        val vRow = v.rowStride; val vPix = v.pixelStride
        // absolute puts on disjoint indices are thread-safe
        com.faceswap.core.Par.rows(h, w) { from, to ->
            for (j in from until to) for (i in 0 until w) {
                val p = src.pixels[j * src.width + i]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                val yy = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                yb.put(j * yRow + i * yPix, yy.coerceIn(0, 255).toByte())
                if (j % 2 == 0 && i % 2 == 0) {
                    val cu = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val cv = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    ub.put((j / 2) * uRow + (i / 2) * uPix, cu.coerceIn(0, 255).toByte())
                    vb.put((j / 2) * vRow + (i / 2) * vPix, cv.coerceIn(0, 255).toByte())
                }
            }
        }
    }

    /** Clockwise rotation by 0/90/180/270. */
    fun rotate(src: RgbImage, degrees: Int): RgbImage {
        val d = ((degrees % 360) + 360) % 360
        if (d == 0) return src
        val w = src.width; val h = src.height
        return when (d) {
            90 -> RgbImage(h, w).also { o -> for (y in 0 until h) for (x in 0 until w) o.pixels[x * h + (h - 1 - y)] = src.pixels[y * w + x] }
            180 -> RgbImage(w, h).also { o -> for (i in src.pixels.indices) o.pixels[w * h - 1 - i] = src.pixels[i] }
            270 -> RgbImage(h, w).also { o -> for (y in 0 until h) for (x in 0 until w) o.pixels[(w - 1 - x) * h + y] = src.pixels[y * w + x] }
            else -> src
        }
    }
}
