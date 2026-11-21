package com.faceswap.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** Creates ORT sessions. Android supplies one with NNAPI/XNNPACK options; tests use plain CPU. */
fun interface SessionFactory {
    fun create(env: OrtEnvironment, modelPath: String): OrtSession
}

abstract class OnnxModel(protected val env: OrtEnvironment, file: File, factory: SessionFactory) : Closeable {
    protected val session: OrtSession = factory.create(env, file.absolutePath)
    protected val inputNames: List<String> = session.inputNames.toList()

    protected fun tensor(data: FloatArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)

    override fun close() = session.close()
}

private fun OrtSession.Result.floatAt(i: Int): FloatArray {
    val t = get(i) as OnnxTensor
    val fb = t.floatBuffer
    return FloatArray(fb.remaining()).also { fb.get(it) }
}

/** SCRFD face detector (InsightFace). Outputs: scores×3, boxes×3, keypoints×3 for strides 8/16/32. */
class FaceDetector(env: OrtEnvironment, file: File, factory: SessionFactory) : OnnxModel(env, file, factory) {

    fun detect(img: RgbImage, inputSize: Int = 640, scoreThreshold: Float = 0.5f, nmsThreshold: Float = 0.4f): List<Face> {
        val ratio = min(inputSize.toFloat() / img.width, inputSize.toFloat() / img.height)
        // letterbox: scale, pad right/bottom with black
        val scaled = ImageOps.warpAffine(img, Affine.scale(ratio), inputSize, inputSize)
        val sw = (img.width * ratio).toInt(); val sh = (img.height * ratio).toInt()
        for (y in 0 until inputSize) for (x in 0 until inputSize) if (x >= sw || y >= sh) scaled.pixels[y * inputSize + x] = 0xFF000000.toInt()
        val input = ImageOps.toChw(scaled, mean = 127.5f, std = 128f)
        val faces = ArrayList<Face>()
        tensor(input, 1, 3, inputSize.toLong(), inputSize.toLong()).use { t ->
            session.run(mapOf(inputNames[0] to t)).use { res ->
                val strides = intArrayOf(8, 16, 32)
                for ((si, stride) in strides.withIndex()) {
                    val scores = res.floatAt(si)
                    val boxes = res.floatAt(si + 3)
                    val kps = res.floatAt(si + 6)
                    val fw = inputSize / stride
                    val anchors = scores.size / (fw * fw).coerceAtLeast(1)
                    for (i in scores.indices) {
                        val s = scores[i]
                        if (s < scoreThreshold) continue
                        val loc = i / anchors
                        val cx = (loc % fw) * stride.toFloat()
                        val cy = (loc / fw) * stride.toFloat()
                        val lm = FloatArray(10)
                        for (k in 0 until 5) {
                            lm[2 * k] = (cx + kps[i * 10 + 2 * k] * stride) / ratio
                            lm[2 * k + 1] = (cy + kps[i * 10 + 2 * k + 1] * stride) / ratio
                        }
                        faces += Face(
                            (cx - boxes[i * 4] * stride) / ratio, (cy - boxes[i * 4 + 1] * stride) / ratio,
                            (cx + boxes[i * 4 + 2] * stride) / ratio, (cy + boxes[i * 4 + 3] * stride) / ratio,
                            lm, s,
                        )
                    }
                }
            }
        }
        return Geometry.nms(faces, nmsThreshold)
    }
}

/** ArcFace w600k_r50 – 512-d identity embedding from a 112×112 aligned crop. */
class FaceRecognizer(env: OrtEnvironment, file: File, factory: SessionFactory) : OnnxModel(env, file, factory) {
    fun embed(img: RgbImage, face: Face): FloatArray {
        val m = Geometry.estimateSimilarity(face.landmarks, Templates.ARCFACE_112, 112f)
        val crop = ImageOps.warpAffine(img, m, 112, 112)
        val input = ImageOps.toChw(crop, 127.5f, 127.5f)
        return tensor(input, 1, 3, 112, 112).use { t ->
            session.run(mapOf(inputNames[0] to t)).use { Geometry.normalize(it.floatAt(0)) }
        }
    }
}

/** InSwapper 128 – swaps identity into a 128×128 aligned crop. */
class FaceSwapper(env: OrtEnvironment, private val file: File, factory: SessionFactory, emapCache: File? = null) :
    OnnxModel(env, file, factory) {

    private val emap: FloatArray = loadEmap(emapCache)
    private val targetName = inputNames.firstOrNull { it.contains("target") } ?: inputNames[0]
    private val sourceName = inputNames.firstOrNull { it.contains("source") } ?: inputNames[1]

    private fun loadEmap(cache: File?): FloatArray {
        if (cache != null && cache.length() == 512L * 512 * 4) {
            val bytes = cache.readBytes()
            val bb = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            return FloatArray(512 * 512) { bb.float }
        }
        val m = OnnxInitializerReader.readFloatInitializer(file, expectedSize = 512 * 512)
        if (cache != null) {
            val bb = java.nio.ByteBuffer.allocate(m.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            m.forEach { bb.putFloat(it) }
            cache.parentFile?.mkdirs(); cache.writeBytes(bb.array())
        }
        return m
    }

    /** Converts a normalised ArcFace embedding to the swapper's latent space: normalize(e · emap). */
    fun latent(embedding: FloatArray): FloatArray {
        val out = FloatArray(512)
        for (i in 0 until 512) {
            val e = embedding[i]
            if (e == 0f) continue
            val row = i * 512
            for (j in 0 until 512) out[j] += e * emap[row + j]
        }
        return Geometry.normalize(out)
    }

    /** [crop] must be 128×128 aligned with [Templates.ARCFACE_128]. */
    fun swap(crop: RgbImage, latent: FloatArray): RgbImage {
        val target = ImageOps.toChw(crop, 0f, 1f, scale = 1f / 255f)
        tensor(target, 1, 3, 128, 128).use { t ->
            tensor(latent, 1, 512).use { s ->
                session.run(mapOf(targetName to t, sourceName to s)).use { res ->
                    return ImageOps.fromChw(res.floatAt(0), 128, 128, 0f, 1f, scale = 1f / 255f)
                }
            }
        }
    }
}

/** GFPGAN 1.4 face restoration, 512×512 aligned with FFHQ template. Range [-1,1] RGB. */
class FaceEnhancer(env: OrtEnvironment, file: File, factory: SessionFactory) : OnnxModel(env, file, factory) {
    val size = 512
    fun enhance(crop: RgbImage): RgbImage {
        val input = ImageOps.toChw(crop, 0.5f, 0.5f, scale = 1f / 255f)
        tensor(input, 1, 3, size.toLong(), size.toLong()).use { t ->
            val feeds = HashMap<String, OnnxTensor>()
            feeds[inputNames[0]] = t
            session.run(feeds).use { res ->
                val out = res.floatAt(0)
                for (i in out.indices) out[i] = out[i].coerceIn(-1f, 1f)
                return ImageOps.fromChw(out, size, size, 0.5f, 0.5f, scale = 1f / 255f)
            }
        }
    }
}

/** Yahoo open_nsfw classifier – returns probability that an image is sexually explicit. */
class ContentAnalyser(env: OrtEnvironment, file: File, factory: SessionFactory) : OnnxModel(env, file, factory) {
    fun explicitProbability(img: RgbImage): Float {
        val small = ImageOps.resize(img, 224, 224)
        val data = FloatArray(224 * 224 * 3)
        for (i in 0 until 224 * 224) { // NHWC, BGR, mean-subtracted
            val p = small.pixels[i]
            data[i * 3] = (p and 0xFF) - 104f
            data[i * 3 + 1] = ((p shr 8) and 0xFF) - 117f
            data[i * 3 + 2] = ((p shr 16) and 0xFF) - 123f
        }
        return tensor(data, 1, 224, 224, 3).use { t ->
            session.run(mapOf(inputNames[0] to t)).use { max(0f, it.floatAt(0)[1]) }
        }
    }
}
