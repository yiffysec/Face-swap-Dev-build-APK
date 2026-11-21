package com.faceswap.core

import ai.onnxruntime.OrtEnvironment
import java.io.Closeable
import java.io.File

/** One identity built from one or more photos (embeddings averaged for a more stable likeness). */
class SourceFace(
    val id: String,
    val label: String,
    embeddings: List<FloatArray>,
    val provenance: SourceProvenance = SourceProvenance.GALLERY,
) {
    val embeddings: List<FloatArray> = embeddings.map { Geometry.normalize(it) }
    val averaged: FloatArray = Geometry.normalize(FloatArray(512) { i -> this.embeddings.sumOf { it[i].toDouble() }.toFloat() })
    @Volatile internal var latent: FloatArray? = null
}

/** Maps a source identity onto target faces. [reference] = embedding of the target person to replace (null = selector rule). */
data class FaceMapping(val source: SourceFace, val reference: FloatArray? = null)

enum class FaceSelector { ALL, LARGEST, LEFTMOST, RIGHTMOST, CENTER }
enum class MaskShape { BOX, ELLIPSE, BOX_AND_ELLIPSE }

data class SwapConfig(
    val selector: FaceSelector = FaceSelector.ALL,
    /** Cosine similarity needed to match a target face to a mapping reference. */
    val referenceThreshold: Float = 0.35f,
    /** 128 = native, 256/384/512 = pixel boost (k² sub-swaps, sharper result, k² slower). */
    val pixelBoost: Int = 256,
    val detectorSize: Int = 640,
    val minDetectionScore: Float = 0.5f,
    val minFaceSize: Int = 24,
    val maskShape: MaskShape = MaskShape.BOX,
    val maskBlur: Float = 0.3f,
    /** top, right, bottom, left as fraction of crop. */
    val maskPadding: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
    val swapOpacity: Float = 1f,
    val colorMatch: Float = 0f,
    val sharpen: Float = 0f,
    val enhance: Boolean = false,
    val enhanceBlend: Float = 0.8f,
)

data class SwapResult(
    val image: RgbImage,
    val facesFound: Int,
    val facesSwapped: Int,
    val decision: PolicyDecision,
    val millis: Long,
)

/**
 * Lightweight tracker for video: associates detections across frames by IoU and smooths landmarks with an
 * exponential moving average, which removes most frame-to-frame jitter/flicker.
 */
class FaceTracker(private val smoothing: Float = 0.55f) {
    private class Track(var face: Face, var misses: Int = 0, var embedding: FloatArray? = null)
    private val tracks = ArrayList<Track>()

    fun update(faces: List<Face>): List<Face> {
        val out = ArrayList<Face>()
        val used = HashSet<Track>()
        for (f in faces) {
            val best = tracks.filter { it !in used }.maxByOrNull { Geometry.iou(it.face, f) }
            if (best != null && Geometry.iou(best.face, f) > 0.3f) {
                val a = smoothing
                val lm = FloatArray(10) { best.face.landmarks[it] * a + f.landmarks[it] * (1 - a) }
                // big motion -> trust the new detection more
                val move = kotlin.math.abs(best.face.centerX - f.centerX) + kotlin.math.abs(best.face.centerY - f.centerY)
                val sm = if (move > f.width * 0.15f) f.landmarks else lm
                val nf = Face(f.x1, f.y1, f.x2, f.y2, sm, f.score).also { it.embedding = best.embedding }
                best.face = nf; best.misses = 0; used += best
                out += nf
            } else {
                val t = Track(f); tracks += t; used += t; out += f
            }
        }
        tracks.forEach { if (it !in used) it.misses++ }
        tracks.removeAll { it.misses > 5 }
        return out
    }

    /** Cache identity embedding on the track so recognition only runs once per tracked person. */
    fun remember(face: Face, embedding: FloatArray) {
        tracks.firstOrNull { it.face === face }?.embedding = embedding
    }

    fun reset() = tracks.clear()
}

class ModelFiles(
    val detector: File,
    val recognizer: File,
    val swapper: File,
    val contentAnalyser: File,
    val enhancer: File? = null,
    val emapCache: File? = null,
)

class FaceSwapEngine(
    files: ModelFiles,
    factory: SessionFactory,
    verifier: IdentityVerifier = NoIdentityVerifier,
) : Closeable {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    val detector = FaceDetector(env, files.detector, factory)
    val recognizer = FaceRecognizer(env, files.recognizer, factory)
    // Swapper (~280 MB) and content analyser are loaded on first use, so face import / library work only
    // needs the detector + recognizer in memory.
    private val swapperLazy = lazy { FaceSwapper(env, files.swapper, factory, files.emapCache) }
    val swapper: FaceSwapper by swapperLazy
    val policy = ContentPolicy({ ContentAnalyser(env, files.contentAnalyser, factory) }, verifier)
    private var enhancerFile = files.enhancer
    private var enhancerModel: FaceEnhancer? = null
    private val factoryRef = factory

    @Synchronized
    private fun enhancer(): FaceEnhancer? {
        if (enhancerModel == null && enhancerFile?.exists() == true) enhancerModel = FaceEnhancer(env, enhancerFile!!, factoryRef)
        return enhancerModel
    }

    val hasEnhancer get() = enhancerFile?.exists() == true

    fun detect(img: RgbImage, config: SwapConfig = SwapConfig()): List<Face> =
        detector.detect(img, config.detectorSize, config.minDetectionScore)
            .filter { it.width >= config.minFaceSize && it.height >= config.minFaceSize }
            .sortedBy { it.x1 }

    fun embed(img: RgbImage, face: Face): FloatArray = face.embedding ?: recognizer.embed(img, face).also { face.embedding = it }

    /** Builds a [SourceFace] from (image, face) pairs. */
    fun buildSource(id: String, label: String, samples: List<Pair<RgbImage, Face>>): SourceFace =
        SourceFace(id, label, samples.map { (img, f) -> embed(img, f) })

    private fun latentFor(src: SourceFace): FloatArray = src.latent ?: swapper.latent(src.averaged).also { src.latent = it }

    /** Decides which target faces receive which source. */
    fun plan(img: RgbImage, faces: List<Face>, mappings: List<FaceMapping>, config: SwapConfig, tracker: FaceTracker? = null): List<Pair<Face, SourceFace>> {
        if (faces.isEmpty() || mappings.isEmpty()) return emptyList()
        val referenced = mappings.filter { it.reference != null }
        if (referenced.isNotEmpty()) {
            val result = ArrayList<Pair<Face, SourceFace>>()
            for (f in faces) {
                val e = embed(img, f)
                tracker?.remember(f, e)
                val best = referenced.maxByOrNull { Geometry.cosine(e, it.reference!!) }!!
                if (Geometry.cosine(e, best.reference!!) >= config.referenceThreshold) result += f to best.source
            }
            return result
        }
        val src = mappings.first().source
        val chosen = when (config.selector) {
            FaceSelector.ALL -> faces
            FaceSelector.LARGEST -> listOf(faces.maxBy { it.area })
            FaceSelector.LEFTMOST -> listOf(faces.minBy { it.centerX })
            FaceSelector.RIGHTMOST -> listOf(faces.maxBy { it.centerX })
            FaceSelector.CENTER -> listOf(faces.minBy { kotlin.math.abs(it.centerX - img.width / 2f) + kotlin.math.abs(it.centerY - img.height / 2f) })
        }
        return chosen.map { it to src }
    }

    /**
     * Full pipeline on one image / frame.
     * [explicitOverride] lets video callers reuse a cached content check instead of analysing every frame.
     */
    fun process(
        target: RgbImage,
        mappings: List<FaceMapping>,
        config: SwapConfig,
        tracker: FaceTracker? = null,
        explicitOverride: Boolean? = null,
    ): SwapResult {
        val t0 = System.currentTimeMillis()
        val explicit = explicitOverride ?: policy.isExplicit(target)
        val decision = policy.evaluate(explicit, mappings.map { it.source }.distinct())
        if (!decision.allowed) return SwapResult(target, 0, 0, decision, System.currentTimeMillis() - t0)

        var faces = detect(target, config)
        if (tracker != null) faces = tracker.update(faces)
        val plan = plan(target, faces, mappings, config, tracker)
        val out = target.copy()
        for ((face, src) in plan) swapFace(out, target, face, latentFor(src), config)
        if (config.enhance) enhancer()?.let { enh -> for ((face, _) in plan) enhanceFace(out, face, enh, config) }
        return SwapResult(out, faces.size, plan.size, decision, System.currentTimeMillis() - t0)
    }

    private fun swapFace(out: RgbImage, original: RgbImage, face: Face, latent: FloatArray, config: SwapConfig) {
        val size = (config.pixelBoost / 128).coerceIn(1, 4) * 128
        val k = size / 128
        val m = Geometry.estimateSimilarity(face.landmarks, Templates.ARCFACE_128, size.toFloat())
        val crop = ImageOps.warpAffine(original, m, size, size)
        var swapped = if (k == 1) swapper.swap(crop, latent) else {
            // pixel boost: interleaved sub-sampling into k² 128px tiles, swap each, re-interleave
            val result = RgbImage(size, size)
            for (ti in 0 until k) for (tj in 0 until k) {
                val tile = RgbImage(128, 128)
                for (y in 0 until 128) for (x in 0 until 128) tile.pixels[y * 128 + x] = crop.pixels[(y * k + ti) * size + (x * k + tj)]
                val s = swapper.swap(tile, latent)
                for (y in 0 until 128) for (x in 0 until 128) result.pixels[(y * k + ti) * size + (x * k + tj)] = s.pixels[y * 128 + x]
            }
            result
        }
        val mask = when (config.maskShape) {
            MaskShape.BOX -> ImageOps.boxMask(size, size, config.maskBlur, config.maskPadding)
            MaskShape.ELLIPSE -> ImageOps.ellipseMask(size, size, config.maskBlur)
            MaskShape.BOX_AND_ELLIPSE -> ImageOps.multiplyMasks(
                ImageOps.boxMask(size, size, config.maskBlur, config.maskPadding), ImageOps.ellipseMask(size, size, config.maskBlur),
            )
        }
        if (config.colorMatch > 0f) swapped = ImageOps.colorTransfer(swapped, crop, mask, config.colorMatch)
        if (config.sharpen > 0f) swapped = ImageOps.sharpen(swapped, config.sharpen)
        ImageOps.pasteBack(out, swapped, mask, m, config.swapOpacity)
    }

    private fun enhanceFace(out: RgbImage, face: Face, enh: FaceEnhancer, config: SwapConfig) {
        val m = Geometry.estimateSimilarity(face.landmarks, Templates.FFHQ_512, enh.size.toFloat())
        val crop = ImageOps.warpAffine(out, m, enh.size, enh.size)
        val restored = enh.enhance(crop)
        val mixed = ImageOps.mix(crop, restored, config.enhanceBlend)
        ImageOps.pasteBack(out, mixed, ImageOps.boxMask(enh.size, enh.size, 0.3f, floatArrayOf(0.05f, 0.05f, 0.05f, 0.05f)), m)
    }

    override fun close() {
        detector.close(); recognizer.close(); if (swapperLazy.isInitialized()) swapper.close(); policy.close(); enhancerModel?.close()
    }
}
