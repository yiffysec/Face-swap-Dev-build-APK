package com.faceswap.core

import java.io.Closeable
import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

// =====================================================================================================================
// Detectors
// =====================================================================================================================

interface FaceDetector : Closeable {
    fun detect(img: RgbImage, requestedSize: Int = 640, scoreThreshold: Float = 0.5f, nmsThreshold: Float = 0.4f): List<Face>
}

/** Letterboxes [img] into a square of [size] (top-left aligned, black padding). Returns (image, scale). */
private fun letterbox(img: RgbImage, size: Int): Pair<RgbImage, Float> {
    val ratio = min(size.toFloat() / img.width, size.toFloat() / img.height)
    val scaled = ImageOps.warpAffine(img, Affine.scale(ratio), size, size)
    val sw = (img.width * ratio).toInt(); val sh = (img.height * ratio).toInt()
    for (y in 0 until size) for (x in 0 until size) if (x >= sw || y >= sh) scaled.pixels[y * size + x] = 0xFF000000.toInt()
    return scaled to ratio
}

private fun fixedSquare(model: InferenceModel): Int? = model.inputShape(model.inputNames[0])?.let { s ->
    if (s.size == 4 && s[2] > 0) s[2].toInt() else null
}

/** SCRFD (InsightFace) – outputs scores×3, boxes×3, keypoints×3 for strides 8/16/32. */
class ScrfdDetector(private val model: InferenceModel) : FaceDetector {
    private val fixedSize = fixedSquare(model)

    override fun detect(img: RgbImage, requestedSize: Int, scoreThreshold: Float, nmsThreshold: Float): List<Face> {
        val size = fixedSize ?: requestedSize
        val (lb, ratio) = letterbox(img, size)
        val out = model.run(mapOf(model.inputNames[0] to Tensor(ImageOps.toChw(lb, 127.5f, 128f), 1, 3, size, size))).values.toList()
        val faces = ArrayList<Face>()
        for ((si, stride) in intArrayOf(8, 16, 32).withIndex()) {
            val scores = out[si].data; val boxes = out[si + 3].data; val kps = out[si + 6].data
            val fw = size / stride
            val anchors = max(1, scores.size / (fw * fw))
            for (i in scores.indices) {
                val s = scores[i]
                if (s < scoreThreshold) continue
                val loc = i / anchors
                val cx = (loc % fw) * stride.toFloat(); val cy = (loc / fw) * stride.toFloat()
                val lm = FloatArray(10) { k -> ((if (k % 2 == 0) cx else cy) + kps[i * 10 + k] * stride) / ratio }
                faces += Face(
                    (cx - boxes[i * 4] * stride) / ratio, (cy - boxes[i * 4 + 1] * stride) / ratio,
                    (cx + boxes[i * 4 + 2] * stride) / ratio, (cy + boxes[i * 4 + 3] * stride) / ratio, lm, s,
                )
            }
        }
        return Geometry.nms(faces, nmsThreshold)
    }

    override fun close() = model.close()
}

/** YuNet (OpenCV, MIT, 230 KB) – BGR 0..255 input; cls×3, obj×3, bbox×3, kps×3. ~8× faster than SCRFD. */
class YunetDetector(private val model: InferenceModel) : FaceDetector {
    private val fixedSize = fixedSquare(model)

    override fun detect(img: RgbImage, requestedSize: Int, scoreThreshold: Float, nmsThreshold: Float): List<Face> {
        val size = fixedSize ?: requestedSize
        val (lb, ratio) = letterbox(img, size)
        val input = ImageOps.toChw(lb, 0f, 1f, bgr = true)
        val out = model.run(mapOf(model.inputNames[0] to Tensor(input, 1, 3, size, size))).values.toList()
        val faces = ArrayList<Face>()
        for ((si, stride) in intArrayOf(8, 16, 32).withIndex()) {
            val cls = out[si].data; val obj = out[si + 3].data; val bbox = out[si + 6].data; val kps = out[si + 9].data
            val fw = size / stride
            for (j in cls.indices) {
                val s = cls[j] * obj[j]
                if (s < scoreThreshold) continue
                val ax = (j % fw) * stride.toFloat(); val ay = (j / fw) * stride.toFloat()
                val cx = bbox[j * 4] * stride + ax; val cy = bbox[j * 4 + 1] * stride + ay
                val bw = exp(bbox[j * 4 + 2]) * stride; val bh = exp(bbox[j * 4 + 3]) * stride
                val lm = FloatArray(10) { k -> (kps[j * 10 + k] * stride + (if (k % 2 == 0) ax else ay)) / ratio }
                faces += Face((cx - bw / 2) / ratio, (cy - bh / 2) / ratio, (cx + bw / 2) / ratio, (cy + bh / 2) / ratio, lm, s)
            }
        }
        return Geometry.nms(faces, nmsThreshold)
    }

    override fun close() = model.close()
}

// =====================================================================================================================
// 68-point landmarks (HRFFA, MIT)
// =====================================================================================================================

class Landmarker68(private val model: InferenceModel) : Closeable {
    private val size = 256

    /** Returns 136 values (x,y × 68) in image coordinates. */
    fun detect(img: RgbImage, face: Face): FloatArray {
        val scale = size / (max(face.width, face.height).coerceAtLeast(1f) * 1.7f)
        val tx = (size - (face.x1 + face.x2) * scale) * 0.5f
        val ty = (size - (face.y1 + face.y2) * scale) * 0.5f
        val m = Affine(scale, 0f, tx, 0f, scale, ty)
        val crop = ImageOps.warpAffine(img, m, size, size)
        val out = model.run(mapOf(model.inputNames[0] to Tensor(ImageOps.toChw(crop, 0.5f, 0.5f, scale = 1f / 255f), 1, 3, size, size)))
        val lm = (out["landmarks"] ?: out.first()).data
        val pts = FloatArray(136) { lm[it] * size }
        return Geometry.transformPoints(pts, m.invert())
    }

    override fun close() = model.close()
}

// =====================================================================================================================
// Recognition (ArcFace w600k_r50)
// =====================================================================================================================

class FaceRecognizer(private val model: InferenceModel) : Closeable {
    fun embed(img: RgbImage, face: Face): FloatArray {
        val m = Geometry.estimateSimilarity(face.landmarks, Templates.ARCFACE_112, 112f)
        val crop = ImageOps.warpAffine(img, m, 112, 112)
        return Geometry.normalize(model.run(mapOf(model.inputNames[0] to Tensor(ImageOps.toChw(crop, 127.5f, 127.5f), 1, 3, 112, 112))).first().data)
    }

    override fun close() = model.close()
}

// =====================================================================================================================
// Swappers
// =====================================================================================================================

/** Result of one swap pass on an aligned crop. [mask] is the model's own face mask (0..1, crop-sized) if it has one. */
class SwapOutput(val image: RgbImage, val mask: FloatArray?)

interface FaceSwapModel : Closeable {
    val id: String
    /** Native model resolution (pixel boost uses multiples of it). */
    val size: Int
    val template: FloatArray
    /** Per-identity source representation (computed once per source and cached). */
    fun sourceLatent(embedding: FloatArray): FloatArray
    fun swap(crop: RgbImage, latent: FloatArray): SwapOutput
}

/** InSwapper 128 (InsightFace). latent = normalize(embedding · emap). */
class InSwapper(private val model: InferenceModel, file: File, emapCache: File?) : FaceSwapModel {
    override val id = file.nameWithoutExtension
    override val size = 128
    override val template = Templates.ARCFACE_128
    private val emap: FloatArray = loadEmap(file, emapCache)
    private val targetName = model.inputNames.firstOrNull { it.contains("target") } ?: model.inputNames[0]
    private val sourceName = model.inputNames.firstOrNull { it.contains("source") } ?: model.inputNames[1]

    private fun loadEmap(file: File, cache: File?): FloatArray {
        if (cache != null && cache.length() == 512L * 512 * 4) {
            val bb = java.nio.ByteBuffer.wrap(cache.readBytes()).order(java.nio.ByteOrder.LITTLE_ENDIAN)
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

    override fun sourceLatent(embedding: FloatArray): FloatArray {
        val out = FloatArray(512)
        for (i in 0 until 512) {
            val e = embedding[i]
            if (e == 0f) continue
            val row = i * 512
            for (j in 0 until 512) out[j] += e * emap[row + j]
        }
        return Geometry.normalize(out)
    }

    override fun swap(crop: RgbImage, latent: FloatArray): SwapOutput {
        val out = model.run(mapOf(
            targetName to Tensor(ImageOps.toChw(crop, 0f, 1f, scale = 1f / 255f), 1, 3, 128, 128),
            sourceName to Tensor(latent, 1, 512),
        ))
        return SwapOutput(ImageOps.fromChw(out.first().data, 128, 128, 0f, 1f, scale = 1f / 255f), null)
    }

    override fun close() = model.close()
}

/** HyperSwap 1a/1b/1c (FaceFusion, 256px). Source = normalised ArcFace embedding. Outputs image + face mask. */
class HyperSwap(private val model: InferenceModel, file: File) : FaceSwapModel {
    override val id = file.nameWithoutExtension
    override val size = 256
    override val template = Templates.ARCFACE_128

    override fun sourceLatent(embedding: FloatArray) = Geometry.normalize(embedding)

    override fun swap(crop: RgbImage, latent: FloatArray): SwapOutput {
        val out = model.run(mapOf(
            "target" to Tensor(ImageOps.toChw(crop, 0.5f, 0.5f, scale = 1f / 255f), 1, 3, 256, 256),
            "source" to Tensor(latent, 1, 512),
        ))
        val img = out["output"] ?: out.first()
        val mask = out["mask"]?.data?.let { m -> FloatArray(m.size) { m[it].coerceIn(0f, 1f) } }
        return SwapOutput(ImageOps.fromChw(img.data, 256, 256, 0.5f, 0.5f, scale = 1f / 255f), mask)
    }

    override fun close() = model.close()
}

/**
 * MobileFaceSwap (Baidu, AAAI-22, Apache-2.0): a weight builder runs once per identity, then a 364 KB
 * identity-specific network runs per frame (~15× faster than HyperSwap). Lower fidelity – used for live preview.
 * Needs its own ArcFace (ImageNet-normalised, also returns a 7×7 feature map).
 */
class MobileFaceSwap(
    private val runtime: InferenceModel,
    private val builder: InferenceModel,
    private val arcface: InferenceModel,
) : Closeable {
    val size = 224
    val template = Templates.MOBILEFACESWAP

    /** Identity weights for one source face (18 tensors). */
    class Identity(internal val weights: Map<String, Tensor>)

    fun buildIdentity(img: RgbImage, face: Face): Identity {
        val m = Geometry.estimateSimilarity(face.landmarks, template, size.toFloat())
        val aligned = ImageOps.resize(ImageOps.warpAffine(img, m, size, size), 112, 112)
        val n = 112 * 112
        val x = FloatArray(3 * n)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f); val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (i in 0 until n) {
            val p = aligned.pixels[i]
            x[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            x[n + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            x[2 * n + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
        val a = arcface.run(mapOf(arcface.inputNames[0] to Tensor(x, 1, 3, 112, 112))).values.toList()
        val emb = Geometry.normalize(a[0].data)
        val w = builder.run(mapOf("id_emb" to Tensor(emb, 1, 512), "id_feature" to a[1])).values.toList()
        val names = runtime.inputNames.filter { it != "att_img" }
        return Identity(names.zip(w).toMap())
    }

    fun swap(crop: RgbImage, id: Identity): SwapOutput {
        val feeds = HashMap(id.weights)
        feeds["att_img"] = Tensor(ImageOps.toChw(crop, 0f, 1f, scale = 1f / 255f), 1, 3, size, size)
        val out = runtime.run(feeds).values.toList()
        return SwapOutput(ImageOps.fromChw(out[0].data, size, size, 0f, 1f, scale = 1f / 255f), MaskOps.binaryFeather(out[1].data, size, size))
    }

    override fun close() { runtime.close(); builder.close(); arcface.close() }
}

// =====================================================================================================================
// Face parsing (BiSeNet ResNet-18, MIT) → region masks
// =====================================================================================================================

enum class FaceRegion(val label: Int) {
    SKIN(1), LEFT_EYEBROW(2), RIGHT_EYEBROW(3), LEFT_EYE(4), RIGHT_EYE(5), GLASSES(6), NOSE(10), MOUTH(11), UPPER_LIP(12), LOWER_LIP(13);

    companion object { val DEFAULT = entries.toSet() }
}

class FaceParser(private val model: InferenceModel) : Closeable {
    private val size = 512

    /** Mask (crop-sized, 0..1) of pixels belonging to [regions]. Anything else (hair, hands, objects) is excluded. */
    fun regionMask(crop: RgbImage, regions: Set<FaceRegion>): FloatArray {
        val small = if (crop.width == size) crop else ImageOps.resize(crop, size, size)
        val n = size * size
        val x = FloatArray(3 * n)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f); val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (i in 0 until n) {
            val p = small.pixels[i]
            x[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            x[n + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            x[2 * n + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
        val logits = model.run(mapOf(model.inputNames[0] to Tensor(x, 1, 3, size, size))).first().data
        val classes = logits.size / n
        val wanted = BooleanArray(classes).also { w -> regions.forEach { if (it.label < classes) w[it.label] = true } }
        val m = FloatArray(n)
        for (i in 0 until n) {
            var best = 0; var bv = logits[i]
            for (c in 1 until classes) { val v = logits[c * n + i]; if (v > bv) { bv = v; best = c } }
            m[i] = if (wanted[best]) 1f else 0f
        }
        val resized = if (crop.width == size) m else MaskOps.resize(m, size, size, crop.width, crop.height)
        return MaskOps.softenHard(resized, crop.width, crop.height, sigma = 5f * crop.width / 512f)
    }

    override fun close() = model.close()
}

// =====================================================================================================================
// Face restoration (GFPGAN / GPEN / RestoreFormer++ / CodeFormer)
// =====================================================================================================================

class FaceRestorer(private val model: InferenceModel, val id: String) : Closeable {
    val size: Int = model.inputShape(model.inputNames[0])?.getOrNull(2)?.toInt()?.takeIf { it > 0 } ?: 512
    /** GPEN-256 was trained on ArcFace-aligned crops, the others on FFHQ. */
    val template: FloatArray = if (id.startsWith("gpen_bfr_256")) Templates.ARCFACE_128 else Templates.FFHQ_512
    private val hasWeight = "weight" in model.inputNames

    /** [fidelity] only affects CodeFormer (0 = more restoration, 1 = closer to input). */
    fun enhance(crop: RgbImage, fidelity: Float = 0.5f): RgbImage {
        val feeds = HashMap<String, Tensor>()
        feeds[model.inputNames.first { it != "weight" }] = Tensor(ImageOps.toChw(crop, 0.5f, 0.5f, scale = 1f / 255f), 1, 3, size, size)
        if (hasWeight) feeds["weight"] = Tensor.scalar(fidelity)
        val out = model.run(feeds).first().data
        for (i in out.indices) out[i] = out[i].coerceIn(-1f, 1f)
        return ImageOps.fromChw(out, size, size, 0.5f, 0.5f, scale = 1f / 255f)
    }

    override fun close() = model.close()
}

// =====================================================================================================================
// Content analyser (open_nsfw)
// =====================================================================================================================

class ContentAnalyser(private val model: InferenceModel) : Closeable {
    fun explicitProbability(img: RgbImage): Float {
        val small = ImageOps.resize(img, 224, 224)
        val data = FloatArray(224 * 224 * 3)
        for (i in 0 until 224 * 224) { // NHWC, BGR, mean-subtracted
            val p = small.pixels[i]
            data[i * 3] = (p and 0xFF) - 104f
            data[i * 3 + 1] = ((p shr 8) and 0xFF) - 117f
            data[i * 3 + 2] = ((p shr 16) and 0xFF) - 123f
        }
        return max(0f, model.run(mapOf(model.inputNames[0] to Tensor(data, 1, 224, 224, 3))).first().data[1])
    }

    override fun close() = model.close()
}
