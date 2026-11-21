package com.faceswap.core

import java.io.Closeable
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Ports of FaceFusion's extra processors: age modifier (FRAN), background remover (MODNet / U²-Netp),
 * frame enhancer (Real-ESRGAN / SPAN, tiled), expression restorer + face editor (LivePortrait).
 */
class Processors(
    private val has: (String) -> Boolean,
    private val load: (String) -> InferenceModel,
) : Closeable {
    private val models = ConcurrentHashMap<String, InferenceModel>()
    private fun m(id: String) = models.getOrPut(id) { load(id) }

    val hasLivePortrait get() = has(ModelIds.LP_FEATURE) && has(ModelIds.LP_MOTION) && has(ModelIds.LP_GENERATOR)
    val hasAge get() = has(ModelIds.FRAN)

    // =================================================================================================== age (FRAN)

    /**
     * Re-ages one face. [currentAge] is the person's approximate age, [targetAge] the desired one (both years).
     * FRAN works on a 1024² FFHQ-aligned crop.
     */
    fun modifyAge(img: RgbImage, face: Face, currentAge: Float, targetAge: Float, blur: Float = 0.3f): RgbImage {
        val size = 1024
        val mat = Geometry.estimateSimilarity(face.landmarks, Templates.FFHQ_512, size.toFloat())
        val crop = ImageOps.warpAffine(img, mat, size, size)
        val dir = floatArrayOf((currentAge / 100f).coerceIn(0f, 1f), (targetAge / 100f).coerceIn(0f, 1f))
        val model = m(ModelIds.FRAN)
        val out = model.run(mapOf(
            "target" to Tensor(ImageOps.toChw(crop, 0f, 1f, scale = 1f / 255f), 1, 3, size, size),
            "direction" to Tensor(dir, 2),
        )).first()
        val aged = ImageOps.fromChw(out.data, size, size, 0f, 1f, scale = 1f / 255f)
        val result = img.copy()
        ImageOps.pasteBack(result, aged, ImageOps.boxMask(size, size, blur), mat)
        return result
    }

    // =================================================================================================== background

    sealed class Background {
        data class Blur(val strength: Float = 0.6f) : Background()
        data class Color(val argb: Int) : Background()
        object Transparent : Background()
        data class Image(val image: RgbImage) : Background()
    }

    /** Foreground alpha (0..1, image-sized). MODNet is portrait-matting quality; U²-Netp is the 4.6 MB fallback. */
    fun foregroundMask(img: RgbImage): FloatArray {
        val useModnet = has(ModelIds.MODNET)
        val id = if (useModnet) ModelIds.MODNET else ModelIds.U2NETP
        val size = if (useModnet) 512 else 320
        val small = ImageOps.resize(img, size, size)
        val input = if (useModnet) ImageOps.toChw(small, 0.5f, 0.5f, scale = 1f / 255f) else imagenet(small)
        val model = m(id)
        val out = model.run(mapOf(model.inputNames[0] to Tensor(input, 1, 3, size, size))).first().data
        val mask = FloatArray(size * size) { out[it].coerceIn(0f, 1f) }
        return MaskOps.resize(mask, size, size, img.width, img.height)
    }

    fun replaceBackground(img: RgbImage, bg: Background): RgbImage {
        val alpha = foregroundMask(img)
        val out = RgbImage(img.width, img.height)
        val backdrop: RgbImage? = when (bg) {
            is Background.Blur -> blurImage(img, bg.strength)
            is Background.Image -> if (bg.image.width == img.width && bg.image.height == img.height) bg.image else coverResize(bg.image, img.width, img.height)
            else -> null
        }
        for (i in img.pixels.indices) {
            val a = alpha[i]
            out.pixels[i] = when (bg) {
                is Background.Transparent -> ((a * 255).roundToInt().coerceIn(0, 255) shl 24) or (img.pixels[i] and 0xFFFFFF)
                is Background.Color -> ImageOps.blend(bg.argb or (0xFF shl 24), img.pixels[i], a)
                else -> ImageOps.blend(backdrop!!.pixels[i], img.pixels[i], a)
            }
        }
        return out
    }

    private fun coverResize(src: RgbImage, w: Int, h: Int): RgbImage {
        val s = max(w.toFloat() / src.width, h.toFloat() / src.height)
        val tx = (w - src.width * s) / 2; val ty = (h - src.height * s) / 2
        return ImageOps.warpAffine(src, Affine(s, 0f, tx, 0f, s, ty), w, h)
    }

    /** Cheap large-radius blur: downscale → gaussian → upscale. */
    private fun blurImage(img: RgbImage, strength: Float): RgbImage {
        val f = (8 * strength.coerceIn(0.1f, 1f)).coerceAtLeast(2f)
        val sw = max(8, (img.width / f).toInt()); val sh = max(8, (img.height / f).toInt())
        val small = ImageOps.resize(img, sw, sh)
        val chans = Array(3) { c -> FloatArray(sw * sh) { ((small.pixels[it] shr (16 - 8 * c)) and 0xFF).toFloat() } }
        val blurred = chans.map { MaskOps.gaussian(it, sw, sh, 3f) }
        val bs = RgbImage(sw, sh)
        for (i in bs.pixels.indices) bs.pixels[i] = (0xFF shl 24) or
            (blurred[0][i].roundToInt().coerceIn(0, 255) shl 16) or (blurred[1][i].roundToInt().coerceIn(0, 255) shl 8) or blurred[2][i].roundToInt().coerceIn(0, 255)
        return ImageOps.resize(bs, img.width, img.height)
    }

    private fun imagenet(img: RgbImage): FloatArray {
        val n = img.width * img.height
        val x = FloatArray(3 * n)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f); val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (i in 0 until n) {
            val p = img.pixels[i]
            x[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            x[n + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            x[2 * n + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
        return x
    }

    // =================================================================================================== frame enhancer

    fun frameScale(id: String) = when {
        id.contains("x8") -> 8; id.contains("x4") -> 4; id.contains("x2") -> 2; else -> 4
    }

    /**
     * Tiled super-resolution. Each tile is processed with a [margin] of context that is discarded afterwards, so tile
     * seams never show. [blend] 1 = fully enhanced, 0 = plain upscale.
     */
    fun enhanceFrame(img: RgbImage, id: String, blend: Float = 1f, onProgress: (Float) -> Unit = {}): RgbImage {
        val model = m(id)
        val scale = frameScale(id)
        val fixed = model.inputShape(model.inputNames[0])?.getOrNull(2)?.toInt()?.takeIf { it > 0 }
        val tile = fixed ?: if (id.startsWith("span")) 128 else 256
        val margin = if (tile <= 128) 8 else 16
        val core = tile - 2 * margin
        val out = RgbImage(img.width * scale, img.height * scale)
        val tilesX = (img.width + core - 1) / core; val tilesY = (img.height + core - 1) / core
        var done = 0
        for (ty in 0 until tilesY) for (tx in 0 until tilesX) {
            val x0 = tx * core - margin; val y0 = ty * core - margin
            val t = RgbImage(tile, tile)
            for (y in 0 until tile) for (x in 0 until tile) {
                val sx = (x0 + x).coerceIn(0, img.width - 1); val sy = (y0 + y).coerceIn(0, img.height - 1)
                t.pixels[y * tile + x] = img.pixels[sy * img.width + sx]
            }
            val r = model.run(mapOf(model.inputNames[0] to Tensor(ImageOps.toChw(t, 0f, 1f, scale = 1f / 255f), 1, 3, tile, tile))).first()
            val ts = tile * scale
            val up = ImageOps.fromChw(r.data, ts, ts, 0f, 1f, scale = 1f / 255f)
            for (y in margin * scale until (margin + core) * scale) {
                val oy = (y0 * scale + y); if (oy < 0 || oy >= out.height) continue
                for (x in margin * scale until (margin + core) * scale) {
                    val ox = (x0 * scale + x); if (ox < 0 || ox >= out.width) continue
                    out.pixels[oy * out.width + ox] = up.pixels[y * ts + x]
                }
            }
            done++; onProgress(done.toFloat() / (tilesX * tilesY))
        }
        if (blend < 0.999f) {
            val plain = ImageOps.resize(img, out.width, out.height)
            return ImageOps.mix(plain, out, blend)
        }
        return out
    }

    // =================================================================================================== LivePortrait

    private class Motion(val pitch: Float, val yaw: Float, val roll: Float, val scale: Float, val translation: FloatArray, val expression: FloatArray, val points: FloatArray)

    private fun lpInput(crop512: RgbImage): Tensor {
        val small = ImageOps.resize(crop512, 256, 256)
        return Tensor(ImageOps.toChw(small, 0f, 1f, scale = 1f / 255f), 1, 3, 256, 256)
    }

    private fun motion(input: Tensor): Motion {
        val o = m(ModelIds.LP_MOTION).run(mapOf("input" to input))
        return Motion(o["pitch"]!!.scalar(), o["yaw"]!!.scalar(), o["roll"]!!.scalar(), o["scale"]!!.scalar(),
            o["translation"]!!.data, o["expression"]!!.data, o["motion_points"]!!.data)
    }

    /** scipy Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True) – extrinsic: Rz·Ry·Rx. */
    private fun rotation(pitch: Float, yaw: Float, roll: Float): FloatArray {
        val p = Math.toRadians(pitch.toDouble()); val y = Math.toRadians(yaw.toDouble()); val r = Math.toRadians(roll.toDouble())
        val cx = cos(p); val sx = sin(p); val cy = cos(y); val sy = sin(y); val cz = cos(r); val sz = sin(r)
        return floatArrayOf(
            (cz * cy).toFloat(), (cz * sy * sx - sz * cx).toFloat(), (cz * sy * cx + sz * sx).toFloat(),
            (sz * cy).toFloat(), (sz * sy * sx + cz * cx).toFloat(), (sz * sy * cx - cz * sx).toFloat(),
            (-sy).toFloat(), (cy * sx).toFloat(), (cy * cx).toFloat(),
        )
    }

    /** scale * (points · Rᵀ + expression) + translation, for 21 keypoints. */
    private fun keypoints(mo: Motion, rot: FloatArray, expression: FloatArray): FloatArray = FloatArray(63).also { out ->
        for (k in 0 until 21) {
            val x = mo.points[3 * k]; val y = mo.points[3 * k + 1]; val z = mo.points[3 * k + 2]
            for (r in 0..2) {
                val v = x * rot[3 * r] + y * rot[3 * r + 1] + z * rot[3 * r + 2] + expression[3 * k + r]
                out[3 * k + r] = mo.scale * v + mo.translation[r]
            }
        }
    }

    private fun generate(feature: Tensor, source: FloatArray, target: FloatArray): RgbImage {
        val o = m(ModelIds.LP_GENERATOR).run(mapOf("feature_volume" to feature, "source" to Tensor(source, 1, 21, 3), "target" to Tensor(target, 1, 21, 3)))
        val d = o.first().data
        for (i in d.indices) d[i] = d[i].coerceIn(0f, 1f)
        return ImageOps.fromChw(d, 512, 512, 0f, 1f, scale = 1f / 255f)
    }

    /**
     * Expression restorer: after swapping, re-applies the ORIGINAL target's expression (mouth/eyes) to the swapped face,
     * fixing the "neutral mouth" look swappers produce.
     */
    fun restoreExpression(original: RgbImage, swapped: RgbImage, face: Face, factor: Float, areas: Set<MaskOps.Area>, blur: Float) {
        val size = 512
        val mat = Geometry.estimateSimilarity(face.landmarks, Templates.ARCFACE_128, size.toFloat())
        val targetCrop = ImageOps.warpAffine(original, mat, size, size)
        val tempCrop = ImageOps.warpAffine(swapped, mat, size, size)
        val tempIn = lpInput(tempCrop)
        val feature = m(ModelIds.LP_FEATURE).run(mapOf("input" to tempIn)).first()
        val targetExp = motion(lpInput(targetCrop)).expression.copyOf()
        val mo = motion(tempIn)
        val rot = rotation(mo.pitch, mo.yaw, mo.roll)
        val tempExp = mo.expression
        // restrict to chosen areas (indices per FaceFusion)
        val upper = intArrayOf(1, 2, 6, 10, 11, 12, 13, 15, 16); val lower = intArrayOf(3, 7, 14, 17, 18, 19, 20); val always = intArrayOf(0, 4, 5, 8, 9)
        fun keep(idx: IntArray) { for (k in idx) for (c in 0..2) targetExp[3 * k + c] = tempExp[3 * k + c] }
        if (MaskOps.Area.UPPER_FACE !in areas) keep(upper)
        if (MaskOps.Area.LOWER_FACE !in areas && MaskOps.Area.MOUTH !in areas) keep(lower)
        keep(always)
        val f = (factor * 1.2f).coerceIn(0f, 1.2f)
        val mixed = FloatArray(63) { (targetExp[it] * f + tempExp[it] * (1 - f)).coerceIn(LivePortraitLimits.MIN[it], LivePortraitLimits.MAX[it]) }
        val restored = generate(feature, keypoints(mo, rot, mixed), keypoints(mo, rot, tempExp))
        ImageOps.pasteBack(swapped, restored, ImageOps.boxMask(size, size, blur), mat)
    }

    /**
     * Face editor (port of FaceFusion's face_editor). All values are -1..1 like FaceFusion's sliders.
     */
    data class FaceEdit(
        val eyebrowDirection: Float = 0f, val eyeGazeHorizontal: Float = 0f, val eyeGazeVertical: Float = 0f,
        val eyeOpen: Float = 0f, val lipOpen: Float = 0f, val mouthGrim: Float = 0f, val mouthPout: Float = 0f,
        val mouthPurse: Float = 0f, val mouthSmile: Float = 0f, val mouthHorizontal: Float = 0f, val mouthVertical: Float = 0f,
        val headPitch: Float = 0f, val headYaw: Float = 0f, val headRoll: Float = 0f,
    ) { val isEmpty get() = this == FaceEdit() }

    val hasFaceEditor get() = hasLivePortrait && has(ModelIds.LP_STITCHER)

    fun editFace(img: RgbImage, face: Face, e: FaceEdit, blur: Float = 0.3f): RgbImage {
        val size = 512
        val mat = Geometry.estimateSimilarity(face.landmarks, Templates.FFHQ_512, size.toFloat())
        val crop = ImageOps.warpAffine(img, mat, size, size)
        val inp = lpInput(crop)
        val feature = m(ModelIds.LP_FEATURE).run(mapOf("input" to inp)).first()
        val mo = motion(inp)
        val target = keypoints(mo, rotation(mo.pitch, mo.yaw, mo.roll), mo.expression)
        val x = mo.expression.copyOf()
        fun add(k: Int, c: Int, v: Float) { x[3 * k + c] += v }
        // eye gaze
        if (e.eyeGazeHorizontal > 0) { add(11, 0, 0.015f * e.eyeGazeHorizontal); add(15, 0, 0.020f * e.eyeGazeHorizontal) }
        else { add(11, 0, 0.020f * e.eyeGazeHorizontal); add(15, 0, 0.015f * e.eyeGazeHorizontal) }
        add(1, 1, 0.0025f * e.eyeGazeVertical); add(2, 1, -0.0025f * e.eyeGazeVertical)
        add(11, 1, -0.010f * e.eyeGazeVertical); add(13, 1, -0.005f * e.eyeGazeVertical)
        add(15, 1, -0.010f * e.eyeGazeVertical); add(16, 1, -0.005f * e.eyeGazeVertical)
        // mouth grim
        if (e.mouthGrim > 0) { add(17, 2, -0.005f * e.mouthGrim); add(19, 2, 0.01f * e.mouthGrim); add(20, 1, -0.06f * e.mouthGrim); add(20, 2, -0.03f * e.mouthGrim) }
        else { add(19, 1, -0.05f * e.mouthGrim); add(19, 2, -0.02f * e.mouthGrim); add(20, 2, -0.03f * e.mouthGrim) }
        // mouth position
        add(19, 0, 0.05f * e.mouthHorizontal); add(20, 0, 0.04f * e.mouthHorizontal)
        if (e.mouthVertical > 0) { add(19, 1, -0.04f * e.mouthVertical); add(20, 1, -0.02f * e.mouthVertical) }
        else { add(19, 1, -0.05f * e.mouthVertical); add(20, 1, -0.04f * e.mouthVertical) }
        // pout
        if (e.mouthPout > 0) { add(19, 1, -0.022f * e.mouthPout); add(19, 2, 0.025f * e.mouthPout); add(20, 2, -0.002f * e.mouthPout) }
        else { add(19, 1, 0.022f * e.mouthPout); add(19, 2, 0.025f * e.mouthPout); add(20, 2, -0.002f * e.mouthPout) }
        // purse
        if (e.mouthPurse > 0) { add(19, 1, -0.04f * e.mouthPurse); add(19, 2, -0.02f * e.mouthPurse) }
        else { add(14, 1, -0.02f * e.mouthPurse); add(17, 2, 0.01f * e.mouthPurse); add(19, 2, -0.015f * e.mouthPurse); add(20, 2, -0.002f * e.mouthPurse) }
        // smile
        val sm = e.mouthSmile
        if (sm > 0) { add(20, 1, -0.015f * sm); add(14, 1, -0.025f * sm); add(17, 1, 0.01f * sm); add(17, 2, 0.004f * sm); add(3, 1, -0.0045f * sm); add(7, 1, -0.0045f * sm) }
        else { add(14, 1, -0.02f * sm); add(17, 1, 0.003f * sm); add(19, 1, 0.02f * sm); add(19, 2, -0.005f * sm); add(20, 2, 0.01f * sm); add(3, 1, 0.0045f * sm); add(7, 1, 0.0045f * sm) }
        // eyebrows
        val eb = e.eyebrowDirection
        if (eb > 0) { add(1, 1, 0.015f * eb); add(2, 1, -0.020f * eb) }
        else { add(1, 0, -0.015f * eb); add(2, 0, 0.020f * eb); add(1, 1, 0.005f * eb); add(2, 1, -0.005f * eb) }
        for (i in x.indices) x[i] = x[i].coerceIn(LivePortraitLimits.MIN[i], LivePortraitLimits.MAX[i])
        // head rotation (FaceFusion: pitch ±20, yaw ±60, roll ±15, limited)
        fun lim(v: Float, orig: Float, lo: Float, hi: Float) = v.coerceIn(if (orig < 0) minOf(orig, lo) else lo, if (orig >= 0) maxOf(orig, hi) else hi)
        val rot = rotation(
            lim(mo.pitch - 20f * e.headPitch, mo.pitch, -30f, 30f),
            lim(mo.yaw - 60f * e.headYaw, mo.yaw, -60f, 60f),
            lim(mo.roll + 15f * e.headRoll, mo.roll, -20f, 20f),
        )
        var source = keypoints(mo, rot, x)
        face.landmarks68?.let { l68 ->
            fun ratio(t: Int, b: Int, l: Int, r: Int): Float {
                val v = kotlin.math.hypot(l68[2 * t] - l68[2 * b], l68[2 * t + 1] - l68[2 * b + 1])
                val h = kotlin.math.hypot(l68[2 * l] - l68[2 * r], l68[2 * l + 1] - l68[2 * r + 1])
                return v / (h + 1e-6f)
            }
            if (e.eyeOpen != 0f && has(ModelIds.LP_EYE)) {
                val inp2 = target + floatArrayOf(ratio(37, 40, 39, 36), ratio(43, 46, 45, 42), if (e.eyeOpen < 0) 0f else 0.6f)
                val d = m(ModelIds.LP_EYE).run(mapOf("input" to Tensor(inp2, 1, 66))).first().data
                source = FloatArray(63) { source[it] + d[it] * kotlin.math.abs(e.eyeOpen) }
            }
            if (e.lipOpen != 0f && has(ModelIds.LP_LIP)) {
                val inp2 = target + floatArrayOf(ratio(62, 66, 54, 48), if (e.lipOpen < 0) 0f else 1f)
                val d = m(ModelIds.LP_LIP).run(mapOf("input" to Tensor(inp2, 1, 65))).first().data
                source = FloatArray(63) { source[it] + d[it] * kotlin.math.abs(e.lipOpen) }
            }
        }
        if (has(ModelIds.LP_STITCHER)) {
            source = m(ModelIds.LP_STITCHER).run(mapOf("source" to Tensor(source, 1, 21, 3), "target" to Tensor(target, 1, 21, 3))).first().data
        }
        val edited = generate(feature, source, target)
        val out = img.copy()
        ImageOps.pasteBack(out, edited, ImageOps.boxMask(size, size, blur), mat)
        return out
    }

    override fun close() { models.values.forEach { runCatching { it.close() } }; models.clear() }
}

/** Developer overlay (FaceFusion face_debugger): boxes, 5/68 landmarks, swapped faces highlighted. */
object Debugger {
    fun draw(img: RgbImage, faces: List<Face>, swapped: Set<Face>) {
        for (f in faces) {
            val color = if (f in swapped) 0xFF00E676.toInt() else 0xFFFFC107.toInt()
            rect(img, f.x1.toInt(), f.y1.toInt(), f.x2.toInt(), f.y2.toInt(), color)
            f.landmarks68?.let { l -> for (i in 0 until 68) dot(img, l[2 * i].toInt(), l[2 * i + 1].toInt(), 1, 0xFF40C4FF.toInt()) }
            for (i in 0 until 5) dot(img, f.landmarks[2 * i].toInt(), f.landmarks[2 * i + 1].toInt(), 2, 0xFFFF1744.toInt())
        }
    }

    private fun set(img: RgbImage, x: Int, y: Int, c: Int) { if (x in 0 until img.width && y in 0 until img.height) img.pixels[y * img.width + x] = c }
    private fun dot(img: RgbImage, x: Int, y: Int, r: Int, c: Int) { for (dy in -r..r) for (dx in -r..r) set(img, x + dx, y + dy, c) }
    private fun rect(img: RgbImage, x1: Int, y1: Int, x2: Int, y2: Int, c: Int) {
        val t = max(1, min(img.width, img.height) / 400)
        for (k in 0 until t) {
            for (x in x1..x2) { set(img, x, y1 + k, c); set(img, x, y2 - k, c) }
            for (y in y1..y2) { set(img, x1 + k, y, c); set(img, x2 - k, y, c) }
        }
    }
}

/** LivePortrait expression clip limits (21×3), from FaceFusion. */
internal object LivePortraitLimits {
    val MIN = floatArrayOf(
        -2.88067125e-02f, -8.12731311e-02f, -1.70541159e-03f, -4.88598682e-02f, -3.32196616e-02f, -1.67431499e-04f,
        -6.75425082e-02f, -4.28681746e-02f, -1.98950816e-04f, -7.23103955e-02f, -3.28503326e-02f, -7.31324719e-04f,
        -3.87073644e-02f, -6.01546466e-02f, -5.50269964e-04f, -6.38048723e-02f, -2.23840728e-01f, -7.13261834e-04f,
        -3.02710701e-02f, -3.93195450e-02f, -8.24086510e-06f, -2.95799859e-02f, -5.39318882e-02f, -1.74219604e-04f,
        -2.92359516e-02f, -1.53050944e-02f, -6.30460854e-05f, -5.56493877e-03f, -2.34344602e-02f, -1.26858242e-04f,
        -4.37593013e-02f, -2.77768299e-02f, -2.70503685e-02f, -1.76926646e-02f, -1.91676542e-02f, -1.15090821e-04f,
        -8.34268332e-03f, -3.99775570e-03f, -3.27481248e-05f, -3.40162888e-02f, -2.81868968e-02f, -1.96679524e-04f,
        -2.91855410e-02f, -3.97511162e-02f, -2.81230678e-05f, -1.50395725e-02f, -2.49494594e-02f, -9.42573533e-05f,
        -1.67938769e-02f, -2.00953931e-02f, -4.00750607e-04f, -1.86435618e-02f, -2.48535164e-02f, -2.74416432e-02f,
        -4.61211195e-03f, -1.21660791e-02f, -2.93173041e-04f, -4.10017073e-02f, -7.43824020e-02f, -4.42762971e-02f,
        -1.90370996e-02f, -3.74363363e-02f, -1.34740388e-02f,
    )
    val MAX = floatArrayOf(
        4.46682945e-02f, 7.08772913e-02f, 4.08344204e-04f, 2.14308221e-02f, 6.15894832e-02f, 4.85319615e-05f,
        3.02363783e-02f, 4.45043296e-02f, 1.28298725e-05f, 3.05869691e-02f, 3.79812494e-02f, 6.57040102e-04f,
        4.45670523e-02f, 3.97259220e-02f, 7.10966764e-04f, 9.43699256e-02f, 9.85926315e-02f, 2.02551950e-04f,
        1.61131397e-02f, 2.92906128e-02f, 3.44733417e-06f, 5.23825921e-02f, 1.07065082e-01f, 6.61510974e-04f,
        2.85718683e-03f, 8.32320191e-03f, 2.39314613e-04f, 2.57947259e-02f, 1.60935968e-02f, 2.41853559e-05f,
        4.90833223e-02f, 3.43903080e-02f, 3.22353356e-02f, 1.44766076e-02f, 3.39248963e-02f, 1.42291479e-04f,
        8.75749043e-04f, 6.82212645e-03f, 2.76097053e-05f, 1.86958015e-02f, 3.84016186e-02f, 7.33085908e-05f,
        2.01714113e-02f, 4.90544215e-02f, 2.34028921e-05f, 2.46518422e-02f, 3.29151377e-02f, 3.48571630e-05f,
        2.22457591e-02f, 1.21796541e-02f, 1.56396593e-04f, 1.72109623e-02f, 3.01626958e-02f, 1.36556877e-02f,
        1.83460284e-02f, 1.61141958e-02f, 2.87440169e-04f, 3.57594155e-02f, 1.80554688e-01f, 2.75554154e-02f,
        2.17450950e-02f, 8.66811201e-02f, 3.34241726e-02f,
    )
}
