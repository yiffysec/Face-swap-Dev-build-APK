package com.faceswap.core

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** ARGB_8888 packed image. Platform independent so the whole pipeline can run (and be tested) on the JVM. */
class RgbImage(val width: Int, val height: Int, val pixels: IntArray = IntArray(width * height)) {
    init {
        require(pixels.size == width * height) { "pixel buffer size mismatch" }
    }

    fun copy(): RgbImage = RgbImage(width, height, pixels.copyOf())

    inline fun r(i: Int) = (pixels[i] shr 16) and 0xFF
    inline fun g(i: Int) = (pixels[i] shr 8) and 0xFF
    inline fun b(i: Int) = pixels[i] and 0xFF
}

/** 2x3 affine matrix [a b c; d e f] mapping (x,y) -> (a*x + b*y + c, d*x + e*y + f). */
data class Affine(val a: Float, val b: Float, val c: Float, val d: Float, val e: Float, val f: Float) {
    fun mapX(x: Float, y: Float) = a * x + b * y + c
    fun mapY(x: Float, y: Float) = d * x + e * y + f

    fun invert(): Affine {
        val det = a * e - b * d
        require(kotlin.math.abs(det) > 1e-12f) { "singular affine" }
        val ia = e / det
        val ib = -b / det
        val id = -d / det
        val ie = a / det
        return Affine(ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f))
    }

    /** Returns this ∘ other (apply other first). */
    fun compose(other: Affine) = Affine(
        a * other.a + b * other.d, a * other.b + b * other.e, a * other.c + b * other.f + c,
        d * other.a + e * other.d, d * other.b + e * other.e, d * other.c + e * other.f + f,
    )

    companion object {
        fun scale(s: Float) = Affine(s, 0f, 0f, 0f, s, 0f)
    }
}

object ImageOps {

    /**
     * Warps [src] into a [w]x[h] image. [m] maps SOURCE coords to DESTINATION coords (like cv2.warpAffine's M).
     * Bilinear sampling, border = replicate.
     */
    fun warpAffine(src: RgbImage, m: Affine, w: Int, h: Int): RgbImage {
        val inv = m.invert()
        val out = RgbImage(w, h)
        val sp = src.pixels
        val sw = src.width
        val sh = src.height
        val op = out.pixels
        Par.rows(h, w) { from, to ->
            for (y in from until to) {
                var sx = inv.b * y + inv.c
                var sy = inv.e * y + inv.f
                val row = y * w
                for (x in 0 until w) {
                    op[row + x] = sampleBilinear(sp, sw, sh, sx, sy)
                    sx += inv.a
                    sy += inv.d
                }
            }
        }
        return out
    }

    fun sampleBilinear(p: IntArray, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x = fx.coerceIn(0f, (w - 1).toFloat())
        val y = fy.coerceIn(0f, (h - 1).toFloat())
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val ax = x - x0
        val ay = y - y0
        val p00 = p[y0 * w + x0]
        val p01 = p[y0 * w + x1]
        val p10 = p[y1 * w + x0]
        val p11 = p[y1 * w + x1]
        if (p00 == p01 && p00 == p10 && p00 == p11) return p00 or (0xFF shl 24) // flat area / exact pixel hit
        val w00 = (1 - ax) * (1 - ay); val w01 = ax * (1 - ay); val w10 = (1 - ax) * ay; val w11 = ax * ay
        val r = (((p00 shr 16) and 0xFF) * w00 + ((p01 shr 16) and 0xFF) * w01 + ((p10 shr 16) and 0xFF) * w10 + ((p11 shr 16) and 0xFF) * w11 + 0.5f).toInt()
        val g = (((p00 shr 8) and 0xFF) * w00 + ((p01 shr 8) and 0xFF) * w01 + ((p10 shr 8) and 0xFF) * w10 + ((p11 shr 8) and 0xFF) * w11 + 0.5f).toInt()
        val b = ((p00 and 0xFF) * w00 + (p01 and 0xFF) * w01 + (p10 and 0xFF) * w10 + (p11 and 0xFF) * w11 + 0.5f).toInt()
        return (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)
    }

    fun resize(src: RgbImage, w: Int, h: Int): RgbImage =
        warpAffine(src, Affine(w.toFloat() / src.width, 0f, 0f, 0f, h.toFloat() / src.height, 0f), w, h)

    /** CHW float tensor, RGB order, value = (pixel * scale - mean) / std. */
    fun toChw(img: RgbImage, mean: Float, std: Float, scale: Float = 1f, bgr: Boolean = false): FloatArray {
        val n = img.width * img.height
        val out = FloatArray(3 * n)
        val p = img.pixels
        // v' = (v*scale - mean)/std = v*k + o
        val k = scale / std; val o = -mean / std
        val c0 = if (bgr) 0 else 16; val c2 = if (bgr) 16 else 0
        Par.rows(img.height, img.width) { from, to ->
            for (i in from * img.width until to * img.width) {
                val v = p[i]
                out[i] = ((v shr c0) and 0xFF) * k + o
                out[n + i] = ((v shr 8) and 0xFF) * k + o
                out[2 * n + i] = ((v shr c2) and 0xFF) * k + o
            }
        }
        return out
    }

    /** Inverse of [toChw] for RGB: pixel = (v * std + mean) / scale. */
    fun fromChw(data: FloatArray, w: Int, h: Int, mean: Float, std: Float, scale: Float = 1f): RgbImage {
        val n = w * h
        val out = RgbImage(w, h)
        val k = std / scale; val o = mean / scale + 0.5f
        val op = out.pixels
        Par.rows(h, w) { from, to ->
            for (i in from * w until to * w) {
                val r = (data[i] * k + o).toInt().coerceIn(0, 255)
                val g = (data[n + i] * k + o).toInt().coerceIn(0, 255)
                val b = (data[2 * n + i] * k + o).toInt().coerceIn(0, 255)
                op[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return out
    }

    /**
     * Soft box mask for a square/rect crop: 1 in the centre, feathered to 0 at the edges.
     * [blur] 0..1 = fraction of half-size used for feathering, [padding] = fraction of size cut from each side (top,right,bottom,left).
     */
    fun boxMask(w: Int, h: Int, blur: Float, padding: FloatArray = floatArrayOf(0f, 0f, 0f, 0f)): FloatArray {
        val mask = FloatArray(w * h)
        val feather = max(1f, min(w, h) * 0.5f * blur)
        val edge = max(1f, feather * 0.5f) // matches FaceFusion's hard border before blurring
        val padT = padding[0] * h; val padR = padding[1] * w; val padB = padding[2] * h; val padL = padding[3] * w
        for (y in 0 until h) {
            val dy = min(y - padT, (h - 1 - y) - padB)
            for (x in 0 until w) {
                val dx = min(x - padL, (w - 1 - x) - padR)
                val d = min(dx, dy) - edge * 0.5f
                val t = (d / feather).coerceIn(0f, 1f)
                mask[y * w + x] = t * t * (3 - 2 * t) // smoothstep
            }
        }
        return mask
    }

    /** Elliptical feathered mask – tighter fit around the face, avoids "boxy" seams on hair/background. */
    fun ellipseMask(w: Int, h: Int, blur: Float, scaleX: Float = 0.46f, scaleY: Float = 0.56f, centerY: Float = 0.54f): FloatArray {
        val mask = FloatArray(w * h)
        val cx = w / 2f
        val cy = h * centerY
        val rx = w * scaleX
        val ry = h * scaleY
        val feather = max(0.02f, blur * 0.5f)
        for (y in 0 until h) for (x in 0 until w) {
            val nx = (x - cx) / rx
            val ny = (y - cy) / ry
            val r = kotlin.math.sqrt(nx * nx + ny * ny)
            val t = ((1f - r) / feather).coerceIn(0f, 1f)
            mask[y * w + x] = t * t * (3 - 2 * t)
        }
        return mask
    }

    fun multiplyMasks(a: FloatArray, b: FloatArray): FloatArray = FloatArray(a.size) { a[it] * b[it] }

    /**
     * Pastes [crop] (with alpha [mask], same size as crop) back into [target] in place.
     * [m] is the affine that mapped target -> crop space.
     */
    fun pasteBack(target: RgbImage, crop: RgbImage, mask: FloatArray, m: Affine, opacity: Float = 1f) {
        val inv = m.invert()
        val cw = crop.width
        val ch = crop.height
        // bounding box of crop corners in target space
        val xs = floatArrayOf(inv.mapX(0f, 0f), inv.mapX(cw.toFloat(), 0f), inv.mapX(0f, ch.toFloat()), inv.mapX(cw.toFloat(), ch.toFloat()))
        val ys = floatArrayOf(inv.mapY(0f, 0f), inv.mapY(cw.toFloat(), 0f), inv.mapY(0f, ch.toFloat()), inv.mapY(cw.toFloat(), ch.toFloat()))
        val x0 = max(0, floor(xs.min()).toInt())
        val x1 = min(target.width - 1, kotlin.math.ceil(xs.max()).toInt())
        val y0 = max(0, floor(ys.min()).toInt())
        val y1 = min(target.height - 1, kotlin.math.ceil(ys.max()).toInt())
        if (x0 > x1 || y0 > y1) return
        val tp = target.pixels
        val tw = target.width
        Par.rows(y1 - y0 + 1, x1 - x0 + 1) { from, to ->
            for (y in y0 + from until y0 + to) {
                for (x in x0..x1) {
                    val cx = m.mapX(x.toFloat(), y.toFloat())
                    val cy = m.mapY(x.toFloat(), y.toFloat())
                    if (cx < 0f || cy < 0f || cx > cw - 1 || cy > ch - 1) continue
                    val alpha = sampleMask(mask, cw, ch, cx, cy) * opacity
                    if (alpha <= 0.002f) continue
                    val src = sampleBilinear(crop.pixels, cw, ch, cx, cy)
                    val idx = y * tw + x
                    tp[idx] = blend(tp[idx], src, alpha)
                }
            }
        }
    }

    private fun sampleMask(mask: FloatArray, w: Int, h: Int, fx: Float, fy: Float): Float {
        val x0 = floor(fx).toInt().coerceIn(0, w - 1)
        val y0 = floor(fy).toInt().coerceIn(0, h - 1)
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val ax = fx - x0
        val ay = fy - y0
        val top = mask[y0 * w + x0] * (1 - ax) + mask[y0 * w + x1] * ax
        val bot = mask[y1 * w + x0] * (1 - ax) + mask[y1 * w + x1] * ax
        return top * (1 - ay) + bot * ay
    }

    fun blend(dst: Int, src: Int, a: Float): Int {
        if (a >= 0.999f) return src
        val ia = 1f - a
        val r = (((dst shr 16) and 0xFF) * ia + ((src shr 16) and 0xFF) * a + 0.5f).toInt()
        val g = (((dst shr 8) and 0xFF) * ia + ((src shr 8) and 0xFF) * a + 0.5f).toInt()
        val b = ((dst and 0xFF) * ia + (src and 0xFF) * a + 0.5f).toInt()
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    /** Blends two same-size images: out = a*(1-t) + b*t. */
    fun mix(a: RgbImage, b: RgbImage, t: Float): RgbImage {
        val out = RgbImage(a.width, a.height)
        Par.rows(a.height, a.width) { from, to -> for (i in from * a.width until to * a.width) out.pixels[i] = blend(a.pixels[i], b.pixels[i], t) }
        return out
    }

    /**
     * Sharpness = variance of the Laplacian of luma (on a ≤[maxSide] downscale). Used to skip the face enhancer when
     * the swapped face is already as sharp as the original (enhancing then adds nothing but time).
     */
    fun sharpness(img: RgbImage, maxSide: Int = 160): Double {
        val s = if (maxOf(img.width, img.height) > maxSide) resize(img, maxSide * img.width / maxOf(img.width, img.height), maxSide * img.height / maxOf(img.width, img.height)) else img
        val w = s.width; val h = s.height
        if (w < 3 || h < 3) return 0.0
        val l = FloatArray(w * h) { val p = s.pixels[it]; 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF) }
        var sum = 0.0; var sq = 0.0; var n = 0
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            val v = (l[i - 1] + l[i + 1] + l[i - w] + l[i + w] - 4 * l[i]).toDouble()
            sum += v; sq += v * v; n++
        }
        val mean = sum / n
        return sq / n - mean * mean
    }

    /**
     * Mean/std colour transfer in RGB from [reference] to [img] (inside mask weight). Reduces skin-tone mismatch
     * between swapped face and target lighting.
     */
    fun colorTransfer(img: RgbImage, reference: RgbImage, mask: FloatArray, strength: Float): RgbImage {
        if (strength <= 0f) return img
        val n = img.pixels.size
        val ms = DoubleArray(3); val mr = DoubleArray(3); val ss = DoubleArray(3); val sr = DoubleArray(3)
        var wsum = 0.0
        for (i in 0 until n) {
            val w = mask[i].toDouble(); if (w < 0.5) continue
            wsum += w
            for (c in 0..2) {
                val sh = 16 - 8 * c
                val s = ((img.pixels[i] shr sh) and 0xFF).toDouble()
                val r = ((reference.pixels[i] shr sh) and 0xFF).toDouble()
                ms[c] += s * w; mr[c] += r * w; ss[c] += s * s * w; sr[c] += r * r * w
            }
        }
        if (wsum < 16) return img
        val gain = FloatArray(3); val off = FloatArray(3)
        for (c in 0..2) {
            val mS = ms[c] / wsum; val mR = mr[c] / wsum
            val sdS = kotlin.math.sqrt(max(1.0, ss[c] / wsum - mS * mS))
            val sdR = kotlin.math.sqrt(max(1.0, sr[c] / wsum - mR * mR))
            val g = (sdR / sdS).coerceIn(0.6, 1.6)
            gain[c] = (1 + (g - 1) * strength).toFloat()
            off[c] = ((mR - mS * g) * strength).toFloat() + ((1 - strength) * 0f)
            // with strength: v' = v + strength*((mR + (v-mS)*g) - v)
            gain[c] = (1 - strength + strength * g).toFloat()
            off[c] = (strength * (mR - mS * g)).toFloat()
        }
        val out = RgbImage(img.width, img.height)
        for (i in 0 until n) {
            val p = img.pixels[i]
            val r = (((p shr 16) and 0xFF) * gain[0] + off[0]).roundToInt().coerceIn(0, 255)
            val g = (((p shr 8) and 0xFF) * gain[1] + off[1]).roundToInt().coerceIn(0, 255)
            val b = ((p and 0xFF) * gain[2] + off[2]).roundToInt().coerceIn(0, 255)
            out.pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return out
    }

    /** Light unsharp mask on a crop – recovers some crispness lost by the 128px swapper without a full enhancer. */
    fun sharpen(img: RgbImage, amount: Float): RgbImage {
        if (amount <= 0f) return img
        val w = img.width; val h = img.height
        val out = RgbImage(w, h)
        val p = img.pixels
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (x == 0 || y == 0 || x == w - 1 || y == h - 1) { out.pixels[i] = p[i]; continue }
            var res = 0xFF shl 24
            for (sh in intArrayOf(16, 8, 0)) {
                val c = (p[i] shr sh) and 0xFF
                val blur = (((p[i - 1] shr sh) and 0xFF) + ((p[i + 1] shr sh) and 0xFF) +
                        ((p[i - w] shr sh) and 0xFF) + ((p[i + w] shr sh) and 0xFF)) / 4f
                val v = (c + (c - blur) * amount).roundToInt().coerceIn(0, 255)
                res = res or (v shl sh)
            }
            out.pixels[i] = res
        }
        return out
    }
}
