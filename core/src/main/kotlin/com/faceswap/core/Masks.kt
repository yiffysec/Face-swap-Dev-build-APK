package com.faceswap.core

import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** Float mask utilities (row-major, values 0..1). */
object MaskOps {

    fun resize(m: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int): FloatArray {
        val out = FloatArray(dw * dh)
        val sx = sw.toFloat() / dw; val sy = sh.toFloat() / dh
        for (y in 0 until dh) {
            val fy = ((y + 0.5f) * sy - 0.5f).coerceIn(0f, (sh - 1).toFloat())
            val y0 = floor(fy).toInt(); val y1 = min(y0 + 1, sh - 1); val ay = fy - y0
            for (x in 0 until dw) {
                val fx = ((x + 0.5f) * sx - 0.5f).coerceIn(0f, (sw - 1).toFloat())
                val x0 = floor(fx).toInt(); val x1 = min(x0 + 1, sw - 1); val ax = fx - x0
                val top = m[y0 * sw + x0] * (1 - ax) + m[y0 * sw + x1] * ax
                val bot = m[y1 * sw + x0] * (1 - ax) + m[y1 * sw + x1] * ax
                out[y * dw + x] = top * (1 - ay) + bot * ay
            }
        }
        return out
    }

    /** Separable gaussian blur. */
    fun gaussian(m: FloatArray, w: Int, h: Int, sigma: Float): FloatArray {
        if (sigma < 0.3f) return m.copyOf()
        val r = ceil(sigma * 3).toInt()
        val k = FloatArray(2 * r + 1) { val d = (it - r).toFloat(); exp(-d * d / (2 * sigma * sigma)) }
        val ks = k.sum(); for (i in k.indices) k[i] /= ks
        val tmp = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (i in -r..r) s += m[y * w + (x + i).coerceIn(0, w - 1)] * k[i + r]
            tmp[y * w + x] = s
        }
        val out = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f
            for (i in -r..r) s += tmp[(y + i).coerceIn(0, h - 1) * w + x] * k[i + r]
            out[y * w + x] = s
        }
        return out
    }

    /** FaceFusion-style softening of a hard mask: blur, then (clip(0.5,1)-0.5)*2 so edges stay inside the region. */
    fun softenHard(m: FloatArray, w: Int, h: Int, sigma: Float = 5f): FloatArray {
        val b = gaussian(m, w, h, sigma)
        for (i in b.indices) b[i] = ((b[i].coerceIn(0.5f, 1f)) - 0.5f) * 2f
        return b
    }

    fun minimum(masks: List<FloatArray>): FloatArray {
        val out = masks[0].copyOf()
        for (j in 1 until masks.size) { val m = masks[j]; for (i in out.indices) out[i] = min(out[i], m[i]) }
        return out
    }

    private fun morph(m: FloatArray, w: Int, h: Int, r: Int, dilate: Boolean): FloatArray {
        val tmp = FloatArray(w * h); val out = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var v = if (dilate) 0f else 1f
            for (i in -r..r) { val s = m[y * w + (x + i).coerceIn(0, w - 1)]; v = if (dilate) max(v, s) else min(v, s) }
            tmp[y * w + x] = v
        }
        for (y in 0 until h) for (x in 0 until w) {
            var v = if (dilate) 0f else 1f
            for (i in -r..r) { val s = tmp[(y + i).coerceIn(0, h - 1) * w + x]; v = if (dilate) max(v, s) else min(v, s) }
            out[y * w + x] = v
        }
        return out
    }

    private fun boxBlur(m: FloatArray, w: Int, h: Int, k: Int): FloatArray {
        val r = k / 2
        val tmp = FloatArray(w * h); val out = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) { var s = 0f; for (i in -r..r) s += m[y * w + (x + i).coerceIn(0, w - 1)]; tmp[y * w + x] = s / (2 * r + 1) }
        for (y in 0 until h) for (x in 0 until w) { var s = 0f; for (i in -r..r) s += tmp[(y + i).coerceIn(0, h - 1) * w + x]; out[y * w + x] = s / (2 * r + 1) }
        return out
    }

    /** MobileFaceSwap reference mask post-processing (threshold, close, blur, inner box fade). */
    fun binaryFeather(raw: FloatArray, w: Int, h: Int): FloatArray {
        var m = FloatArray(raw.size) { if (raw[it] > 0.001f) 1f else 0f }
        m = morph(m, w, h, 5, true)
        m = morph(morph(m, w, h, 5, false), w, h, 5, false)
        m = boxBlur(m, w, h, 7)
        val inner = FloatArray(w * h); for (y in 10 until h - 10) for (x in 10 until w - 10) inner[y * w + x] = 1f
        val ib = boxBlur(inner, w, h, 11)
        for (i in m.indices) m[i] *= ib[i]
        return m
    }

    /** Fills the convex hull of [points] (x,y pairs) into a w×h mask. */
    fun convexHullMask(points: FloatArray, w: Int, h: Int): FloatArray {
        val pts = (0 until points.size / 2).map { points[2 * it] to points[2 * it + 1] }.distinct().sortedWith(compareBy({ it.first }, { it.second }))
        if (pts.size < 3) return FloatArray(w * h)
        fun cross(o: Pair<Float, Float>, a: Pair<Float, Float>, b: Pair<Float, Float>) =
            (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first)
        val lower = ArrayList<Pair<Float, Float>>()
        for (p in pts) { while (lower.size >= 2 && cross(lower[lower.size - 2], lower.last(), p) <= 0) lower.removeAt(lower.size - 1); lower += p }
        val upper = ArrayList<Pair<Float, Float>>()
        for (p in pts.asReversed()) { while (upper.size >= 2 && cross(upper[upper.size - 2], upper.last(), p) <= 0) upper.removeAt(upper.size - 1); upper += p }
        val hull = lower.dropLast(1) + upper.dropLast(1)
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val fy = y + 0.5f
            var xmin = Float.MAX_VALUE; var xmax = -Float.MAX_VALUE
            for (i in hull.indices) {
                val a = hull[i]; val b = hull[(i + 1) % hull.size]
                if ((a.second <= fy && b.second > fy) || (b.second <= fy && a.second > fy)) {
                    val x = a.first + (fy - a.second) / (b.second - a.second) * (b.first - a.first)
                    xmin = min(xmin, x); xmax = max(xmax, x)
                }
            }
            if (xmin <= xmax) for (x in max(0, ceil(xmin - 0.5f).toInt())..min(w - 1, floor(xmax - 0.5f).toInt())) out[y * w + x] = 1f
        }
        return out
    }

    /** FaceFusion face areas over 68 landmarks. */
    enum class Area(val indices: IntArray) {
        UPPER_FACE(intArrayOf(0, 1, 2, 31, 32, 33, 34, 35, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17)),
        LOWER_FACE(intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 35, 34, 33, 32, 31)),
        MOUTH(intArrayOf(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67)),
    }

    /** Area mask from 68 landmarks already transformed into crop space. */
    fun areaMask(lm68Crop: FloatArray, areas: Set<Area>, w: Int, h: Int): FloatArray {
        val idx = areas.flatMap { it.indices.toList() }
        val pts = FloatArray(idx.size * 2) { k -> lm68Crop[2 * idx[k / 2] + k % 2] }
        return softenHard(convexHullMask(pts, w, h), w, h, 5f * w / 256f)
    }
}
