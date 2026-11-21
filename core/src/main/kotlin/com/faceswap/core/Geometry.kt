package com.faceswap.core

/** A detected face: box in image pixels, 5 landmarks (L-eye, R-eye, nose, L-mouth, R-mouth) and a detector score. */
data class Face(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val landmarks: FloatArray, // 10 values: x0,y0,x1,y1,...
    val score: Float,
) {
    val width get() = x2 - x1
    val height get() = y2 - y1
    val area get() = width * height
    val centerX get() = (x1 + x2) / 2
    val centerY get() = (y1 + y2) / 2

    /** Embedding gets attached lazily by the engine (normalised 512-d ArcFace vector). */
    var embedding: FloatArray? = null

    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}

/** Normalised (0..1) 5-point alignment templates. */
object Templates {
    /** InsightFace ArcFace template for 112x112 crops. */
    val ARCFACE_112 = floatArrayOf(
        38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
    ).map { it / 112f }.toFloatArray()

    /** Template used by inswapper (ArcFace shifted +8px for 128 crops). */
    val ARCFACE_128 = floatArrayOf(
        46.2946f, 51.6963f, 81.5318f, 51.5014f, 64.0252f, 71.7366f, 49.5493f, 92.3655f, 78.7299f, 92.2041f,
    ).map { it / 128f }.toFloatArray()

    /** FFHQ template used by GFPGAN / GPEN restoration models. */
    val FFHQ_512 = floatArrayOf(
        0.37691676f, 0.46864664f, 0.62285697f, 0.46912813f, 0.50123859f, 0.61331904f,
        0.39308822f, 0.72541100f, 0.61150205f, 0.72490465f,
    )
}

object Geometry {
    /**
     * Least-squares similarity transform (rotation + uniform scale + translation, Umeyama without reflection)
     * mapping [src] landmarks onto [template] * [size].
     */
    fun estimateSimilarity(src: FloatArray, template: FloatArray, size: Float): Affine {
        val n = src.size / 2
        var mx = 0f; var my = 0f; var ux = 0f; var uy = 0f
        for (i in 0 until n) {
            mx += src[2 * i]; my += src[2 * i + 1]
            ux += template[2 * i] * size; uy += template[2 * i + 1] * size
        }
        mx /= n; my /= n; ux /= n; uy /= n
        var a = 0f; var b = 0f; var norm = 0f
        for (i in 0 until n) {
            val xs = src[2 * i] - mx; val ys = src[2 * i + 1] - my
            val xd = template[2 * i] * size - ux; val yd = template[2 * i + 1] * size - uy
            a += xs * xd + ys * yd
            b += xs * yd - ys * xd
            norm += xs * xs + ys * ys
        }
        val c = a / norm
        val s = b / norm
        return Affine(c, -s, ux - (c * mx - s * my), s, c, uy - (s * mx + c * my))
    }

    fun iou(a: Face, b: Face): Float {
        val ix = maxOf(0f, minOf(a.x2, b.x2) - maxOf(a.x1, b.x1))
        val iy = maxOf(0f, minOf(a.y2, b.y2) - maxOf(a.y1, b.y1))
        val inter = ix * iy
        return inter / (a.area + b.area - inter + 1e-6f)
    }

    fun nms(faces: List<Face>, threshold: Float): List<Face> {
        val sorted = faces.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Face>()
        while (sorted.isNotEmpty()) {
            val f = sorted.removeAt(0)
            keep += f
            sorted.removeAll { iou(f, it) > threshold }
        }
        return keep
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        var d = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return d / (kotlin.math.sqrt(na * nb) + 1e-9f)
    }

    fun normalize(v: FloatArray): FloatArray {
        var n = 0f
        for (x in v) n += x * x
        n = kotlin.math.sqrt(n) + 1e-9f
        return FloatArray(v.size) { v[it] / n }
    }
}
