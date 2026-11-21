package com.faceswap.core

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * A detected face: box in image pixels, 5 landmarks (L-eye, R-eye, nose, L-mouth, R-mouth) and a detector score.
 * When a 68-point landmarker ran, [landmarks68] is set and [landmarks] holds the more accurate 5 points derived from it.
 */
class Face(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    var landmarks: FloatArray, // 10 values: x0,y0,x1,y1,...
    val score: Float,
) {
    val width get() = x2 - x1
    val height get() = y2 - y1
    val area get() = width * height
    val centerX get() = (x1 + x2) / 2
    val centerY get() = (y1 + y2) / 2

    /** Detector's raw 5 points (kept for debugging). */
    val detectorLandmarks: FloatArray = landmarks.copyOf()

    /** 68-point landmarks (136 values) if a landmarker ran. */
    var landmarks68: FloatArray? = null

    /** Normalised 512-d ArcFace embedding, attached lazily. */
    var embedding: FloatArray? = null

    fun copyWith(landmarks: FloatArray) = Face(x1, y1, x2, y2, landmarks, score).also {
        it.landmarks68 = landmarks68; it.embedding = embedding
    }
}

/** Normalised (0..1) 5-point alignment templates. */
object Templates {
    /** InsightFace ArcFace template for 112×112 crops. */
    val ARCFACE_112 = floatArrayOf(
        38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
    ).map { it / 112f }.toFloatArray()

    /** ArcFace template shifted +8px for 128 crops (inswapper, HyperSwap, GPEN-256, LivePortrait restorer). */
    val ARCFACE_128 = floatArrayOf(
        46.2946f, 51.6963f, 81.5318f, 51.5014f, 64.0252f, 71.7366f, 49.5493f, 92.3655f, 78.7299f, 92.2041f,
    ).map { it / 128f }.toFloatArray()

    /** FFHQ template (GFPGAN, GPEN-512, RestoreFormer++, CodeFormer, FRAN, LivePortrait editor). */
    val FFHQ_512 = floatArrayOf(
        0.37691676f, 0.46864664f, 0.62285697f, 0.46912813f, 0.50123859f, 0.61331904f,
        0.39308822f, 0.72541100f, 0.61150205f, 0.72490465f,
    )

    /** MobileFaceSwap: ArcFace points +8px, scaled size/112 (not /128). */
    val MOBILEFACESWAP = floatArrayOf(
        38.29459953f, 51.69630051f, 73.53179932f, 51.50139999f, 56.02519989f, 71.73660278f,
        41.54930115f, 92.3655014f, 70.72990036f, 92.20410156f,
    ).map { it / 112f }.toFloatArray()
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

    /** 68 → 5 points (eye centres, nose tip, mouth corners), as FaceFusion's '5/68'. */
    fun fiveFrom68(lm: FloatArray): FloatArray {
        fun mean(from: Int, to: Int, axis: Int): Float { var s = 0f; for (i in from until to) s += lm[2 * i + axis]; return s / (to - from) }
        return floatArrayOf(
            mean(36, 42, 0), mean(36, 42, 1), mean(42, 48, 0), mean(42, 48, 1),
            lm[60], lm[61], lm[96], lm[97], lm[108], lm[109],
        )
    }

    fun transformPoints(points: FloatArray, m: Affine): FloatArray = FloatArray(points.size).also { out ->
        for (i in 0 until points.size / 2) {
            out[2 * i] = m.mapX(points[2 * i], points[2 * i + 1]); out[2 * i + 1] = m.mapY(points[2 * i], points[2 * i + 1])
        }
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
        return d / (sqrt(na * nb) + 1e-9f)
    }

    fun normalize(v: FloatArray): FloatArray {
        var n = 0f
        for (x in v) n += x * x
        n = sqrt(n) + 1e-9f
        return FloatArray(v.size) { v[it] / n }
    }

    /** In-plane roll of a face from its eye line, degrees. */
    fun roll(face: Face): Float {
        val lm = face.landmarks
        return Math.toDegrees(atan2((lm[3] - lm[1]).toDouble(), (lm[2] - lm[0]).toDouble())).toFloat()
    }

    @Suppress("unused") private fun unused() = abs(0)
}
