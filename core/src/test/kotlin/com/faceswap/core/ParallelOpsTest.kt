package com.faceswap.core

import org.junit.Test
import kotlin.random.Random
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/** The row-parallel pixel loops must produce bit-identical output to the serial path. */
class ParallelOpsTest {
    private fun noise(w: Int, h: Int, seed: Int) = RgbImage(w, h, IntArray(w * h) { Random(seed + it).nextInt() or (0xFF shl 24) })

    private fun <T> both(block: () -> T): Pair<T, T> {
        Par.enabled = false
        val a = block()
        Par.enabled = true
        return a to block()
    }

    @Test
    fun identicalToSerial() {
        val img = noise(700, 500, 1)
        val m = Affine(0.83f, 0.12f, -20f, -0.1f, 0.85f, 30f)
        val (w1, w2) = both { ImageOps.warpAffine(img, m, 512, 512).pixels }
        assertContentEquals(w1, w2)
        val (c1, c2) = both { ImageOps.toChw(img, 127.5f, 128f) }
        assertContentEquals(c1, c2)
        val (f1, f2) = both { ImageOps.fromChw(c1, 700, 500, 127.5f, 128f).pixels }
        assertContentEquals(f1, f2)
        assertContentEquals(img.pixels, f1, "toChw/fromChw round trip must be lossless")
        val mask = FloatArray(512 * 512) { Random(it).nextFloat() }
        val (g1, g2) = both { MaskOps.gaussian(mask, 512, 512, 6f) }
        assertContentEquals(g1, g2)
        val crop = noise(512, 512, 7)
        val (p1, p2) = both { img.copy().also { ImageOps.pasteBack(it, crop, g1, m) }.pixels }
        assertContentEquals(p1, p2)
        val (r1, r2) = both { MaskOps.resize(mask, 512, 512, 300, 260) }
        assertContentEquals(r1, r2)
    }

    @Test
    fun sharpnessOrdersBlur() {
        val img = noise(160, 160, 3)
        val blurred = ImageOps.resize(ImageOps.resize(img, 40, 40), 160, 160)
        assertTrue(ImageOps.sharpness(img) > ImageOps.sharpness(blurred) * 4)
    }
}
