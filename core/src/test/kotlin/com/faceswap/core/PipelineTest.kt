package com.faceswap.core

import ai.onnxruntime.OrtSession
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Test-only verifier. Never shipped: lives in test sources. */
class FakeIdentityVerifier(private val embedding: FloatArray?) : IdentityVerifier {
    override fun current() = embedding?.let { VerifiedIdentity(it, "test", 0L) }
    override val providerName = "fake"
}

class PipelineTest {
    private val dir = File(System.getProperty("models.dir") ?: "")
    private val out = File("build/test-output").apply { mkdirs() }

    private fun load(f: File): RgbImage {
        val bi = ImageIO.read(f)
        val px = IntArray(bi.width * bi.height)
        bi.getRGB(0, 0, bi.width, bi.height, px, 0, bi.width)
        return RgbImage(bi.width, bi.height, px)
    }

    private fun save(img: RgbImage, name: String) {
        val bi = BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB)
        bi.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
        ImageIO.write(bi, "png", File(out, name))
    }

    private val factory = SessionFactory { env, path ->
        env.createSession(path, OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
    }

    private fun engine(verifier: IdentityVerifier = NoIdentityVerifier) = FaceSwapEngine(
        ModelFiles(
            File(dir, "scrfd_2.5g.onnx"), File(dir, "arcface_w600k_r50.onnx"), File(dir, "inswapper_128_fp16.onnx"),
            File(dir, "open_nsfw.onnx"), File(dir, "gfpgan_1.4.onnx"), File(out, "emap.bin"),
        ), factory, verifier,
    )

    @Test
    fun emapMatchesReference() {
        assumeTrue(File(dir, "inswapper_128_fp16.onnx").exists())
        val m = OnnxInitializerReader.readFloatInitializer(File(dir, "inswapper_128_fp16.onnx"))
        assertEquals(512 * 512, m.size)
        assertEquals(0.12484695f, m[0], 1e-6f)
        assertEquals(-0.08580378f, m[512 * 512 - 1], 1e-6f)
    }

    @Test
    fun fullPipeline() {
        assumeTrue(File(dir, "scrfd_2.5g.onnx").exists() && File(dir, "ex/source.jpg").exists() && File(dir, "ex/target.jpg").exists())
        val src = load(File(dir, "ex/source.jpg"))
        val tgt = load(File(dir, "ex/target.jpg"))
        engine().use { e ->
            val sf = e.detect(src)
            val tf = e.detect(tgt)
            println("source faces=${sf.size} ${sf.map { it.score }}  target faces=${tf.size}")
            assertTrue(sf.isNotEmpty() && tf.isNotEmpty())
            // detection visualisation
            val vis = tgt.copy()
            for (f in tf) for (k in 0 until 5) {
                val x = f.landmarks[2 * k].toInt(); val y = f.landmarks[2 * k + 1].toInt()
                for (dy in -1..1) for (dx in -1..1) if (x + dx in 0 until vis.width && y + dy in 0 until vis.height) vis.pixels[(y + dy) * vis.width + x + dx] = 0xFF00FF00.toInt()
            }
            save(vis, "detect.png")

            val source = e.buildSource("s", "Source", listOf(src to sf.maxBy { it.area }))
            val selfSim = Geometry.cosine(source.averaged, e.embed(src, sf.maxBy { it.area }))
            val crossSim = Geometry.cosine(source.averaged, e.embed(tgt, tf.maxBy { it.area }))
            println("self sim=$selfSim  source-vs-target sim=$crossSim")

            for (boost in listOf(128, 256, 512)) {
                val r = e.process(tgt, listOf(FaceMapping(source)), SwapConfig(pixelBoost = boost))
                println("boost=$boost swapped=${r.facesSwapped} in ${r.millis}ms decision=${r.decision}")
                assertTrue(r.facesSwapped > 0)
                save(r.image, "swap_$boost.png")
                // the swapped face should now look like the source identity
                val after = Geometry.cosine(source.averaged, e.recognizer.embed(r.image, e.detect(r.image).maxBy { it.area }))
                println("  identity similarity after swap=$after (before=$crossSim)")
                assertTrue(after > crossSim + 0.2f)
            }
            val r = e.process(tgt, listOf(FaceMapping(source)), SwapConfig(pixelBoost = 256, enhance = true))
            println("enhanced in ${r.millis}ms")
            save(r.image, "swap_256_enhanced.png")
            println("explicit prob target=${e.policy.isExplicit(tgt)}")
        }
    }

    @Test
    fun policyRules() {
        assumeTrue(File(dir, "scrfd_2.5g.onnx").exists() && File(dir, "ex/source.jpg").exists() && File(dir, "ex/target.jpg").exists())
        val src = load(File(dir, "ex/source.jpg"))
        val tgt = load(File(dir, "ex/target.jpg"))
        engine().use { e ->
            val s = e.buildSource("s", "S", listOf(src to e.detect(src).maxBy { it.area }))
            val other = e.buildSource("o", "O", listOf(tgt to e.detect(tgt).maxBy { it.area }))
            // non-explicit: always allowed
            assertTrue(e.policy.evaluate(false, listOf(other)).allowed)
            // explicit + nobody verified: blocked
            assertTrue(e.policy.evaluate(true, listOf(s)) is PolicyDecision.ExplicitRequiresVerification)
            // explicit + verified as S: own face allowed, someone else blocked, mixed blocked
            e.policy.verifier = FakeIdentityVerifier(s.averaged)
            assertTrue(e.policy.evaluate(true, listOf(s)).allowed)
            assertTrue(e.policy.evaluate(true, listOf(other)) is PolicyDecision.ExplicitIdentityMismatch)
            assertTrue(!e.policy.evaluate(true, listOf(s, other)).allowed)
            // internal build live-capture mode: gallery source blocked, live source allowed
            e.policy.verifier = NoIdentityVerifier
            e.policy.internalLiveCaptureMode = true
            val live = SourceFace("l", "L", s.embeddings, SourceProvenance.LIVE_CAMERA)
            assertTrue(e.policy.evaluate(true, listOf(s)) is PolicyDecision.ExplicitRequiresLiveCapture)
            assertTrue(e.policy.evaluate(true, listOf(live)).allowed)
            assertTrue(!e.policy.evaluate(true, listOf(live, other)).allowed)
            e.policy.internalLiveCaptureMode = false
            assertTrue(!e.policy.evaluate(true, listOf(live)).allowed)
            // explicit override on a real frame is blocked without verification
            e.policy.verifier = NoIdentityVerifier
            val r = e.process(tgt, listOf(FaceMapping(s)), SwapConfig(), explicitOverride = true)
            assertEquals(0, r.facesSwapped)
        }
    }
}
