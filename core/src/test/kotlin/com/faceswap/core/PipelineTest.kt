package com.faceswap.core

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
        val alpha = img.pixels.any { (it ushr 24) != 0xFF }
        val bi = BufferedImage(img.width, img.height, if (alpha) BufferedImage.TYPE_INT_ARGB else BufferedImage.TYPE_INT_RGB)
        bi.setRGB(0, 0, img.width, img.height, img.pixels, 0, img.width)
        ImageIO.write(bi, "png", File(out, name))
    }

    private fun has(id: String) = File(dir, "$id.onnx").exists()
    private fun ready() = has(ModelIds.ARCFACE) && File(dir, "ex/source.jpg").exists() && File(dir, "ex/target.jpg").exists()

    private fun engine(verifier: IdentityVerifier = NoIdentityVerifier): FaceSwapEngine {
        val loader = OrtModel.cpuLoader(4)
        return FaceSwapEngine({ id -> File(dir, "$id.onnx") }, { _, f -> loader.load(f) }, verifier, File(out, "emap.bin"))
    }

    private val src by lazy { load(File(dir, "ex/source.jpg")) }
    private val tgt by lazy { load(File(dir, "ex/target.jpg")) }

    private fun identityAfter(e: FaceSwapEngine, img: RgbImage, source: SourceFace) =
        Geometry.cosine(source.averaged, e.recognizer.embed(img, e.detect(img, SwapConfig(landmarks68 = false)).maxBy { it.area }))

    @Test
    fun emapMatchesReference() {
        assumeTrue(has(ModelIds.INSWAPPER))
        val m = OnnxInitializerReader.readFloatInitializer(File(dir, "${ModelIds.INSWAPPER}.onnx"))
        assertEquals(512 * 512, m.size)
        assertEquals(0.12484695f, m[0], 1e-6f)
        assertEquals(-0.08580378f, m[512 * 512 - 1], 1e-6f)
    }

    @Test
    fun detectorsAgree() {
        assumeTrue(ready() && has(ModelIds.YUNET) && has(ModelIds.SCRFD))
        engine().use { e ->
            val y = e.detect(tgt, SwapConfig(detectorId = ModelIds.YUNET, landmarks68 = false))
            val s = e.detect(tgt, SwapConfig(detectorId = ModelIds.SCRFD, landmarks68 = false))
            println("yunet=${y.size} scrfd=${s.size}")
            assertTrue(y.size == 1 && s.size == 1)
            var maxDiff = 0f
            for (i in 0 until 10) maxDiff = maxOf(maxDiff, kotlin.math.abs(y[0].landmarks[i] - s[0].landmarks[i]))
            println("max 5-pt landmark diff yunet vs scrfd = $maxDiff px")
            assertTrue(maxDiff < 8f)
            if (has(ModelIds.HRFFA)) {
                val r = e.detect(tgt, SwapConfig())
                assertTrue(r[0].landmarks68 != null)
                val d = (0 until 10).maxOf { kotlin.math.abs(r[0].landmarks[it] - s[0].landmarks[it]) }
                println("hrffa-derived 5-pt vs scrfd max diff = $d px")
                assertTrue(d < 10f)
                save(tgt.copy().also { Debugger.draw(it, r, emptySet()) }, "debug_overlay.png")
            }
        }
    }

    @Test
    fun swappersAndMasks() {
        assumeTrue(ready() && has(ModelIds.HYPERSWAP_1A))
        engine().use { e ->
            val source = e.buildSource("s", "S", listOf(src to e.detect(src).maxBy { it.area }))
            val before = identityAfter(e, tgt, source)
            val cases = listOf(
                "hyperswap_box_model" to SwapConfig(),
                "hyperswap_region_area" to SwapConfig(maskTypes = setOf(MaskType.BOX, MaskType.MODEL, MaskType.REGION, MaskType.AREA)),
                "hyperswap_boost512" to SwapConfig(pixelBoost = 512),
                "inswapper_boost256" to SwapConfig(swapperId = ModelIds.INSWAPPER, pixelBoost = 256, maskTypes = setOf(MaskType.BOX)),
                "hyperswap_gpen256" to SwapConfig(enhancerId = ModelIds.GPEN_256),
                "hyperswap_gfpgan" to SwapConfig(enhancerId = ModelIds.GFPGAN),
                "hyperswap_restoreformer" to SwapConfig(enhancerId = ModelIds.RESTOREFORMER),
                "hyperswap_codeformer" to SwapConfig(enhancerId = ModelIds.CODEFORMER),
                "hyperswap_expression" to SwapConfig(expressionRestore = 0.8f),
            )
            for ((name, cfg) in cases) {
                if (!e.has(cfg.swapperId) || (cfg.enhancerId != null && !e.has(cfg.enhancerId))) { println("skip $name"); continue }
                val r = e.process(tgt, listOf(FaceMapping(source)), cfg)
                val after = identityAfter(e, r.image, source)
                println("%-26s swapped=%d %5d ms identity %.3f→%.3f  %s".format(name, r.facesSwapped, r.millis, before, after, r.timings))
                save(r.image, "$name.png")
                assertTrue(r.facesSwapped == 1)
                assertTrue(after > before + 0.3f, "$name identity too low: $after")
                if (cfg.enhancerId != null || cfg.expressionRestore > 0f) e.unloadAll()
            }
        }
    }

    @Test
    fun draftAndAdaptiveEnhance() {
        assumeTrue(ready() && has(ModelIds.HYPERSWAP_1A) && has(ModelIds.GPEN_256))
        engine().use { e ->
            val source = e.buildSource("s", "S", listOf(src to e.detect(src).maxBy { it.area }))
            val cfg = SwapConfig(pixelBoost = 512, enhancerId = ModelIds.GPEN_256)
            val plan = e.analyze(tgt, listOf(FaceMapping(source)), cfg, explicitOverride = false)
            e.render(tgt, plan, cfg) // warm-up: model loading isn't part of the comparison
            var t = System.currentTimeMillis()
            val draft = e.render(tgt, plan, cfg, draft = true)
            val draftMs = System.currentTimeMillis() - t
            t = System.currentTimeMillis()
            val full = e.render(tgt, plan, cfg)
            val fullMs = System.currentTimeMillis() - t
            println("draft $draftMs ms vs final $fullMs ms")
            assertTrue(identityAfter(e, draft, source) > 0.4f)
            assertTrue(draftMs < fullMs)
            // the 240p example face is small: the swap is already as sharp as the original, so the enhancer is skipped
            val before = e.enhanceSkipped
            e.render(tgt, plan, cfg.copy(adaptiveEnhance = true))
            println("adaptive enhance skipped ${e.enhanceSkipped - before} face(s)")
            save(full, "final_boost512_gpen.png"); save(draft, "draft.png")
        }
    }

    @Test
    fun processors() {
        assumeTrue(ready())
        engine().use { e ->
            val face = e.detect(tgt).maxBy { it.area }
            fun p() = e.processors
            if (e.has(ModelIds.MODNET)) {
                val t = System.currentTimeMillis()
                val bg = p().replaceBackground(tgt, Processors.Background.Color(0x00FFFFFF))
                println("background (modnet) ${System.currentTimeMillis() - t} ms")
                val fg = p().foregroundMask(tgt)
                val c = fg[(face.centerY.toInt()) * tgt.width + face.centerX.toInt()]
                val corner = fg[5 * tgt.width + tgt.width - 5]
                println("  alpha on face=$c, top-right corner=$corner")
                assertTrue(c > 0.8f && corner < 0.2f)
                save(bg, "background_white.png")
                save(p().replaceBackground(tgt, Processors.Background.Transparent), "background_transparent.png")
                e.unloadAll()
            }
            if (e.has(ModelIds.SPAN_X4)) {
                val small = ImageOps.resize(tgt, 160, 85)
                val t = System.currentTimeMillis()
                val up = p().enhanceFrame(small, ModelIds.SPAN_X4)
                println("frame enhancer span x4 160x85 -> ${up.width}x${up.height} in ${System.currentTimeMillis() - t} ms")
                assertEquals(640, up.width)
                save(up, "frame_span_x4.png")
            }
            if (e.has(ModelIds.ESRGAN_X2)) {
                val small = ImageOps.resize(tgt, 160, 85)
                val up = p().enhanceFrame(small, ModelIds.ESRGAN_X2)
                assertEquals(320, up.width)
                save(up, "frame_esrgan_x2.png")
                e.unloadAll()
            }
            if (p().hasAge) {
                val t = System.currentTimeMillis()
                val older = p().modifyAge(tgt, face, 25f, 70f)
                println("age 25->70 ${System.currentTimeMillis() - t} ms")
                save(older, "age_70.png")
                val diff = older.pixels.indices.count { older.pixels[it] != tgt.pixels[it] }
                assertTrue(diff > 1000)
                e.unloadAll()
            }
            if (p().hasFaceEditor) {
                val t = System.currentTimeMillis()
                val ed = p().editFace(tgt, face, Processors.FaceEdit(mouthSmile = 0.8f, headYaw = 0.2f, eyeOpen = 0.5f))
                println("face editor ${System.currentTimeMillis() - t} ms")
                save(ed, "edit_smile.png")
                val r = e.recognizer.embed(ed, e.detect(ed).maxBy { it.area })
                val o = e.recognizer.embed(tgt, face)
                println("  identity kept after edit: ${Geometry.cosine(r, o)}")
                assertTrue(Geometry.cosine(r, o) > 0.5f)
                e.unloadAll()
            }
            e.mobileFaceSwap()?.let { mfs ->
                val id = mfs.buildIdentity(src, e.detect(src).maxBy { it.area })
                val frame = tgt.copy()
                val t = System.nanoTime()
                e.previewSwap(frame, listOf(face), id)
                println("mobilefaceswap preview ${(System.nanoTime() - t) / 1_000_000} ms")
                save(frame, "preview_mobilefaceswap.png")
                val s = e.buildSource("s", "S", listOf(src to e.detect(src).maxBy { it.area }))
                val after = identityAfter(e, frame, s)
                println("  preview identity $after")
                assertTrue(after > 0.35f)
            }
        }
    }

    @Test
    fun policyRules() {
        assumeTrue(ready() && has(ModelIds.OPEN_NSFW))
        engine().use { e ->
            val s = e.buildSource("s", "S", listOf(src to e.detect(src).maxBy { it.area }))
            val other = e.buildSource("o", "O", listOf(tgt to e.detect(tgt).maxBy { it.area }))
            assertTrue(e.policy.evaluate(false, listOf(other)).allowed)
            assertTrue(e.policy.evaluate(true, listOf(s)) is PolicyDecision.ExplicitRequiresVerification)
            e.policy.verifier = FakeIdentityVerifier(s.averaged)
            assertTrue(e.policy.evaluate(true, listOf(s)).allowed)
            assertTrue(e.policy.evaluate(true, listOf(other)) is PolicyDecision.ExplicitIdentityMismatch)
            assertTrue(!e.policy.evaluate(true, listOf(s, other)).allowed)
            e.policy.verifier = NoIdentityVerifier
            e.policy.internalLiveCaptureMode = true
            val live = SourceFace("l", "L", s.embeddings, SourceProvenance.LIVE_CAMERA)
            assertTrue(e.policy.evaluate(true, listOf(s)) is PolicyDecision.ExplicitRequiresLiveCapture)
            assertTrue(e.policy.evaluate(true, listOf(live)).allowed)
            assertTrue(!e.policy.evaluate(true, listOf(live, other)).allowed)
            e.policy.internalLiveCaptureMode = false
            assertTrue(!e.policy.evaluate(true, listOf(live)).allowed)
            val r = e.process(tgt, listOf(FaceMapping(s)), SwapConfig(), explicitOverride = true)
            assertEquals(0, r.facesSwapped)
            assertTrue(!e.policy.isExplicit(tgt))
        }
    }
}
