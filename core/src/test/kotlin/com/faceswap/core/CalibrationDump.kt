package com.faceswap.core

import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import javax.imageio.ImageIO

/** Wraps a model and saves its real inputs (up to a byte budget) for post-training quantization. */
private class Budget(val maxSamples: Int, val maxBytes: Long) { val n = AtomicInteger(0); val bytes = AtomicLong(0) }

private class RecordingModel(private val inner: InferenceModel, private val dir: File, budget: Budget) : InferenceModel by inner {
    private val n = budget.n
    private val bytes = budget.bytes
    private val maxSamples = budget.maxSamples
    private val maxBytes = budget.maxBytes
    override fun run(inputs: Map<String, Tensor>): Map<String, Tensor> {
        val b = TensorFile.bytes(inputs)
        if (n.get() < maxSamples && bytes.get() + b <= maxBytes) {
            val i = n.getAndIncrement()
            if (i < maxSamples) { bytes.addAndGet(b); TensorFile.write(File(dir, "%03d.bin".format(i)), inputs) }
        }
        return inner.run(inputs)
    }
}

/**
 * Runs the real pipeline over the example frames and records every model's inputs into -Dcalib.dir/<model id>/.
 * tools/quantize_qnn.py turns these into calibrated w8a16 QDQ models for the Snapdragon NPU. Skipped unless
 * -Dcalib.dir is set (CI only).
 */
class CalibrationDump {
    private val dir = File(System.getProperty("models.dir") ?: "")
    private val calib = System.getProperty("calib.dir")?.takeIf { it.isNotBlank() }?.let(::File)

    private fun load(f: File): RgbImage? = runCatching {
        val bi = ImageIO.read(f)
        val px = IntArray(bi.width * bi.height)
        bi.getRGB(0, 0, bi.width, bi.height, px, 0, bi.width)
        RgbImage(bi.width, bi.height, px)
    }.getOrNull()

    private fun flip(img: RgbImage) = RgbImage(img.width, img.height).also { o ->
        for (y in 0 until img.height) for (x in 0 until img.width) o.pixels[y * img.width + x] = img.pixels[y * img.width + img.width - 1 - x]
    }

    private fun gain(img: RgbImage, g: Float) = RgbImage(img.width, img.height).also { o ->
        for (i in img.pixels.indices) {
            val p = img.pixels[i]
            fun c(s: Int) = ((((p shr s) and 0xFF) * g).toInt()).coerceIn(0, 255)
            o.pixels[i] = (0xFF shl 24) or (c(16) shl 16) or (c(8) shl 8) or c(0)
        }
    }

    @Test
    fun dump() {
        assumeTrue(calib != null && File(dir, "${ModelIds.ARCFACE}.onnx").exists())
        val out = calib!!
        val budget = (System.getProperty("calib.mb")?.toLongOrNull() ?: 96L) shl 20
        val maxSamples = System.getProperty("calib.samples")?.toIntOrNull() ?: 48
        val heavy = setOf(ModelIds.FRAN, ModelIds.LP_GENERATOR, ModelIds.GPEN_512, ModelIds.GFPGAN, ModelIds.CODEFORMER, ModelIds.RESTOREFORMER)
        val cpu = OrtModel.cpuLoader(Runtime.getRuntime().availableProcessors())
        val budgets = ConcurrentHashMap<String, Budget>()
        // models left out (memory/time on small CI runners): treated as not installed
        val skip = System.getProperty("calib.skip").orEmpty().split(',').map { it.trim() }.filter { it.isNotEmpty() }.toSet()
        val e = FaceSwapEngine({ id -> File(dir, "$id.onnx").takeIf { id !in skip } }, { id, f ->
            val b = budgets.getOrPut(id) { Budget(if (id in heavy) maxSamples / 3 else maxSamples, if (id in heavy) budget * 2 else budget) }
            RecordingModel(cpu.load(f), File(out, id), b)
        }, emapCache = File(out, "emap.bin"))

        val frames = (listOf(File(dir, "ex/target.jpg"), File(dir, "ex/source.jpg")) +
            (File(dir, "ex/frames").listFiles()?.filter { it.extension == "jpg" }?.sorted() ?: emptyList()))
            .mapNotNull { load(it) }
        val variants = frames.flatMap { listOf(it, flip(it), gain(it, 0.75f), gain(it, 1.2f)) }
        println("calibration: ${frames.size} frames → ${variants.size} variants")

        // identities: every distinct face we can find (source + targets), so swapper latents cover real embeddings
        val sources = variants.take(12).mapNotNull { img -> e.detect(img).maxByOrNull { it.area }?.let { img to it } }
            .mapIndexed { i, (img, f) -> e.buildSource("s$i", "S$i", listOf(img to f)) }
        check(sources.isNotEmpty()) { "no faces in calibration frames" }

        val swappers = ModelIds.SWAPPERS.filter { e.has(it) }
        for ((vi, img) in variants.withIndex()) {
            runCatching { e.policy.isExplicit(img) }
            val src = sources[vi % sources.size]
            for (det in listOf(ModelIds.YUNET, ModelIds.SCRFD).filter { e.has(it) }) e.detect(img, SwapConfig(detectorId = det))
            for (sw in swappers) {
                val cfg = SwapConfig(swapperId = sw, detectorId = ModelIds.YUNET, maskTypes = setOf(MaskType.BOX, MaskType.MODEL, MaskType.REGION),
                    pixelBoost = if (sw.startsWith("inswapper")) 256 else 256)
                runCatching { e.process(img, listOf(FaceMapping(src)), cfg, explicitOverride = false) }
            }
            for (enh in ModelIds.ENHANCERS.filter { e.has(it) }) if (vi % (if (enh in heavy) 3 else 1) == 0)
                runCatching { e.process(img, listOf(FaceMapping(src)), SwapConfig(swapperId = swappers.first(), enhancerId = enh), explicitOverride = false) }
            val face = e.detect(img).maxByOrNull { it.area } ?: continue
            val p = e.processors
            if (vi % 2 == 0) {
                if (e.has(ModelIds.MODNET) || e.has(ModelIds.U2NETP)) runCatching { p.foregroundMask(img) }
                val small = ImageOps.resize(img, 240, 240 * img.height / img.width)
                for (up in ModelIds.FRAME_ENHANCERS.filter { e.has(it) }) runCatching { p.enhanceFrame(small, up) }
            }
            if (vi % 3 == 0) {
                if (p.hasAge) runCatching { p.modifyAge(img, face, 30f, (20 + vi * 7 % 60).toFloat()) }
                if (p.hasLivePortrait) runCatching { e.process(img, listOf(FaceMapping(src)), SwapConfig(swapperId = swappers.first(), expressionRestore = 0.8f), explicitOverride = false) }
                if (p.hasFaceEditor) runCatching { p.editFace(img, face, Processors.FaceEdit(mouthSmile = 0.6f, eyeOpen = 0.4f, lipOpen = 0.3f, headYaw = 0.1f)) }
            }
            if (vi % 4 == 3) e.unloadAll() // keep peak memory bounded on CI runners
        }
        e.close()
        val summary = out.listFiles()?.filter { it.isDirectory }?.associate { it.name to (it.listFiles()?.size ?: 0) }
        println("calibration samples: $summary")
    }
}
