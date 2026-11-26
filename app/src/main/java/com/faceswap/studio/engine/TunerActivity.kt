package com.faceswap.studio.engine

import android.app.Activity
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.faceswap.core.Tensor
import com.faceswap.core.TensorFile
import com.faceswap.studio.data.ModelCatalog
import com.faceswap.studio.ui.theme.FaceSwapTheme
import com.faceswap.studio.ui.theme.Good
import com.faceswap.studio.ui.theme.Warn
import java.io.File
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Benchmarks every installed model on every candidate backend in the isolated `:tuner` process (see manifest), so a
 * backend that crashes natively only kills this process; the main app relaunches it and the crashed trial is
 * recorded and skipped.
 *
 * Correctness: each backend's outputs (all of them) are compared to the fp32 CPU result on the model's real sample
 * input (`<id>.sample.bin`, captured from the pipeline in CI) – fast-but-wrong backends (fp16 overflow, bad
 * quantization) are rejected. NPU candidates are first loaded with CPU fallback disabled to prove full offload;
 * if that fails they're retried with fallback and marked "partial".
 */
class TunerActivity : ComponentActivity() {
    companion object {
        const val RESULT_USER_CANCELLED = Activity.RESULT_FIRST_USER
        private const val MIN_SIMILARITY = 0.995
    }

    private data class Row(val text: String, val kind: Int) // 0 info, 1 ok, 2 fail
    private val rows = mutableStateListOf<Row>()
    private val progress = mutableStateOf(0f)
    private val headline = mutableStateOf("Preparing…")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { setResult(RESULT_USER_CANCELLED); finish(); android.os.Process.killProcess(android.os.Process.myPid()) }
        })
        setContent {
            FaceSwapTheme {
                Surface(Modifier.fillMaxSize()) {
                    Column(Modifier.statusBarsPadding().padding(20.dp)) {
                        Text("Tuning for this device", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Text(CpuInfo.chipName, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(12.dp))
                        Text(headline.value, style = MaterialTheme.typography.titleSmall)
                        LinearProgressIndicator(progress = { progress.value }, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp))
                        Text(
                            "Each model runs on the backends this chip offers and is checked against the CPU result on a real face. " +
                                "NPU compilation of the big models takes a while once – later app starts reuse it. If this screen closes it restarts on its own and skips what crashed. Back cancels.",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.height(10.dp))
                        val list = rememberLazyListState()
                        LaunchedEffect(rows.size) { if (rows.isNotEmpty()) list.animateScrollToItem(rows.size - 1) }
                        LazyColumn(
                            Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(12.dp)).padding(10.dp),
                            state = list, verticalArrangement = Arrangement.spacedBy(2.dp),
                        ) {
                            items(rows) { r ->
                                Text(r.text, style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                                    color = when (r.kind) { 1 -> Good; 2 -> Warn; else -> MaterialTheme.colorScheme.onSurface })
                            }
                        }
                    }
                }
            }
        }
        Thread({ runTuning() }, "tuner").start()
    }

    private fun ui(block: () -> Unit) = runOnUiThread(block)

    private fun runTuning() {
        val store = TuningStore(this)
        val trials = store.trials().toMutableList()
        store.crashedTrial()?.let { (m, b) ->
            trials.removeAll { it.model == m && it.backend == b }
            trials += Trial(m, b, "crashed", 0, Double.MAX_VALUE, 0.0)
            store.clearInProgress(); store.save(trials, false)
        }
        ui { trials.forEach { rows += describe(it) } }

        val modelsDir = File(filesDir, "models")
        val scope = store.scope
        val models = ModelCatalog.all.map { it.id }.filter { File(modelsDir, "$it.onnx").exists() && (scope == null || it in scope) }
        val plan = models.associateWith { Backends.candidates(this, modelsDir, it) }
        val total = plan.values.sumOf { it.size }.coerceAtLeast(1)
        val threads = CpuInfo.bigCores
        val started = System.currentTimeMillis()
        if (CpuInfo.isSnapdragon) ui { rows += Row("QNN: ${QnnPlugin.describe(this)}", 0) }
        if (scope != null) ui { rows += Row("Quick tune: ${models.size} models in use", 0) }

        fun result(model: String, backend: Backend) = trials.firstOrNull { it.model == model && it.backend == backend }
        fun best(model: String) = trials.filter { it.model == model && it.ok }.minOfOrNull { it.avgMs } ?: Double.MAX_VALUE

        var finished = 0
        for (model in models) {
            val file = File(modelsDir, "$model.onnx")
            val sample = File(modelsDir, "$model.sample.bin").takeIf { it.exists() }?.let { runCatching { TensorFile.read(it) }.getOrNull() }
            var reference: List<FloatArray>? = null
            // after a crash-restart, models whose every candidate already has a result are not re-run (CPU included)
            if (plan.getValue(model).all { b -> result(model, b) != null }) { finished += plan.getValue(model).size; continue }
            for (backend in plan.getValue(model)) {
                finished++
                val eta = if (finished > 1) ((System.currentTimeMillis() - started) / (finished - 1)) * (total - finished + 1) / 1000 else -1
                ui {
                    progress.value = (finished - 1).toFloat() / total
                    headline.value = "$model · ${backend.label}" + if (eta > 0) "  ·  ~${eta / 60}:%02d left".format(eta % 60) else ""
                }
                val existing = result(model, backend)
                // CPU always runs first (also after a restart) to have a reference output.
                if (existing != null && !(backend == Backend.CPU && reference == null && existing.ok)) continue
                // Candidates that can't win are skipped (each would cost a compile):
                //  - fp16 NPU when the INT8 graph fully offloaded and verified (int8 runs on the faster tensor units)
                //  - plain fp16 GPU when fp16 + fp32 accumulation verified (same speed class); fp32 GPU unless both failed
                val skip = when (backend) {
                    Backend.QNN_HTP -> result(model, Backend.QNN_HTP_INT8)?.status == "ok"
                    Backend.LITERT_GPU -> result(model, Backend.LITERT_GPU_ACC32)?.ok == true
                    Backend.LITERT_GPU_FP32 -> trials.any { it.model == model && it.ok && (it.backend == Backend.LITERT_GPU || it.backend == Backend.LITERT_GPU_ACC32) }
                    else -> false
                }
                if (skip) continue
                store.markInProgress(model, backend)
                var partial = false
                val bestSoFar = best(model)
                val t = runCatching { measure(file, model, backend, threads, sample, strict = backend.isQnn, bestSoFar) }.recoverCatching { e ->
                    // fp16 graph with a few unsupported ops: measure with CPU fallback and let the numbers decide.
                    // A partly offloaded INT8 graph practically never wins – no second compile for it.
                    if (backend != Backend.QNN_HTP) throw e
                    partial = true
                    measure(file, model, backend, threads, sample, strict = false, bestSoFar)
                }
                store.clearInProgress()
                val (firstMs, avgMs, out) = t.getOrElse { e ->
                    val tr = Trial(model, backend, "failed: ${(e.message ?: e.javaClass.simpleName).lines().first().take(90)}", 0, Double.MAX_VALUE, 0.0)
                    trials.removeAll { it.model == model && it.backend == backend }; trials += tr
                    store.save(trials, false); ui { rows += describe(tr) }
                    continue
                }
                if (backend == Backend.CPU) reference = out
                val sim = reference?.let { similarity(it, out) } ?: 1.0
                val status = when { sim < MIN_SIMILARITY -> "mismatch"; partial -> "ok-partial"; else -> "ok" }
                val tr = Trial(model, backend, status, firstMs, avgMs, sim)
                trials.removeAll { it.model == model && it.backend == backend }; trials += tr
                store.save(trials, false); ui { rows += describe(tr) }
            }
        }
        val took = (System.currentTimeMillis() - started) / 1000
        ui { rows += Row("Finished in ${took / 60}:%02d".format(took % 60), 0) }
        store.scope = null
        store.save(trials, true)
        val best = store.best()
        ui {
            progress.value = 1f
            headline.value = "Done"
            rows += Row("", 0)
            rows += Row("Selected:", 0)
            best.forEach { (m, b) ->
                val ms = trials.firstOrNull { it.model == m && it.backend == b }?.avgMs ?: 0.0
                val cpu = trials.firstOrNull { it.model == m && it.backend == Backend.CPU && it.ok }?.avgMs
                rows += Row("  $m → ${b.short} ${"%.0f".format(ms)} ms" + (cpu?.let { " (${"%.1f".format(it / ms)}×)" } ?: ""), 1)
            }
        }
        Thread.sleep(1500)
        ui { setResult(Activity.RESULT_OK); finish() }
        // end the isolated process so its native memory (NPU-compiled models) is released
        Thread.sleep(500); android.os.Process.killProcess(android.os.Process.myPid())
    }

    private fun describe(t: Trial) = when {
        t.ok -> Row("✓ ${t.model} · ${t.backend.short}: ${"%.1f".format(t.avgMs)} ms (first ${t.firstRunMs} ms)${if (t.status == "ok-partial") " · partial" else ""}", 1)
        t.status == "mismatch" -> Row("✗ ${t.model} · ${t.backend.short}: wrong output (${"%.4f".format(t.similarity)})", 2)
        else -> Row("✗ ${t.model} · ${t.backend.short}: ${t.status}", 2)
    }

    /**
     * Returns (load + first run ms, steady-state ms per run, outputs of the first run).
     * Steady state = one warm-up run, then runs until ~1.2 s are spent (max 6). A backend whose warm run is already
     * 2.5× slower than the best one measured for this model stops there – more runs can't change the decision.
     * (Old fixed 8 runs per backend made slow CPU baselines like FRAN or LivePortrait take minutes each.)
     */
    private fun measure(file: File, id: String, backend: Backend, threads: Int, sample: Map<String, Tensor>?, strict: Boolean, bestMs: Double): Triple<Long, Double, List<FloatArray>> {
        val t0 = System.nanoTime()
        Backends.load(this, id, file, backend, threads, strict).use { model ->
            val inputs = model.inputNames.associateWith { name ->
                sample?.get(name)?.takeIf { s -> model.inputShape(name)?.let { it.size == s.shape.size } != false } ?: randomInput(model.inputShape(name), name, id)
            }
            val out = model.run(inputs).values.map { it.data }
            val first = (System.nanoTime() - t0) / 1_000_000
            var s = System.nanoTime()
            model.run(inputs) // warm-up: caches, clocks, lazy GPU/NPU allocations
            val warm = (System.nanoTime() - s) / 1e6
            if (warm > bestMs * 2.5) return Triple(first, warm, out)
            var runs = 0
            s = System.nanoTime()
            do { model.run(inputs); runs++ } while (runs < 6 && (System.nanoTime() - s) < 1_200_000_000L)
            return Triple(first, (System.nanoTime() - s) / 1e6 / runs, out)
        }
    }

    private val rnd = Random(42)
    private fun randomInput(shape: LongArray?, name: String, id: String): Tensor {
        val s = (shape ?: longArrayOf()).map { if (it > 0) it else defaultDim(id) }.toLongArray()
        val n = s.fold(1L) { a, b -> a * b }.toInt().coerceAtLeast(1)
        val data = FloatArray(n) { rnd.nextFloat() }
        if (name.contains("source") || name.contains("emb")) { val l = sqrt(data.sumOf { (it * it).toDouble() }).toFloat(); for (i in data.indices) data[i] /= l }
        return Tensor(data, s)
    }

    private fun defaultDim(model: String): Long = when {
        model.startsWith("scrfd") || model.startsWith("yunet") -> 640
        model.startsWith("span") -> 128
        model.startsWith("real_esrgan") -> 256
        model.startsWith("modnet") || model.startsWith("bisenet") -> 512
        else -> 1
    }

    /** Worst cosine similarity over all outputs. */
    private fun similarity(a: List<FloatArray>, b: List<FloatArray>): Double {
        if (a.size != b.size) return 0.0
        return a.indices.minOf { i -> cosine(a[i], b[i]) }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        if (a.size != b.size) return 0.0
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) {
            val x = a[i].toDouble(); val y = b[i].toDouble()
            if (!x.isFinite() || !y.isFinite()) return 0.0
            d += x * y; na += x * x; nb += y * y
        }
        if (na == 0.0 && nb == 0.0) return 1.0
        return d / (sqrt(na * nb) + 1e-12)
    }
}
