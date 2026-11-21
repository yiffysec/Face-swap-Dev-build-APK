package com.faceswap.studio.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.app.Activity
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.faceswap.studio.data.ModelCatalog
import com.faceswap.studio.ui.theme.FaceSwapTheme
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Benchmarks every model on every backend, in the isolated `:tuner` process (see manifest), so a backend that
 * crashes natively only kills this process. The main app relaunches it, and the crashed trial is recorded and
 * skipped. Each backend's output is checked against the CPU result, so a backend that is fast but numerically
 * wrong (e.g. fp16 overflow on an NPU) is rejected.
 */
class TunerActivity : ComponentActivity() {
    companion object { const val RESULT_USER_CANCELLED = Activity.RESULT_FIRST_USER }

    private val rows = mutableStateListOf<String>()
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
                    Column(Modifier.padding(20.dp)) {
                        Text("Tuning for this device", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Text(headline.value, Modifier.padding(vertical = 8.dp))
                        LinearProgressIndicator(progress = { progress.value }, modifier = Modifier.fillMaxWidth())
                        Text("Each model is timed on CPU, XNNPACK and NNAPI (TPU/NPU). NPU compilation of the large models can take a minute. If the screen closes, it restarts automatically and skips whatever crashed.",
                            Modifier.padding(vertical = 10.dp), style = MaterialTheme.typography.bodySmall)
                        LazyColumn { items(rows) { r -> Row { Text(r, style = MaterialTheme.typography.bodyMedium) } } }
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
        val models = ModelCatalog.all.map { it.id }.filter { File(modelsDir, "$it.onnx").exists() }
        val total = models.size * Backend.entries.size
        val env = OrtEnvironment.getEnvironment()
        val threads = CpuInfo.bigCores
        val rnd = Random(42)

        for (model in models) {
            val file = File(modelsDir, "$model.onnx")
            var reference: FloatArray? = null
            for (backend in Backend.entries) {
                val done = trials.count { it.model in models }
                ui { progress.value = done.toFloat() / total; headline.value = "$model · ${backend.label}" }
                val existing = trials.firstOrNull { it.model == model && it.backend == backend }
                if (existing != null && (backend != Backend.CPU || reference != null || existing.status != "ok")) continue
                // CPU is always run first (also after a restart) to have a reference output.
                store.markInProgress(model, backend)
                val t = runCatching { measure(env, file, backend, threads, rnd, model) }
                store.clearInProgress()
                val (firstMs, avgMs, out) = t.getOrElse { e ->
                    val tr = Trial(model, backend, "failed: ${e.message?.take(80)}", 0, Double.MAX_VALUE, 0.0)
                    trials.removeAll { it.model == model && it.backend == backend }; trials += tr
                    store.save(trials, false); ui { rows += describe(tr) }
                    continue
                }
                if (backend == Backend.CPU) reference = out
                val sim = reference?.let { cosine(it, out) } ?: 1.0
                val status = if (sim >= 0.995) "ok" else "mismatch"
                val tr = Trial(model, backend, status, firstMs, avgMs, sim)
                if (trials.none { it.model == model && it.backend == backend }) {
                    trials += tr; store.save(trials, false); ui { rows += describe(tr) }
                }
            }
        }
        store.save(trials, true)
        val best = store.best()
        ui {
            progress.value = 1f
            headline.value = "Done"
            rows += ""
            rows += "Selected:"
            best.forEach { (m, b) -> rows += "  $m → ${b.label}" }
        }
        Thread.sleep(1500)
        ui { setResult(Activity.RESULT_OK); finish() }
        // end the isolated process so its native memory (NPU-compiled models) is released
        Thread.sleep(500); android.os.Process.killProcess(android.os.Process.myPid())
    }

    private fun describe(t: Trial) = when {
        t.status == "ok" -> "✓ ${t.model} · ${t.backend.label}: ${"%.0f".format(t.avgMs)} ms (first ${t.firstRunMs} ms)"
        t.status == "mismatch" -> "✗ ${t.model} · ${t.backend.label}: wrong output (similarity ${"%.3f".format(t.similarity)})"
        else -> "✗ ${t.model} · ${t.backend.label}: ${t.status}"
    }

    /** Returns (first run incl. compilation ms, average steady-state ms, first output). */
    private fun measure(env: OrtEnvironment, file: File, backend: Backend, threads: Int, rnd: Random, model: String): Triple<Long, Double, FloatArray> {
        val t0 = System.nanoTime()
        Backends.options(backend, threads).use { opts ->
            env.createSession(file.absolutePath, opts).use { session ->
                val inputs = session.inputInfo.mapValues { (name, info) ->
                    val shape = (info.info as TensorInfo).shape.map { if (it > 0) it else defaultDim(model, name) }.toLongArray()
                    val n = shape.fold(1L) { a, b -> a * b }.toInt()
                    val data = FloatArray(n) { rnd.nextFloat() }
                    if (name.contains("source")) { val s = sqrt(data.sumOf { (it * it).toDouble() }).toFloat(); for (i in data.indices) data[i] /= s }
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
                }
                try {
                    var out: FloatArray
                    session.run(inputs).use { out = first(it) }
                    val first = (System.nanoTime() - t0) / 1_000_000
                    val runs = if (file.length() > 100_000_000) 3 else 6
                    val s = System.nanoTime()
                    repeat(runs) { session.run(inputs).use { } }
                    return Triple(first, (System.nanoTime() - s) / 1e6 / runs, out)
                } finally {
                    inputs.values.forEach { it.close() }
                }
            }
        }
    }

    private fun defaultDim(model: String, input: String): Long = if (model.startsWith("scrfd")) 640 else 1

    private fun first(r: OrtSession.Result): FloatArray {
        val t = r.get(0) as OnnxTensor
        val fb = t.floatBuffer
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        if (a.size != b.size) return 0.0
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        if (na == 0.0 && nb == 0.0) return 1.0
        return d / (sqrt(na * nb) + 1e-12)
    }
}
