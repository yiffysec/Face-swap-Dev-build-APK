package com.faceswap.studio.engine

import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.EnumSet

/** Concrete execution backends the tuner measures. */
enum class Backend(val label: String) {
    CPU("CPU"),
    XNNPACK("XNNPACK"),
    NNAPI("NNAPI fp32"),
    NNAPI_FP16("NNAPI fp16 (NPU)"),
}

object CpuInfo {
    /**
     * Number of "big" cores (max freq ≥ 75% of the fastest core). Pixel 8 Pro → 5 (X3 + 4×A715),
     * Galaxy S24 Ultra → 6 (X4 + 5×A720). Running ORT on little cores only slows it down.
     */
    val bigCores: Int by lazy {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { i ->
            runCatching { File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrNull()
        }
        if (freqs.isEmpty()) 4 else {
            val top = freqs.max()
            freqs.count { it >= top * 0.75 }.coerceIn(2, 8)
        }
    }
}

object Backends {
    fun options(backend: Backend, threads: Int): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(threads)
        setInterOpNumThreads(1)
        addConfigEntry("session.intra_op.allow_spinning", "1")
        when (backend) {
            Backend.CPU -> Unit
            Backend.XNNPACK -> {
                // XNNPACK owns its own thread pool; ORT's pool is kept to 1 to avoid oversubscription.
                setIntraOpNumThreads(1)
                addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            }
            Backend.NNAPI -> addNnapi(EnumSet.noneOf(NNAPIFlags::class.java))
            Backend.NNAPI_FP16 -> addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
        }
    }
}

/** One measured (model, backend) combination. */
data class Trial(val model: String, val backend: Backend, val status: String, val firstRunMs: Long, val avgMs: Double, val similarity: Double) {
    fun toJson() = JSONObject().put("model", model).put("backend", backend.name).put("status", status)
        .put("firstRunMs", firstRunMs).put("avgMs", avgMs).put("similarity", similarity)

    companion object {
        fun from(o: JSONObject) = Trial(o.getString("model"), Backend.valueOf(o.getString("backend")), o.getString("status"),
            o.optLong("firstRunMs"), o.optDouble("avgMs"), o.optDouble("similarity"))
    }
}

/** Persisted tuning state, shared by the main process and the isolated `:tuner` process via files. */
class TuningStore(context: Context) {
    private val dir = File(context.filesDir, "tuning").apply { mkdirs() }
    private val resultsFile = File(dir, "results.json")
    private val inProgressFile = File(dir, "in_progress")

    val device = "${Build.MANUFACTURER} ${Build.MODEL} (${socModel()})"

    fun trials(): List<Trial> = runCatching {
        val a = JSONObject(resultsFile.readText()).getJSONArray("trials")
        (0 until a.length()).map { Trial.from(a.getJSONObject(it)) }
    }.getOrDefault(emptyList())

    fun isComplete(): Boolean = runCatching { JSONObject(resultsFile.readText()).optBoolean("complete") }.getOrDefault(false)

    /** Best working backend per model id (file name without .onnx). */
    fun best(): Map<String, Backend> = runCatching {
        val o = JSONObject(resultsFile.readText()).getJSONObject("best")
        o.keys().asSequence().associateWith { Backend.valueOf(o.getString(it)) }
    }.getOrDefault(emptyMap())

    val version: Long get() = if (resultsFile.exists()) resultsFile.lastModified() else 0L

    @Synchronized
    fun save(trials: List<Trial>, complete: Boolean) {
        val best = JSONObject()
        trials.filter { it.status == "ok" }.groupBy { it.model }.forEach { (m, ts) -> best.put(m, ts.minBy { it.avgMs }.backend.name) }
        val o = JSONObject().put("device", device).put("complete", complete).put("best", best)
            .put("trials", JSONArray(trials.map { it.toJson() }))
        val tmp = File(dir, "results.tmp"); tmp.writeText(o.toString(2)); tmp.renameTo(resultsFile)
    }

    fun reset() { resultsFile.delete(); inProgressFile.delete() }

    /** Marker written right before a risky trial; if it still exists on the next start, that trial crashed the process. */
    fun markInProgress(model: String, backend: Backend) = inProgressFile.writeText("$model|${backend.name}")
    fun clearInProgress() = inProgressFile.delete()
    fun crashedTrial(): Pair<String, Backend>? = runCatching {
        val (m, b) = inProgressFile.readText().split("|"); m to Backend.valueOf(b)
    }.getOrNull()
}

fun socModel(): String = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
