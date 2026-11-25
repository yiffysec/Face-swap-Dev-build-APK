package com.faceswap.studio.engine

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtEpDevice
import ai.onnxruntime.OrtHardwareDevice
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.Build
import android.util.Log
import com.faceswap.core.InferenceModel
import com.faceswap.core.OrtModel
import com.faceswap.core.PerfHints
import com.google.ai.edge.litert.CompiledModel
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.EnumSet

/**
 * Concrete execution backends the tuner measures. Names are persisted in tuning results – only append.
 *
 * Research notes that shaped this list (see research/performance-game-plan.md):
 *  - Snapdragon 8 Gen 3 ships no NNAPI driver (NNAPI = reference CPU code there), so NNAPI is never tried on it.
 *  - The Hexagon NPU is fastest on calibrated integer graphs: QNN_HTP_INT8 runs the w8a16 QDQ twin built in CI.
 *  - Tensor G3's TPU is only reachable through (deprecated) NNAPI; its GPU (Mali) goes through LiteRT ML Drift,
 *    where fp16 can overflow → the FP16-with-FP32-accumulation and FP32 variants.
 */
enum class Backend(val label: String, val short: String) {
    CPU("CPU", "CPU"),
    XNNPACK("XNNPACK", "XNNPACK"),
    NNAPI("NNAPI fp32", "NNAPI"),
    NNAPI_FP16("NNAPI fp16 (TPU/NPU)", "NNAPI16"),
    QNN_HTP("QNN NPU fp16", "NPU16"),
    QNN_GPU("QNN GPU (Adreno)", "QNN-GPU"),
    LITERT_GPU("LiteRT GPU fp16", "GPU16"),
    QNN_HTP_INT8("QNN NPU int8 (w8a16)", "NPU8"),
    LITERT_GPU_ACC32("LiteRT GPU fp16 + fp32 accum", "GPU16A"),
    LITERT_GPU_FP32("LiteRT GPU fp32", "GPU32"),
    ;

    val isQnn get() = this == QNN_HTP || this == QNN_HTP_INT8 || this == QNN_GPU
    val isLiteRt get() = this == LITERT_GPU || this == LITERT_GPU_ACC32 || this == LITERT_GPU_FP32
    val isNpu get() = this == QNN_HTP || this == QNN_HTP_INT8 || this == NNAPI_FP16
}

fun socModel(): String = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE

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

    val isSnapdragon: Boolean by lazy {
        val s = socModel().lowercase()
        s.contains("qti") || s.contains("qualcomm") || s.contains(" sm") || Build.HARDWARE.contains("qcom", true)
    }
    val isTensor: Boolean by lazy { socModel().contains("Google", true) || Build.HARDWARE.startsWith("zuma") }
    val chipName: String get() = when {
        isSnapdragon -> "Snapdragon (${socModel()})"
        isTensor -> "Google Tensor (${socModel()})"
        else -> socModel()
    }
}

/** Qualcomm's ONNX Runtime QNN plugin EP (com.qualcomm.qti:onnxruntime-android-qnn, snapdragon flavor only). */
object QnnPlugin {
    private const val LIB = "libonnxruntime_providers_qnn.so"
    @Volatile private var registered: Boolean? = null

    @Synchronized
    fun devices(context: Context): List<OrtEpDevice> {
        val env = OrtEnvironment.getEnvironment()
        if (registered == null) registered = runCatching {
            val lib = File(context.applicationInfo.nativeLibraryDir, LIB)
            check(lib.exists()) { "QNN plugin not packaged" }
            env.registerExecutionProviderLibrary("QNN", lib.absolutePath)
            true
        }.onFailure { Log.i("QnnPlugin", "QNN plugin unavailable: ${it.message}") }.getOrDefault(false)
        if (registered != true) return emptyList()
        return runCatching { env.epDevices.filter { it.epName.contains("QNN", ignoreCase = true) } }.getOrDefault(emptyList())
    }

    fun available(context: Context) = devices(context).isNotEmpty()

    fun describe(context: Context): String = devices(context).joinToString { "${it.epName}/${it.device.type}" }.ifEmpty { "not available" }
}

object Backends {
    /** Backends worth trying on this device for one model (the tuner still verifies each one). */
    fun candidates(context: Context, modelsDir: File, id: String): List<Backend> = buildList {
        add(Backend.CPU); add(Backend.XNNPACK)
        if (CpuInfo.isSnapdragon) {
            add(Backend.QNN_HTP)
            if (File(modelsDir, "$id.qdq.onnx").exists()) add(Backend.QNN_HTP_INT8)
            add(Backend.QNN_GPU)
        } else {
            add(Backend.NNAPI_FP16); add(Backend.NNAPI)
        }
        if (LiteRtModel.available(modelsDir, id)) { add(Backend.LITERT_GPU); add(Backend.LITERT_GPU_ACC32); add(Backend.LITERT_GPU_FP32) }
    }

    /** The model file a backend runs (INT8 runs the calibrated QDQ twin, LiteRT the .tflite twin). */
    fun fileFor(onnx: File, id: String, backend: Backend): File = when {
        backend == Backend.QNN_HTP_INT8 -> File(onnx.parentFile, "$id.qdq.onnx")
        backend.isLiteRt -> File(onnx.parentFile, "$id.tflite")
        else -> onnx
    }

    /** Options newer than some QNN EP builds: dropped on retry if the EP rejects them. */
    private val optionalQnnKeys = setOf("enable_htp_fp16_clamp_overflow", "htp_graph_finalization_optimization_mode", "qnn_context_priority", "offload_graph_io_quantization")

    private fun qnnOptions(backend: Backend): Map<String, String> = when (backend) {
        Backend.QNN_GPU -> mapOf("backend_type" to "gpu", "backend_path" to "libQnnGpu.so", "qnn_context_priority" to "high")
        else -> mapOf(
            "backend_type" to "htp",
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "htp_graph_finalization_optimization_mode" to "3",
            "enable_htp_fp16_precision" to "1",
            // QAIRT 2.49+: clamp fp16 conv overflow to ±65504 instead of producing Inf/NaN (GAN restorers)
            "enable_htp_fp16_clamp_overflow" to "1",
            "qnn_context_priority" to "high",
        ) + if (backend == Backend.QNN_HTP_INT8) mapOf(
            // INT8 graphs: input quantize / output dequantize on the NPU too, so the whole graph is offloaded
            // (with CPU-side Q/DQ the strict full-offload check could never pass)
            "offload_graph_io_quantization" to "0",
        ) else emptyMap()
    }

    /**
     * @param strict NPU/GPU only: every node must run on the accelerator (session creation fails otherwise). The tuner
     * uses it to tell "fully offloaded" from "partly on CPU" – a partial offload often loses to plain CPU.
     */
    fun ortOptions(context: Context, id: String, backend: Backend, threads: Int, strict: Boolean = false, writeContext: File? = null, minimalQnn: Boolean = false): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
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
            Backend.QNN_HTP, Backend.QNN_HTP_INT8, Backend.QNN_GPU -> {
                if (strict) addConfigEntry("session.disable_cpu_ep_fallback", "1")
                // First load: compile and save the graph as an EP-context model; later loads open that file directly.
                if (writeContext != null) {
                    addConfigEntry("ep.context_enable", "1")
                    addConfigEntry("ep.context_file_path", writeContext.absolutePath)
                }
                val wanted = if (backend == Backend.QNN_GPU) OrtHardwareDevice.OrtHardwareDeviceType.GPU else OrtHardwareDevice.OrtHardwareDeviceType.NPU
                val devices = QnnPlugin.devices(context).filter { it.device.type == wanted }
                val opts = qnnOptions(backend).let { o -> if (minimalQnn) o - optionalQnnKeys else o }
                if (devices.isNotEmpty()) addExecutionProvider(devices, opts - "backend_path" - "backend_type")
                else addQnn(opts - "backend_type") // legacy builds with QNN compiled into ORT
            }
            else -> Unit
        }
    }

    /** NPU clock mode per run: burst for one-off photos, sustained for video/live (burst throttles within seconds). */
    private val qnnRunConfig: () -> Map<String, String> = {
        mapOf("qnn.htp_perf_mode" to if (PerfHints.sustained) "sustained_high_performance" else "burst")
    }

    /** Loads [onnx] (or its INT8 / TFLite twin) on [backend]. Throws if the backend can't take it. */
    fun load(context: Context, id: String, onnx: File, backend: Backend, threads: Int, strict: Boolean = false): InferenceModel {
        val file = fileFor(onnx, id, backend)
        check(file.exists()) { "${file.name} not installed" }
        return when {
            backend == Backend.LITERT_GPU -> LiteRtModel(context, file, CompiledModel.GpuOptions.Precision.FP16)
            backend == Backend.LITERT_GPU_ACC32 -> LiteRtModel(context, file, CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM)
            backend == Backend.LITERT_GPU_FP32 -> LiteRtModel(context, file, CompiledModel.GpuOptions.Precision.FP32)
            backend.isQnn -> loadQnn(context, id, file, backend, threads, strict)
            else -> OrtModel(OrtEnvironment.getEnvironment(), file, ortOptions(context, id, backend, threads), "ORT/${backend.label}")
        }
    }

    private fun ctxDir(context: Context) = File(context.filesDir, "qnn_ctx").apply { mkdirs() }

    /** Deletes cached NPU graphs (all backends) for a model – call when its file is replaced or deleted. */
    fun clearCompiled(context: Context, id: String) {
        ctxDir(context).listFiles()?.filter { it.name.startsWith("${id}_") }?.forEach { it.delete() }
    }

    /**
     * QNN with a compiled-graph cache. The EP-context model is only valid for the exact source file and QNN SDK
     * it was built from, so it carries a stamp; a stale or unreadable cache is discarded and rebuilt.
     */
    private fun loadQnn(context: Context, id: String, file: File, backend: Backend, threads: Int, strict: Boolean): InferenceModel {
        val env = OrtEnvironment.getEnvironment()
        val prefix = "${id}_${backend.name}"
        val ctx = File(ctxDir(context), "$prefix.onnx_ctx.onnx")
        val stamp = File(ctxDir(context), "$prefix.stamp")
        val want = "${file.length()}:${file.lastModified()}:${context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime}"
        val label = "ORT/${backend.label}"
        if (ctx.exists() && stamp.exists() && stamp.readText() == want) {
            runCatching { return OrtModel(env, ctx, ortOptions(context, id, backend, threads, strict), "$label (cached)", qnnRunConfig) }
                .onFailure { Log.w("Backends", "cached NPU graph for $id unusable, recompiling", it) }
        }
        ctxDir(context).listFiles()?.filter { it.name.startsWith(prefix) }?.forEach { it.delete() }
        val model = try {
            OrtModel(env, file, ortOptions(context, id, backend, threads, strict, writeContext = ctx), label, qnnRunConfig)
        } catch (e: Exception) {
            // an EP build that doesn't know one of the newer options rejects the whole session – retry without them
            val msg = e.message.orEmpty()
            if (!Regex("option|unknown|invalid|unsupported key", RegexOption.IGNORE_CASE).containsMatchIn(msg)) throw e
            Log.w("Backends", "QNN rejected options for $id, retrying with the minimal set: $msg")
            ctxDir(context).listFiles()?.filter { it.name.startsWith(prefix) }?.forEach { it.delete() }
            OrtModel(env, file, ortOptions(context, id, backend, threads, strict, writeContext = ctx, minimalQnn = true), label, qnnRunConfig)
        }
        if (ctx.exists()) stamp.writeText(want)
        return model
    }
}

/** One measured (model, backend) combination. */
data class Trial(val model: String, val backend: Backend, val status: String, val firstRunMs: Long, val avgMs: Double, val similarity: Double) {
    /** "ok" = verified; "ok-partial" = verified but some nodes fell back to CPU. */
    val ok get() = status.startsWith("ok")

    fun toJson() = JSONObject().put("model", model).put("backend", backend.name).put("status", status)
        .put("firstRunMs", firstRunMs).put("avgMs", if (avgMs.isFinite()) avgMs else -1.0).put("similarity", similarity)

    companion object {
        fun from(o: JSONObject) = runCatching {
            val avg = o.optDouble("avgMs")
            Trial(o.getString("model"), Backend.valueOf(o.getString("backend")), o.getString("status"),
                o.optLong("firstRunMs"), if (avg < 0 || avg.isNaN()) Double.MAX_VALUE else avg, o.optDouble("similarity"))
        }.getOrNull()
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
        (0 until a.length()).mapNotNull { Trial.from(a.getJSONObject(it)) }
    }.getOrDefault(emptyList())

    fun isComplete(): Boolean = runCatching { JSONObject(resultsFile.readText()).optBoolean("complete") }.getOrDefault(false)

    /** Best working backend per model id (file name without .onnx). */
    fun best(): Map<String, Backend> = runCatching {
        val o = JSONObject(resultsFile.readText()).getJSONObject("best")
        o.keys().asSequence().mapNotNull { k -> runCatching { k to Backend.valueOf(o.getString(k)) }.getOrNull() }.toMap()
    }.getOrDefault(emptyMap())

    val version: Long get() = if (resultsFile.exists()) resultsFile.lastModified() else 0L

    @Synchronized
    fun save(trials: List<Trial>, complete: Boolean) {
        val best = JSONObject()
        trials.filter { it.ok }.groupBy { it.model }.forEach { (m, ts) -> best.put(m, ts.minBy { it.avgMs }.backend.name) }
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
