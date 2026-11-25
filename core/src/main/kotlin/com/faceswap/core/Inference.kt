package com.faceswap.core

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.io.File
import java.nio.DoubleBuffer
import java.nio.FloatBuffer

/** Dense float tensor (row-major). Runtime-neutral so models can run on ONNX Runtime or LiteRT. */
class Tensor(val data: FloatArray, val shape: LongArray) {
    constructor(data: FloatArray, vararg dims: Int) : this(data, LongArray(dims.size) { dims[it].toLong() })

    val size get() = data.size
    fun scalar() = data[0]

    companion object {
        fun scalar(v: Float) = Tensor(floatArrayOf(v), LongArray(0))
    }
}

/** A loaded model on some runtime/backend. Implementations must be safe to call from one thread at a time per call. */
interface InferenceModel : Closeable {
    val name: String
    val inputNames: List<String>
    val outputNames: List<String>

    /** Static input shape if known (dynamic dims are -1). */
    fun inputShape(name: String): LongArray?

    /** Runs the model. Returns outputs by name (in [outputNames] order). */
    fun run(inputs: Map<String, Tensor>): Map<String, Tensor>

    /** Human-readable backend description, e.g. "ORT/XNNPACK", "LiteRT/GPU". */
    val backend: String
}

/**
 * Process-wide workload hint. Long jobs (video, live) set [sustained] so NPU backends switch from "burst" clocks
 * (fastest single photo, throttles within seconds) to sustained high-performance clocks.
 */
object PerfHints {
    @Volatile var sustained = false
}

/** Loads a model file on the configured backend. The app supplies one per model (tuned backend choice). */
fun interface ModelLoader {
    fun load(file: File): InferenceModel
}

/** ONNX Runtime implementation. */
class OrtModel(
    private val env: OrtEnvironment,
    file: File,
    options: OrtSession.SessionOptions,
    override val backend: String = "ORT/CPU",
    /** Per-run config entries (e.g. QNN "qnn.htp_perf_mode"), evaluated on every run. */
    private val runConfig: (() -> Map<String, String>)? = null,
) : InferenceModel {
    private val session: OrtSession = env.createSession(file.absolutePath, options)
    override val name: String = file.nameWithoutExtension
    override val inputNames: List<String> = session.inputNames.toList()
    override val outputNames: List<String> = session.outputNames.toList()
    private val inputTypes: Map<String, OnnxJavaType> =
        session.inputInfo.mapValues { (it.value.info as? TensorInfo)?.type ?: OnnxJavaType.FLOAT }

    override fun inputShape(name: String): LongArray? = (session.inputInfo[name]?.info as? TensorInfo)?.shape

    override fun run(inputs: Map<String, Tensor>): Map<String, Tensor> {
        val ort = HashMap<String, OnnxTensor>()
        val ro = runConfig?.invoke()?.takeIf { it.isNotEmpty() }?.let { cfg ->
            OrtSession.RunOptions().also { r -> cfg.forEach { (k, v) -> r.addRunConfigEntry(k, v) } }
        }
        try {
            for ((k, t) in inputs) {
                ort[k] = if (inputTypes[k] == OnnxJavaType.DOUBLE)
                    OnnxTensor.createTensor(env, DoubleBuffer.wrap(DoubleArray(t.size) { t.data[it].toDouble() }), t.shape)
                else OnnxTensor.createTensor(env, FloatBuffer.wrap(t.data), t.shape)
            }
            (if (ro != null) session.run(ort, ro) else session.run(ort)).use { res ->
                val out = LinkedHashMap<String, Tensor>()
                for (i in 0 until res.size()) {
                    val v = res.get(i) as OnnxTensor
                    val shape = v.info.shape
                    val data = when (v.info.type) {
                        OnnxJavaType.DOUBLE -> v.doubleBuffer.let { b -> FloatArray(b.remaining()) { b.get().toFloat() } }
                        OnnxJavaType.INT64 -> v.longBuffer.let { b -> FloatArray(b.remaining()) { b.get().toFloat() } }
                        else -> v.floatBuffer.let { b -> FloatArray(b.remaining()).also { a -> b.get(a) } }
                    }
                    out[outputNames[i]] = Tensor(data, shape)
                }
                return out
            }
        } finally {
            ort.values.forEach { it.close() }
            ro?.close()
        }
    }

    override fun close() = session.close()

    companion object {
        fun cpuLoader(threads: Int = 4) = ModelLoader { f ->
            val env = OrtEnvironment.getEnvironment()
            OrtModel(env, f, OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) })
        }
    }
}

/** Convenience: first output. */
fun Map<String, Tensor>.first(): Tensor = values.first()
