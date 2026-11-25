package com.faceswap.studio.engine

import android.content.Context
import com.faceswap.core.InferenceModel
import com.faceswap.core.Tensor
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.File

/**
 * LiteRT (TFLite) model on the GPU accelerator (OpenCL/GL – Mali on Pixel, Adreno on Snapdragon).
 * Uses the `<id>.tflite.json` sidecar written by tools/convert_tflite.py to map ONNX tensor names/layouts onto the
 * TFLite signature, so the engine code is identical across runtimes.
 */
class LiteRtModel(
    context: Context,
    tflite: File,
    precision: CompiledModel.GpuOptions.Precision = CompiledModel.GpuOptions.Precision.FP16,
    accelerator: Accelerator = Accelerator.GPU,
) : InferenceModel {
    /** [sig] = tensor name in the TFLite signature (LiteRT runs by signature, not by interpreter order). */
    private data class Io(val onnx: String, val sig: String, val nhwc: Boolean, val shape: LongArray)

    override val name = tflite.nameWithoutExtension
    override val backend = "LiteRT/${accelerator.name} ${precision.name.lowercase()}"

    private val inputs: List<Io>
    private val outputs: List<Io>
    private val model: CompiledModel
    private val signature: String
    private val inBuffers: Map<String, TensorBuffer>
    private val outBuffers: Map<String, TensorBuffer>

    init {
        val meta = JSONObject(File(tflite.path + ".json").readText())
        fun parse(key: String) = meta.getJSONArray(key).let { a ->
            (0 until a.length()).map { i ->
                val o = a.getJSONObject(i)
                val s = o.getJSONArray("shape")
                Io(o.getString("onnx"), o.getString("tflite"), o.getBoolean("nhwc"), LongArray(s.length()) { s.getLong(it) })
            }
        }
        require(meta.optInt("version") >= 2) { "outdated TFLite mapping – re-run tools/convert_tflite.py" }
        signature = meta.getString("signature")
        inputs = parse("inputs")
        outputs = parse("outputs")
        val options = CompiledModel.Options(accelerator).apply {
            // infiniteFloatCapping: fp16 overflow (Mali InstanceNorm / large sums) is capped instead of becoming Inf/NaN.
            // The serialized program cache is keyed per precision so variants never share a compiled program.
            if (accelerator == Accelerator.GPU) gpuOptions = CompiledModel.GpuOptions(
                null, precision != CompiledModel.GpuOptions.Precision.FP32, null, precision, null, null,
                File(context.cacheDir, "litert_gpu").apply { mkdirs() }.absolutePath, "$name-${precision.name}", true, null, null, null, null, null, null,
            )
        }
        model = CompiledModel.create(tflite.absolutePath, options, env(context))
        inBuffers = inputs.associate { it.sig to model.createInputBuffer(it.sig, signature) }
        outBuffers = outputs.associate { it.sig to model.createOutputBuffer(it.sig, signature) }
    }

    override val inputNames: List<String> = inputs.map { it.onnx }
    override val outputNames: List<String> = outputs.map { it.onnx }
    override fun inputShape(name: String): LongArray? = inputs.firstOrNull { it.onnx == name }?.shape

    @Synchronized
    override fun run(inputs: Map<String, Tensor>): Map<String, Tensor> {
        for (io in this.inputs) {
            val t = inputs[io.onnx] ?: error("missing input ${io.onnx}")
            inBuffers.getValue(io.sig).writeFloat(if (io.nhwc) nchwToNhwc(t.data, io.shape) else t.data)
        }
        model.run(inBuffers, outBuffers, signature)
        val out = LinkedHashMap<String, Tensor>()
        for (io in outputs) {
            val d = outBuffers.getValue(io.sig).readFloat()
            out[io.onnx] = Tensor(if (io.nhwc) nhwcToNchw(d, io.shape) else d, io.shape)
        }
        return out
    }

    override fun close() {
        inBuffers.values.forEach { runCatching { it.close() } }
        outBuffers.values.forEach { runCatching { it.close() } }
        model.close()
    }

    companion object {
        @Volatile private var environment: Environment? = null
        @Synchronized fun env(context: Context): Environment = environment ?: Environment.create(context.applicationContext).also { environment = it }

        /** True when the TFLite twin + mapping exist for this model. */
        fun available(dir: File, id: String) = File(dir, "$id.tflite").exists() &&
            runCatching { JSONObject(File(dir, "$id.tflite.json").readText()).optInt("version") >= 2 }.getOrDefault(false)

        private fun nchwToNhwc(d: FloatArray, s: LongArray): FloatArray {
            val c = s[1].toInt(); val h = s[2].toInt(); val w = s[3].toInt()
            val o = FloatArray(d.size)
            for (ch in 0 until c) for (y in 0 until h) for (x in 0 until w) o[(y * w + x) * c + ch] = d[(ch * h + y) * w + x]
            return o
        }

        private fun nhwcToNchw(d: FloatArray, s: LongArray): FloatArray {
            val c = s[1].toInt(); val h = s[2].toInt(); val w = s[3].toInt()
            val o = FloatArray(d.size)
            for (ch in 0 until c) for (y in 0 until h) for (x in 0 until w) o[(ch * h + y) * w + x] = d[(y * w + x) * c + ch]
            return o
        }
    }
}
