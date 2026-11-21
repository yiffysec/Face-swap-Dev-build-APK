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
class LiteRtModel(context: Context, tflite: File, accelerator: Accelerator = Accelerator.GPU) : InferenceModel {
    private data class Io(val onnx: String, val index: Int, val nhwc: Boolean, val shape: LongArray)

    override val name = tflite.nameWithoutExtension
    override val backend = "LiteRT/${accelerator.name}"

    private val inputs: List<Io>
    private val outputs: List<Io>
    private val model: CompiledModel
    private val inBuffers: List<TensorBuffer>
    private val outBuffers: List<TensorBuffer>

    init {
        val meta = JSONObject(File(tflite.path + ".json").readText())
        fun parse(key: String) = meta.getJSONArray(key).let { a ->
            (0 until a.length()).map { i ->
                val o = a.getJSONObject(i)
                val s = o.getJSONArray("shape")
                Io(o.getString("onnx"), o.getInt("index"), o.getBoolean("nhwc"), LongArray(s.length()) { s.getLong(it) })
            }
        }
        inputs = parse("inputs")
        outputs = parse("outputs")
        val options = CompiledModel.Options(accelerator).apply {
            if (accelerator == Accelerator.GPU) gpuOptions = CompiledModel.GpuOptions(
                null, null, null, CompiledModel.GpuOptions.Precision.FP16, null, null,
                File(context.cacheDir, "litert_gpu").apply { mkdirs() }.absolutePath, name, true, null, null, null, null, null, null,
            )
        }
        model = CompiledModel.create(tflite.absolutePath, options, env(context))
        inBuffers = model.createInputBuffers(0)
        outBuffers = model.createOutputBuffers(0)
    }

    override val inputNames: List<String> = inputs.map { it.onnx }
    override val outputNames: List<String> = outputs.map { it.onnx }
    override fun inputShape(name: String): LongArray? = inputs.firstOrNull { it.onnx == name }?.shape

    @Synchronized
    override fun run(inputs: Map<String, Tensor>): Map<String, Tensor> {
        for (io in this.inputs) {
            val t = inputs[io.onnx] ?: error("missing input ${io.onnx}")
            inBuffers[io.index].writeFloat(if (io.nhwc) nchwToNhwc(t.data, io.shape) else t.data)
        }
        model.run(inBuffers, outBuffers, 0)
        val out = LinkedHashMap<String, Tensor>()
        for (io in outputs) {
            val d = outBuffers[io.index].readFloat()
            out[io.onnx] = Tensor(if (io.nhwc) nhwcToNchw(d, io.shape) else d, io.shape)
        }
        return out
    }

    override fun close() {
        inBuffers.forEach { runCatching { it.close() } }
        outBuffers.forEach { runCatching { it.close() } }
        model.close()
    }

    companion object {
        @Volatile private var environment: Environment? = null
        @Synchronized fun env(context: Context): Environment = environment ?: Environment.create(context.applicationContext).also { environment = it }

        /** True when the TFLite twin + mapping exist for this model. */
        fun available(dir: File, id: String) = File(dir, "$id.tflite").exists() && File(dir, "$id.tflite.json").exists()

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
