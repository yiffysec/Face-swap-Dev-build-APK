package com.faceswap.core

/**
 * Forwards one model's inference call to a remote server. Implemented by the app layer (HTTP/TLS to the custom
 * backend); blocking, called from a background dispatcher by [FaceSwapEngine] the same way a local [InferenceModel]
 * would be. Throwing propagates as a normal pipeline error (network failure, server error, unknown model id, …).
 */
fun interface RemoteTransport {
    fun infer(modelId: String, inputs: Map<String, Tensor>): Map<String, Tensor>
}

/**
 * An [InferenceModel] with no local weights at all: every [run] is forwarded to [transport]. Used for the "remote
 * server" processing mode, where the phone never downloads or hardcodes any model – only a model *id*, resolved by
 * whatever the connected server happens to host. Input/output names come from the server's own manifest (fetched
 * once at connect time) since there is no local ONNX file to introspect.
 *
 * This is the only piece of client code that knows processing happens over the network: everything above it in the
 * pipeline (detection, swapping, masking, pixel boost, enhancement – see [FaceSwapEngine] and the model wrapper
 * classes in Models.kt) is backend-agnostic and runs unmodified whether a model is local or remote.
 */
class RemoteInferenceModel(
    val modelId: String,
    override val inputNames: List<String>,
    override val outputNames: List<String>,
    private val inputShapes: Map<String, LongArray> = emptyMap(),
    private val transport: RemoteTransport,
) : InferenceModel {
    override val name: String = modelId
    override val backend: String = "Remote"

    override fun inputShape(name: String): LongArray? = inputShapes[name]

    override fun run(inputs: Map<String, Tensor>): Map<String, Tensor> {
        val result = transport.infer(modelId, inputs)
        check(result.isNotEmpty()) { "Server returned no outputs for $modelId" }
        return result
    }

    /** Nothing local to release – the server owns the weights and the session lifetime. */
    override fun close() = Unit
}
