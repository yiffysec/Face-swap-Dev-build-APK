package com.faceswap.core

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RemoteTest {

    @Test
    fun `tensor wire round-trips names, shapes and data`() {
        val tensors = linkedMapOf(
            "target" to Tensor(FloatArray(3 * 4 * 4) { it * 0.5f }, 1, 3, 4, 4),
            "source" to Tensor(FloatArray(512) { -it.toFloat() }, 1, 512),
            "scalar" to Tensor.scalar(0.42f),
        )
        val bytes = ByteArrayOutputStream().apply { TensorWire.writeMap(this, tensors) }.toByteArray()
        val back = TensorWire.readMap(ByteArrayInputStream(bytes))

        assertEquals(tensors.keys, back.keys)
        for (name in tensors.keys) {
            assertEquals(tensors[name]!!.shape.toList(), back[name]!!.shape.toList(), "shape of $name")
            assertEquals(tensors[name]!!.data.toList(), back[name]!!.data.toList(), "data of $name")
        }
    }

    @Test
    fun `tensor wire handles an empty map`() {
        val bytes = ByteArrayOutputStream().apply { TensorWire.writeMap(this, emptyMap()) }.toByteArray()
        assertTrue(TensorWire.readMap(ByteArrayInputStream(bytes)).isEmpty())
    }

    @Test
    fun `tensor wire rejects a stream with a bad magic number`() {
        val garbage = ByteArrayOutputStream().apply {
            java.io.DataOutputStream(this).apply { writeShort(1); writeInt(0xDEADBEEF.toInt()) }
        }.toByteArray()
        assertFailsWith<IllegalStateException> { TensorWire.readMap(ByteArrayInputStream(garbage)) }
    }

    @Test
    fun `RemoteInferenceModel forwards run() to the transport unchanged`() {
        var seenModel: String? = null
        var seenInputs: Map<String, Tensor>? = null
        val transport = RemoteTransport { modelId, inputs ->
            seenModel = modelId; seenInputs = inputs
            mapOf("output" to Tensor(floatArrayOf(1f, 2f, 3f), 3))
        }
        val model = RemoteInferenceModel(
            modelId = "hyperswap_1a_256",
            inputNames = listOf("target", "source"),
            outputNames = listOf("output", "mask"),
            transport = transport,
        )
        val inputs = mapOf("target" to Tensor(FloatArray(16), 1, 1, 4, 4), "source" to Tensor(FloatArray(512), 1, 512))
        val out = model.run(inputs)

        assertEquals("hyperswap_1a_256", seenModel)
        assertEquals(inputs, seenInputs)
        assertEquals(listOf(1f, 2f, 3f), out["output"]!!.data.toList())
        assertEquals("hyperswap_1a_256", model.name)
        assertEquals("Remote", model.backend)
    }

    @Test
    fun `RemoteInferenceModel surfaces transport failures to the caller`() {
        val model = RemoteInferenceModel("missing_model", emptyList(), emptyList(), transport = RemoteTransport { _, _ -> error("HTTP 404") })
        assertFailsWith<IllegalStateException> { model.run(emptyMap()) }
    }
}
