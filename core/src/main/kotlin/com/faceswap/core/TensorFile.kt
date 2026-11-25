package com.faceswap.core

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/**
 * Tiny binary container for named float tensors ("FSIN" v1, big-endian), shared with the Python tooling:
 *   magic "FSIN", int version, int count, then per tensor: UTF name, int rank, long dims…, int n, float data…
 *
 * Used for (1) calibration inputs captured from the real pipeline (quantization in CI) and (2) the `<id>.sample.bin`
 * input the on-device tuner uses, so backends are compared on real face data instead of random noise.
 */
object TensorFile {
    private const val MAGIC = 0x4653494E // "FSIN"

    fun write(file: File, tensors: Map<String, Tensor>) {
        file.parentFile?.mkdirs()
        DataOutputStream(file.outputStream().buffered(1 shl 16)).use { o ->
            o.writeInt(MAGIC); o.writeInt(1); o.writeInt(tensors.size)
            for ((name, t) in tensors) {
                o.writeUTF(name)
                o.writeInt(t.shape.size); t.shape.forEach { o.writeLong(it) }
                o.writeInt(t.data.size); t.data.forEach { o.writeFloat(it) }
            }
        }
    }

    fun read(file: File): Map<String, Tensor> = DataInputStream(file.inputStream().buffered(1 shl 16)).use { i ->
        require(i.readInt() == MAGIC) { "not a tensor file: $file" }
        i.readInt()
        val n = i.readInt()
        val out = LinkedHashMap<String, Tensor>()
        repeat(n) {
            val name = i.readUTF()
            val shape = LongArray(i.readInt()) { i.readLong() }
            val data = FloatArray(i.readInt()) { i.readFloat() }
            out[name] = Tensor(data, shape)
        }
        out
    }

    fun bytes(tensors: Map<String, Tensor>) = tensors.values.sumOf { it.data.size * 4L }
}
