package com.faceswap.core

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream

/**
 * Minimal binary wire format for sending/receiving named float tensors to the remote inference server.
 * Big-endian (`Data{Output,Input}Stream` default), float32 payloads. Mirrors the Python server's
 * `app/tensor_wire.py` byte-for-byte – if you change one, change the other and bump [VERSION].
 *
 * Wire shape:
 *   message   := u16(count) frame*count
 *   frame     := u32(magic) u8(version) u16(name_len) utf8(name) u8(rank) i64(dim)*rank u64(elem_count) f32(elem)*elem_count
 *
 * This carries only the tensor data an [InferenceModel] needs – no error channel. A failed request is reported at
 * the transport level instead (e.g. a non-2xx HTTP status with a JSON `{"error": ...}` body); see
 * `RemoteClient.infer` on the app side and `app/main.py` on the server side.
 */
object TensorWire {
    private const val MAGIC = 0x46535731 // ASCII "FSW1"
    private const val VERSION = 1

    fun writeMap(out: OutputStream, tensors: Map<String, Tensor>) {
        val d = if (out is DataOutputStream) out else DataOutputStream(out)
        d.writeShort(tensors.size)
        for ((name, t) in tensors) writeTensor(d, name, t)
        d.flush()
    }

    private fun writeTensor(d: DataOutputStream, name: String, t: Tensor) {
        d.writeInt(MAGIC)
        d.writeByte(VERSION)
        val nameBytes = name.toByteArray(Charsets.UTF_8)
        require(nameBytes.size <= 0xFFFF) { "tensor name too long: $name" }
        d.writeShort(nameBytes.size)
        d.write(nameBytes)
        require(t.shape.size <= 0xFF) { "tensor rank too high for $name" }
        d.writeByte(t.shape.size)
        for (dim in t.shape) d.writeLong(dim)
        d.writeLong(t.data.size.toLong())
        for (v in t.data) d.writeFloat(v)
    }

    fun readMap(input: InputStream): Map<String, Tensor> {
        val d = if (input is DataInputStream) input else DataInputStream(input)
        val count = d.readUnsignedShort()
        val out = LinkedHashMap<String, Tensor>(count.coerceAtLeast(1) * 2)
        repeat(count) {
            val (name, tensor) = readTensor(d)
            out[name] = tensor
        }
        return out
    }

    private fun readTensor(d: DataInputStream): Pair<String, Tensor> {
        val magic = try { d.readInt() } catch (e: EOFException) { throw EOFException("truncated tensor stream (expected frame magic)") }
        check(magic == MAGIC) { "bad tensor frame magic: 0x${magic.toString(16)}" }
        val version = d.readUnsignedByte()
        check(version == VERSION) { "unsupported tensor frame version $version (expected $VERSION)" }
        val nameLen = d.readUnsignedShort()
        val nameBytes = ByteArray(nameLen)
        d.readFully(nameBytes)
        val name = String(nameBytes, Charsets.UTF_8)
        val rank = d.readUnsignedByte()
        val shape = LongArray(rank) { d.readLong() }
        val count = d.readLong()
        require(count in 0..Int.MAX_VALUE.toLong()) { "tensor '$name' too large ($count elements)" }
        val data = FloatArray(count.toInt()) { d.readFloat() }
        return name to Tensor(data, shape)
    }
}
