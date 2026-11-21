package com.faceswap.core

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal streaming protobuf reader that pulls one initializer tensor out of an .onnx file without loading
 * the whole model into memory. Used to extract inswapper's 512x512 "emap" matrix, which ONNX Runtime prunes
 * because no graph node consumes it.
 *
 * ModelProto.graph = field 7, GraphProto.initializer = field 5, TensorProto: dims=1, data_type=2, float_data=4,
 * name=8, raw_data=9.
 */
object OnnxInitializerReader {

    fun readFloatInitializer(file: File, name: String? = null, expectedSize: Int = 512 * 512): FloatArray {
        RandomAccessFile(file, "r").use { raf ->
            val reader = Reader(raf)
            var result: FloatArray? = null
            // top-level ModelProto fields
            while (reader.pos < raf.length()) {
                val tag = reader.varint()
                val field = (tag ushr 3).toInt()
                val wire = (tag and 7).toInt()
                if (field == 7 && wire == 2) {
                    val len = reader.varint()
                    val end = reader.pos + len
                    while (reader.pos < end) {
                        val gTag = reader.varint()
                        val gField = (gTag ushr 3).toInt()
                        val gWire = (gTag and 7).toInt()
                        if (gField == 5 && gWire == 2) {
                            val tLen = reader.varint()
                            val tEnd = reader.pos + tLen
                            val parsed = parseTensor(reader, tEnd, name, expectedSize)
                            if (parsed != null) result = parsed // keep last match (emap is the last initializer)
                            reader.seek(tEnd)
                        } else reader.skip(gWire)
                    }
                } else reader.skip(wire)
            }
            return result ?: error("initializer ${name ?: "(size $expectedSize)"} not found in ${file.name}")
        }
    }

    private fun parseTensor(r: Reader, end: Long, wantName: String?, expected: Int): FloatArray? {
        var tName = ""
        var dataType = 0
        var rawPos = -1L
        var rawLen = 0L
        val dims = ArrayList<Long>()
        var floatPos = -1L
        var floatLen = 0L
        while (r.pos < end) {
            val tag = r.varint()
            val f = (tag ushr 3).toInt()
            val w = (tag and 7).toInt()
            when {
                f == 1 && w == 0 -> dims += r.varint()
                f == 1 && w == 2 -> { val l = r.varint(); val e = r.pos + l; while (r.pos < e) dims += r.varint() }
                f == 2 && w == 0 -> dataType = r.varint().toInt()
                f == 8 && w == 2 -> { val l = r.varint().toInt(); tName = String(r.bytes(l)) }
                f == 9 && w == 2 -> { rawLen = r.varint(); rawPos = r.pos; r.seek(r.pos + rawLen) }
                f == 4 && w == 2 -> { floatLen = r.varint(); floatPos = r.pos; r.seek(r.pos + floatLen) }
                else -> r.skip(w)
            }
        }
        val count = dims.fold(1L) { a, b -> a * b }
        if (wantName != null && tName != wantName) return null
        if (wantName == null && count != expected.toLong()) return null
        val (pos, len) = if (rawPos >= 0) rawPos to rawLen else floatPos to floatLen
        if (pos < 0) return null
        r.seek(pos)
        val bytes = ByteBuffer.wrap(r.bytes(len.toInt())).order(ByteOrder.LITTLE_ENDIAN)
        return when (dataType) {
            1 -> FloatArray(count.toInt()) { bytes.float }
            10 -> FloatArray(count.toInt()) { Half.toFloat(bytes.short) }
            else -> error("unsupported initializer data type $dataType")
        }
    }

    private class Reader(private val raf: RandomAccessFile) {
        private val buf = ByteArray(1 shl 16)
        private var bufStart = 0L
        private var bufLen = 0
        var pos = 0L
            private set

        fun seek(p: Long) { pos = p }

        private fun byte(): Int {
            if (pos < bufStart || pos >= bufStart + bufLen) {
                raf.seek(pos); bufStart = pos; bufLen = raf.read(buf).coerceAtLeast(0)
                if (bufLen == 0) error("unexpected EOF")
            }
            return buf[(pos++ - bufStart).toInt()].toInt() and 0xFF
        }

        fun varint(): Long {
            var shift = 0; var result = 0L
            while (true) {
                val b = byte()
                result = result or ((b and 0x7F).toLong() shl shift)
                if (b and 0x80 == 0) return result
                shift += 7
            }
        }

        fun bytes(n: Int): ByteArray {
            val out = ByteArray(n)
            raf.seek(pos); raf.readFully(out); pos += n
            bufLen = 0
            return out
        }

        fun skip(wire: Int) {
            when (wire) {
                0 -> varint()
                1 -> pos += 8
                2 -> { val l = varint(); pos += l }
                5 -> pos += 4
                else -> error("bad wire type $wire")
            }
        }
    }

    private object Half {
        fun toFloat(h: Short): Float {
            val bits = h.toInt() and 0xFFFF
            val sign = (bits ushr 15) and 1
            val exp = (bits ushr 10) and 0x1F
            val mant = bits and 0x3FF
            val v = when (exp) {
                0 -> mant / 1024f * Math.pow(2.0, -14.0).toFloat()
                31 -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                else -> (1 + mant / 1024f) * Math.pow(2.0, (exp - 15).toDouble()).toFloat()
            }
            return if (sign == 1) -v else v
        }
    }
}
