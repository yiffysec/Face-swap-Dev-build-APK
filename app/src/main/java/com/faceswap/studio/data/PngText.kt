package com.faceswap.studio.data

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.File
import java.util.zip.CRC32

/** Writes/reads PNG tEXt metadata chunks (readable by exiftool, ImageMagick `identify -verbose`, etc.). */
object PngText {
    private val SIGNATURE = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)

    private class Chunk(val type: String, val data: ByteArray)

    private fun chunks(bytes: ByteArray): List<Chunk> {
        require(bytes.size > 8 && bytes.copyOfRange(0, 8).contentEquals(SIGNATURE)) { "not a PNG" }
        val input = DataInputStream(bytes.inputStream(8, bytes.size - 8))
        val out = ArrayList<Chunk>()
        while (input.available() > 0) {
            val len = input.readInt()
            val type = ByteArray(4).also { input.readFully(it) }
            val data = ByteArray(len).also { input.readFully(it) }
            input.readInt()
            out += Chunk(String(type, Charsets.ISO_8859_1), data)
            if (out.last().type == "IEND") break
        }
        return out
    }

    private fun intBytes(v: Int) = byteArrayOf((v ushr 24).toByte(), (v ushr 16).toByte(), (v ushr 8).toByte(), v.toByte())

    private fun writeChunk(out: ByteArrayOutputStream, type: String, data: ByteArray) {
        val t = type.toByteArray(Charsets.ISO_8859_1)
        out.write(intBytes(data.size)); out.write(t); out.write(data)
        out.write(intBytes(CRC32().apply { update(t); update(data) }.value.toInt()))
    }

    fun write(bmp: Bitmap, file: File, fields: Map<String, String>) {
        val raw = ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
        val out = ByteArrayOutputStream()
        out.write(SIGNATURE)
        for (c in chunks(raw)) {
            if (c.type == "IEND") fields.forEach { (k, v) ->
                writeChunk(out, "tEXt", k.toByteArray(Charsets.ISO_8859_1) + 0 + v.toByteArray(Charsets.ISO_8859_1))
            }
            writeChunk(out, c.type, c.data)
        }
        file.parentFile?.mkdirs()
        file.writeBytes(out.toByteArray())
    }

    fun read(file: File): Map<String, String> {
        val list = runCatching { chunks(file.readBytes()) }.getOrElse { return emptyMap() }
        val fields = LinkedHashMap<String, String>()
        for (c in list.filter { it.type == "tEXt" }) {
            val z = c.data.indexOf(0)
            if (z <= 0) continue
            fields[String(c.data, 0, z, Charsets.ISO_8859_1)] = String(c.data, z + 1, c.data.size - z - 1, Charsets.ISO_8859_1)
        }
        return fields
    }
}
