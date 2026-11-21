package com.faceswap.studio.data

import android.content.Context
import android.graphics.Bitmap
import com.faceswap.core.SourceFace
import com.faceswap.core.SourceProvenance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/** A saved identity in the face library. */
data class LibraryFace(
    val id: String,
    val label: String,
    val createdAt: Long,
    val provenance: SourceProvenance,
    val embeddings: List<FloatArray>,
    val thumbFile: File,
    val photoFile: File?,
) {
    fun toSource() = SourceFace(id, label, embeddings, provenance)
    val isLive get() = provenance == SourceProvenance.LIVE_CAMERA
}

/**
 * Persistent face library in app-private storage:
 *   files/faces/index.json     – ids, labels, embeddings
 *   files/faces/<id>/thumb.png – display thumbnail
 *   files/faces/<id>/live.png  – (live captures) full photo, PNG tEXt metadata: FS:Live=true, FS:CapturedAt, ...
 * Live status is read from the PNG metadata.
 */
class FaceLibrary(context: Context) {
    private val root = File(context.filesDir, "faces").apply { mkdirs() }
    private val index = File(root, "index.json")
    private val _faces = MutableStateFlow<List<LibraryFace>>(emptyList())
    val faces: StateFlow<List<LibraryFace>> = _faces.asStateFlow()

    suspend fun load() = withContext(Dispatchers.IO) { _faces.value = readIndex() }

    private fun readIndex(): List<LibraryFace> {
        if (!index.exists()) return emptyList()
        val arr = runCatching { JSONArray(index.readText()) }.getOrElse { return emptyList() }
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val id = o.getString("id")
            val dir = File(root, id)
            val live = File(dir, "live.png")
            val isLive = live.exists() && PngText.read(live)["FS:Live"] == "true"
            LibraryFace(
                id, o.getString("label"), o.getLong("createdAt"),
                if (isLive) SourceProvenance.LIVE_CAMERA else SourceProvenance.GALLERY,
                o.getJSONArray("embeddings").let { a -> (0 until a.length()).map { decodeEmb(a.getString(it)) } },
                File(dir, "thumb.png"), live.takeIf { it.exists() },
            )
        }
    }

    private fun writeIndex(list: List<LibraryFace>) {
        val arr = JSONArray()
        list.forEach { f ->
            arr.put(JSONObject().apply {
                put("id", f.id); put("label", f.label); put("createdAt", f.createdAt)
                put("embeddings", JSONArray(f.embeddings.map { encodeEmb(it) }))
            })
        }
        val tmp = File(root, "index.json.tmp"); tmp.writeText(arr.toString()); tmp.renameTo(index)
    }

    private fun saveThumb(dir: File, thumb: Bitmap) =
        File(dir, "thumb.png").also { f -> f.outputStream().use { thumb.compress(Bitmap.CompressFormat.PNG, 100, it) } }

    suspend fun addGallery(label: String, embeddings: List<FloatArray>, thumb: Bitmap): LibraryFace = withContext(Dispatchers.IO) {
        val id = UUID.randomUUID().toString()
        val dir = File(root, id).apply { mkdirs() }
        val face = LibraryFace(id, label, System.currentTimeMillis(), SourceProvenance.GALLERY, embeddings, saveThumb(dir, thumb), null)
        commit(_faces.value + face); face
    }

    /** Stores an in-app camera capture as PNG with live metadata. */
    suspend fun addLive(label: String, embeddings: List<FloatArray>, photo: Bitmap, thumb: Bitmap): LibraryFace = withContext(Dispatchers.IO) {
        val id = UUID.randomUUID().toString()
        val dir = File(root, id).apply { mkdirs() }
        val now = System.currentTimeMillis()
        val live = File(dir, "live.png")
        PngText.write(photo, live, mapOf(
            "FS:Live" to "true",
            "FS:CaptureSource" to "in-app-camera",
            "FS:CapturedAt" to java.time.Instant.ofEpochMilli(now).toString(),
            "FS:FaceId" to id,
            "FS:Label" to label,
            "FS:App" to "FaceSwapStudio",
        ))
        val face = LibraryFace(id, label, now, SourceProvenance.LIVE_CAMERA, embeddings, saveThumb(dir, thumb), live)
        commit(_faces.value + face); face
    }

    suspend fun rename(id: String, label: String) = withContext(Dispatchers.IO) {
        commit(_faces.value.map { if (it.id == id) it.copy(label = label) else it })
    }

    suspend fun delete(id: String) = withContext(Dispatchers.IO) {
        File(root, id).deleteRecursively()
        commit(_faces.value.filterNot { it.id == id })
    }

    private fun commit(list: List<LibraryFace>) {
        writeIndex(list)
        _faces.value = readIndex()
    }

    companion object {
        fun encodeEmb(e: FloatArray): String {
            val bb = ByteBuffer.allocate(e.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            e.forEach { bb.putFloat(it) }
            return android.util.Base64.encodeToString(bb.array(), android.util.Base64.NO_WRAP)
        }

        fun decodeEmb(s: String): FloatArray {
            val bb = ByteBuffer.wrap(android.util.Base64.decode(s, android.util.Base64.NO_WRAP)).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(bb.remaining() / 4) { bb.float }
        }
    }
}
