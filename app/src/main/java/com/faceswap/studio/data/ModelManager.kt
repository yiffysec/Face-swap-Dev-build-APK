package com.faceswap.studio.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.CRC32
import kotlin.coroutines.coroutineContext

enum class ModelRole { DETECTOR, RECOGNIZER, SWAPPER, ANALYSER, ENHANCER }

data class ModelSpec(
    val id: String,
    val title: String,
    val description: String,
    val role: ModelRole,
    val sizeBytes: Long,
    val required: Boolean,
    val license: String,
) {
    val fileName get() = "$id.onnx"
    val url get() = "$BASE/$id.onnx"
    val hashUrl get() = "$BASE/$id.hash"

    companion object {
        /** Open model mirror maintained by the FaceFusion project on Hugging Face. */
        const val BASE = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"
    }
}

object ModelCatalog {
    val all = listOf(
        ModelSpec("scrfd_2.5g", "SCRFD 2.5G", "Fast, accurate face + landmark detector", ModelRole.DETECTOR, 3_295_067, true, "InsightFace – non-commercial research"),
        ModelSpec("arcface_w600k_r50", "ArcFace R50", "Identity embeddings (matching, library, mapping)", ModelRole.RECOGNIZER, 174_388_474, true, "InsightFace – non-commercial research"),
        ModelSpec("inswapper_128_fp16", "InSwapper 128 (fp16)", "The face swapping model", ModelRole.SWAPPER, 277_680_829, true, "InsightFace – non-commercial research"),
        ModelSpec("open_nsfw", "Content analyser", "Target content classification for the identity policy", ModelRole.ANALYSER, 23_598_282, true, "BSD-2 (Yahoo open_nsfw)"),
        ModelSpec("gfpgan_1.4", "GFPGAN 1.4", "Optional face restoration – sharper, more detailed faces", ModelRole.ENHANCER, 340_299_087, false, "Apache-2.0 (TencentARC)"),
    )
    fun byRole(role: ModelRole) = all.first { it.role == role }
}

sealed class ModelState {
    object Missing : ModelState()
    data class Downloading(val bytes: Long, val total: Long) : ModelState() {
        val progress get() = if (total > 0) bytes.toFloat() / total else 0f
    }
    object Verifying : ModelState()
    /** Copying a model bundled inside the APK (or imported from storage) into app storage. */
    object Installing : ModelState()
    object Ready : ModelState()
    data class Failed(val message: String) : ModelState()
}

class ModelManager(private val context: Context) {
    val dir = File(context.filesDir, "models").apply { mkdirs() }
    val emapCache = File(dir, "inswapper_emap.bin")

    private val _states = MutableStateFlow(ModelCatalog.all.associate { it.id to initial(it) })
    val states: StateFlow<Map<String, ModelState>> = _states.asStateFlow()

    private fun initial(spec: ModelSpec) = if (file(spec).exists()) ModelState.Ready else ModelState.Missing

    fun file(spec: ModelSpec) = File(dir, spec.fileName)
    fun file(role: ModelRole) = file(ModelCatalog.byRole(role))

    val requiredReady get() = ModelCatalog.all.filter { it.required }.all { file(it).exists() }
    val missingRequiredBytes get() = ModelCatalog.all.filter { it.required && !file(it).exists() }.sumOf { it.sizeBytes }

    private fun set(id: String, s: ModelState) = _states.update { it + (id to s) }

    fun delete(spec: ModelSpec) {
        file(spec).delete(); File(dir, spec.fileName + ".part").delete()
        if (spec.role == ModelRole.SWAPPER) emapCache.delete()
        set(spec.id, ModelState.Missing)
    }

    /** Resumable download with CRC32 verification against the published .hash file. */
    suspend fun download(spec: ModelSpec) = withContext(Dispatchers.IO) {
        val target = file(spec)
        val part = File(dir, spec.fileName + ".part")
        try {
            val expectedHash = runCatching { fetchText(spec.hashUrl).trim().lowercase() }.getOrNull()
            var downloaded = if (part.exists()) part.length() else 0L
            var total = spec.sizeBytes
            var url = URL(spec.url)
            var conn: HttpURLConnection
            var redirects = 0
            while (true) {
                conn = (url.openConnection() as HttpURLConnection).apply {
                    instanceFollowRedirects = false
                    connectTimeout = 20_000; readTimeout = 60_000
                    if (downloaded > 0) setRequestProperty("Range", "bytes=$downloaded-")
                }
                val code = conn.responseCode
                if (code in 300..399 && redirects++ < 8) {
                    url = URL(url, conn.getHeaderField("Location")); conn.disconnect(); continue
                }
                if (code == 416) { downloaded = part.length(); break } // already complete
                if (code != 200 && code != 206) error("HTTP $code")
                if (code == 200) downloaded = 0
                val len = conn.getHeaderFieldLong("Content-Length", -1)
                if (len > 0) total = downloaded + len
                RandomAccessFile(part, "rw").use { raf ->
                    raf.setLength(downloaded); raf.seek(downloaded)
                    conn.inputStream.use { input ->
                        val buf = ByteArray(256 * 1024)
                        var lastEmit = 0L
                        while (coroutineContext.isActive) {
                            val n = input.read(buf)
                            if (n < 0) break
                            raf.write(buf, 0, n)
                            downloaded += n
                            if (downloaded - lastEmit > 1_000_000) { set(spec.id, ModelState.Downloading(downloaded, total)); lastEmit = downloaded }
                        }
                    }
                }
                conn.disconnect()
                break
            }
            if (!coroutineContext.isActive) { set(spec.id, ModelState.Missing); return@withContext }
            set(spec.id, ModelState.Verifying)
            if (expectedHash != null && expectedHash.length == 8) {
                val actual = crc32Hex(part)
                if (actual != expectedHash) { part.delete(); error("Checksum mismatch – please retry") }
            }
            if (!part.renameTo(target)) error("Could not finalise file")
            set(spec.id, ModelState.Ready)
        } catch (t: Throwable) {
            set(spec.id, ModelState.Failed(t.message ?: t.javaClass.simpleName))
        }
    }

    /** True if this APK ships with models in assets/models (offline builds). */
    val hasBundledModels: Boolean by lazy { (context.assets.list("models") ?: emptyArray()).any { it.endsWith(".onnx") } }

    /** Copies models bundled in the APK into app storage (first launch / after update). Idempotent. */
    suspend fun installBundled() = withContext(Dispatchers.IO) {
        val bundled = context.assets.list("models")?.toSet() ?: return@withContext
        for (spec in ModelCatalog.all) {
            if (spec.fileName !in bundled || file(spec).exists()) continue
            set(spec.id, ModelState.Installing)
            try {
                val part = File(dir, spec.fileName + ".part")
                context.assets.open("models/${spec.fileName}").use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
                check(part.renameTo(file(spec)))
                set(spec.id, ModelState.Ready)
            } catch (t: Throwable) {
                set(spec.id, ModelState.Failed("Install failed: ${t.message}"))
            }
        }
    }

    /** Imports .onnx files picked from device storage (USB-copied), matched by file name. Returns names imported. */
    suspend fun importFromUris(uris: List<android.net.Uri>): List<String> = withContext(Dispatchers.IO) {
        val done = ArrayList<String>()
        for (uri in uris) {
            val name = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            } ?: continue
            val spec = ModelCatalog.all.firstOrNull { it.fileName.equals(name, ignoreCase = true) } ?: continue
            set(spec.id, ModelState.Installing)
            try {
                val part = File(dir, spec.fileName + ".part")
                context.contentResolver.openInputStream(uri)!!.use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
                file(spec).delete()
                check(part.renameTo(file(spec)))
                if (spec.role == ModelRole.SWAPPER) emapCache.delete()
                set(spec.id, ModelState.Ready); done += spec.fileName
            } catch (t: Throwable) {
                set(spec.id, ModelState.Failed("Import failed: ${t.message}"))
            }
        }
        done
    }

    private fun fetchText(u: String): String {
        var url = URL(u)
        repeat(8) {
            val c = (url.openConnection() as HttpURLConnection).apply { instanceFollowRedirects = false; connectTimeout = 15_000 }
            if (c.responseCode in 300..399) { url = URL(url, c.getHeaderField("Location")); c.disconnect(); return@repeat }
            return c.inputStream.bufferedReader().use { it.readText() }
        }
        error("too many redirects")
    }

    private fun crc32Hex(f: File): String {
        val crc = CRC32()
        f.inputStream().buffered(1 shl 20).use { s ->
            val buf = ByteArray(1 shl 20)
            while (true) { val n = s.read(buf); if (n < 0) break; crc.update(buf, 0, n) }
        }
        return "%08x".format(crc.value)
    }
}
