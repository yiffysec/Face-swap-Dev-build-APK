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

enum class ModelGroup(val title: String) {
    REQUIRED("Required"), RECOMMENDED("Recommended"), SWAPPERS("Extra swappers"), ENHANCERS("Face enhancers"),
    STUDIO("Studio tools"), LIVE("Live preview"),
}

data class ModelSpec(
    val id: String,
    val title: String,
    val description: String,
    val group: ModelGroup,
    val sizeBytes: Long,
    val license: String,
    /** Download URL of the .onnx file. */
    val url: String,
    /** CRC32 .hash URL (FaceFusion mirrors only). */
    val hashUrl: String? = null,
    /** Bundled into the APK by default (see app/build.gradle.kts "core" set). */
    val core: Boolean = false,
) {
    val fileName get() = "$id.onnx"
    val required get() = group == ModelGroup.REQUIRED

    companion object {
        private fun ff(release: String, id: String) = "https://huggingface.co/facefusion/models-$release/resolve/main/$id"
        fun facefusion(release: String, id: String, title: String, desc: String, group: ModelGroup, size: Long, license: String, core: Boolean = false) =
            ModelSpec(id, title, desc, group, size, license, ff(release, "$id.onnx"), ff(release, "$id.hash"), core)
    }
}

object ModelCatalog {
    private const val MFS = "https://huggingface.co/ZZZHHHHW/mobilefaceswap/resolve/main"
    private val R = ModelGroup.REQUIRED; private val REC = ModelGroup.RECOMMENDED

    val all: List<ModelSpec> = listOf(
        ModelSpec.facefusion("3.4.0", "yunet_2023_mar", "YuNet", "Fast face detector (default)", R, 232_475, "MIT", core = true),
        ModelSpec.facefusion("3.0.0", "scrfd_2.5g", "SCRFD 2.5G", "Alternative face detector", R, 3_295_067, "Non-commercial", core = true),
        ModelSpec.facefusion("3.0.0", "arcface_w600k_r50", "ArcFace R50", "Identity embeddings", R, 174_388_474, "Non-commercial", core = true),
        ModelSpec.facefusion("3.3.0", "hyperswap_1a_256", "HyperSwap 1a", "Default face swapper – 256px, sharp, fast, built-in mask", R, 402_700_000, "ResearchRAIL", core = true),
        ModelSpec.facefusion("3.0.0", "open_nsfw", "Content analyser", "Target content classification for the identity policy", R, 23_598_282, "BSD-2", core = true),

        ModelSpec.facefusion("3.9.0", "hrffa", "HRFFA landmarks", "68-point landmarks – better alignment, area masks, face editor", REC, 35_800_000, "MIT", core = true),
        ModelSpec.facefusion("3.1.0", "bisenet_resnet_18", "BiSeNet face parser", "Region/occlusion masks – keeps hair, hands, glasses", REC, 53_200_000, "MIT", core = true),
        ModelSpec.facefusion("3.0.0", "gpen_bfr_256", "GPEN-BFR 256", "Default face enhancer – 7× faster than GFPGAN", REC, 75_800_000, "Non-commercial", core = true),
        ModelSpec.facefusion("3.5.0", "modnet", "MODNet", "Background removal / blur / replace (portrait matting)", REC, 25_900_000, "Apache-2.0", core = true),
        ModelSpec.facefusion("3.0.0", "span_kendata_x4", "SPAN ×4", "Tiny, fast 4× photo upscaler", REC, 1_700_000, "Non-commercial", core = true),
        ModelSpec.facefusion("3.0.0", "real_esrgan_x2_fp16", "Real-ESRGAN ×2", "High-quality 2× upscaler", REC, 36_100_000, "BSD-3", core = true),

        ModelSpec.facefusion("3.0.0", "inswapper_128_fp16", "InSwapper 128", "Classic swapper – strongest likeness, use with pixel boost", ModelGroup.SWAPPERS, 277_680_829, "Non-commercial"),
        ModelSpec.facefusion("3.3.0", "hyperswap_1b_256", "HyperSwap 1b", "HyperSwap variant", ModelGroup.SWAPPERS, 402_700_000, "ResearchRAIL"),
        ModelSpec.facefusion("3.3.0", "hyperswap_1c_256", "HyperSwap 1c", "HyperSwap variant", ModelGroup.SWAPPERS, 402_700_000, "ResearchRAIL"),

        ModelSpec.facefusion("3.0.0", "gfpgan_1.4", "GFPGAN 1.4", "Classic 512px face restoration", ModelGroup.ENHANCERS, 340_299_087, "Apache-2.0"),
        ModelSpec.facefusion("3.0.0", "gpen_bfr_512", "GPEN-BFR 512", "512px GPEN restoration", ModelGroup.ENHANCERS, 284_300_000, "Non-commercial"),
        ModelSpec.facefusion("3.0.0", "restoreformer_plus_plus", "RestoreFormer++", "Detail-rich restoration", ModelGroup.ENHANCERS, 294_300_000, "Apache-2.0"),
        ModelSpec.facefusion("3.0.0", "codeformer", "CodeFormer", "Restoration with fidelity control", ModelGroup.ENHANCERS, 377_000_000, "S-Lab (non-commercial)"),

        ModelSpec.facefusion("3.6.0", "fran", "FRAN age", "Age modifier (photos)", ModelGroup.STUDIO, 130_800_000, "MIT"),
        ModelSpec.facefusion("3.5.0", "u2netp", "U²-Netp", "Tiny background-removal fallback", ModelGroup.STUDIO, 4_600_000, "Apache-2.0"),
        ModelSpec.facefusion("3.0.0", "real_esrgan_x4_fp16", "Real-ESRGAN ×4", "High-quality 4× upscaler", ModelGroup.STUDIO, 36_100_000, "BSD-3"),
        ModelSpec.facefusion("3.0.0", "live_portrait_feature_extractor", "LivePortrait features", "Expression restorer / face editor", ModelGroup.STUDIO, 3_400_000, "MIT"),
        ModelSpec.facefusion("3.0.0", "live_portrait_motion_extractor", "LivePortrait motion", "Expression restorer / face editor", ModelGroup.STUDIO, 112_600_000, "MIT"),
        ModelSpec.facefusion("3.0.0", "live_portrait_generator", "LivePortrait generator", "Expression restorer / face editor", ModelGroup.STUDIO, 222_300_000, "MIT"),
        ModelSpec.facefusion("3.0.0", "live_portrait_stitcher", "LivePortrait stitcher", "Face editor", ModelGroup.STUDIO, 193_300, "MIT"),
        ModelSpec.facefusion("3.0.0", "live_portrait_eye_retargeter", "LivePortrait eyes", "Face editor – eye open/close", ModelGroup.STUDIO, 581_400, "MIT"),
        ModelSpec.facefusion("3.0.0", "live_portrait_lip_retargeter", "LivePortrait lips", "Face editor – lip open/close", ModelGroup.STUDIO, 151_000, "MIT"),

        ModelSpec("MobileFaceSwapUNetRuntime", "MobileFaceSwap runtime", "Real-time camera preview swapper (364 KB)", ModelGroup.LIVE, 364_176, "Apache-2.0", "$MFS/MobileFaceSwapUNetRuntime.onnx"),
        ModelSpec("BuildFaceSwap", "MobileFaceSwap builder", "Builds per-identity weights (once per face)", ModelGroup.LIVE, 82_512_085, "Apache-2.0", "$MFS/BuildFaceSwap.onnx"),
        ModelSpec("mobilefaceswap_arcface", "MobileFaceSwap ArcFace", "Identity encoder for the preview swapper", ModelGroup.LIVE, 209_111_421, "Apache-2.0", "$MFS/arcface.onnx"),
    )

    fun byId(id: String) = all.firstOrNull { it.id == id }
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
    fun file(id: String) = File(dir, "$id.onnx")
    fun has(id: String) = file(id).exists()
    /** Optional TFLite twin of a model (for the LiteRT GPU backend), produced by CI conversion. */
    fun tflite(id: String) = File(dir, "$id.tflite").takeIf { it.exists() }
    /** Optional calibrated w8a16 QDQ twin (Snapdragon NPU int8), produced by CI quantization. */
    fun int8(id: String) = File(dir, "$id.qdq.onnx").takeIf { it.exists() }

    /** Accelerator twins installed for a model, e.g. "GPU", "NPU int8". */
    fun twins(id: String): List<String> = buildList {
        if (com.faceswap.studio.engine.LiteRtModel.available(dir, id)) add("GPU")
        if (int8(id) != null) add("NPU int8")
    }

    /** Bytes used by a model and all its twins. */
    fun bytesOnDisk(id: String): Long = companions(id).sumOf { it.length() }

    private fun companions(id: String) = listOf("$id.onnx", "$id.tflite", "$id.tflite.json", "$id.qdq.onnx", "$id.sample.bin").map { File(dir, it) }.filter { it.exists() }

    val totalBytes: Long get() = dir.listFiles()?.sumOf { it.length() } ?: 0L

    val requiredReady get() = ModelCatalog.all.filter { it.required }.all { file(it).exists() }
    val missingRequiredBytes get() = ModelCatalog.all.filter { it.required && !file(it).exists() }.sumOf { it.sizeBytes }

    private fun set(id: String, s: ModelState) = _states.update { it + (id to s) }

    fun delete(spec: ModelSpec) {
        companions(spec.id).forEach { it.delete() }
        File(dir, spec.fileName + ".part").delete()
        if (spec.id.startsWith("inswapper")) emapCache.delete()
        com.faceswap.studio.engine.Backends.clearCompiled(context, spec.id)
        set(spec.id, ModelState.Missing)
    }

    /** Resumable download with CRC32 verification against the published .hash file. */
    suspend fun download(spec: ModelSpec) = withContext(Dispatchers.IO) {
        val target = file(spec)
        val part = File(dir, spec.fileName + ".part")
        try {
            val expectedHash = spec.hashUrl?.let { u -> runCatching { fetchText(u).trim().lowercase() }.getOrNull() }
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

    private fun isModelFile(name: String) = name.endsWith(".onnx") || name.endsWith(".tflite") || name.endsWith(".tflite.json") || name.endsWith(".sample.bin")
    /** Catalog id of any model file or twin: x.onnx, x.qdq.onnx, x.tflite, x.tflite.json, x.sample.bin → x */
    private fun idOf(name: String) = name.removeSuffix(".json").removeSuffix(".tflite").removeSuffix(".onnx").removeSuffix(".qdq")
        .removeSuffix(".bin").removeSuffix(".sample")
    /** The main model file (shown as Installing / Ready); twins install silently. */
    private fun isPrimary(name: String) = name.endsWith(".onnx") && !name.endsWith(".qdq.onnx")

    /** Copies models bundled in the APK into app storage (first launch / after update). Idempotent. */
    suspend fun installBundled() = withContext(Dispatchers.IO) {
        val bundled = context.assets.list("models")?.filter(::isModelFile) ?: return@withContext
        // Twins (TFLite + mapping, INT8, sample inputs) first and silently, then the main models (shown as Installing).
        // A twin is re-copied when the APK ships a different size (new CI conversion) – the NPU cache is then rebuilt.
        for (name in bundled.sortedBy { if (isPrimary(it)) 1 else 0 }) {
            val target = File(dir, name)
            val assetSize = runCatching { context.assets.openFd("models/$name").use { it.length } }.getOrDefault(-1L)
            if (target.exists() && (isPrimary(name) || assetSize < 0 || target.length() == assetSize)) continue
            val spec = ModelCatalog.byId(idOf(name))
            if (isPrimary(name) && spec != null) set(spec.id, ModelState.Installing)
            try {
                val part = File(dir, "$name.part")
                context.assets.open("models/$name").use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
                target.delete()
                check(part.renameTo(target))
                if (!isPrimary(name)) com.faceswap.studio.engine.Backends.clearCompiled(context, idOf(name))
                if (isPrimary(name) && spec != null) set(spec.id, ModelState.Ready)
            } catch (t: Throwable) {
                if (spec != null && isPrimary(name)) set(spec.id, ModelState.Failed("Install failed: ${t.message}"))
            }
        }
    }

    /**
     * Imports model files picked from device storage (USB-copied): single .onnx / .tflite / .tflite.json files or
     * model-pack .zip files (stored, as built by CI). Matched to the catalog by file name. Returns names imported.
     */
    suspend fun importFromUris(uris: List<android.net.Uri>): List<String> = withContext(Dispatchers.IO) {
        val done = ArrayList<String>()
        fun store(name: String, write: (File) -> Unit) {
            val base = if (name == "arcface.onnx") "mobilefaceswap_arcface.onnx" else name
            val spec = ModelCatalog.byId(idOf(base)) ?: return
            if (isPrimary(base)) set(spec.id, ModelState.Installing)
            try {
                val part = File(dir, "$base.part")
                write(part)
                File(dir, base).delete()
                check(part.renameTo(File(dir, base)))
                if (spec.id.startsWith("inswapper") && isPrimary(base)) emapCache.delete()
                com.faceswap.studio.engine.Backends.clearCompiled(context, spec.id)
                done += base
            } catch (t: Throwable) {
                set(spec.id, ModelState.Failed("Import failed: ${t.message}"))
            } finally {
                if (file(spec).exists()) set(spec.id, ModelState.Ready)
            }
        }
        for (uri in uris) {
            val name = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }?.substringAfterLast('/') ?: continue
            if (name.endsWith(".zip", ignoreCase = true)) {
                context.contentResolver.openInputStream(uri)?.use { raw ->
                    java.util.zip.ZipInputStream(raw.buffered(1 shl 20)).use { zip ->
                        while (true) {
                            val entry = zip.nextEntry ?: break
                            val base = entry.name.substringAfterLast('/')
                            if (entry.isDirectory || !isModelFile(base)) continue
                            store(base) { f -> f.outputStream().use { zip.copyTo(it, 1 shl 20) } }
                        }
                    }
                }
            } else if (isModelFile(name)) {
                store(name) { f -> context.contentResolver.openInputStream(uri)!!.use { input -> f.outputStream().use { input.copyTo(it, 1 shl 20) } } }
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
