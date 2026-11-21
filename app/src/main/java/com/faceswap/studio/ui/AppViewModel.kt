package com.faceswap.studio.ui

import android.app.Application
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.faceswap.core.Face
import com.faceswap.core.FaceMapping
import com.faceswap.core.PolicyDecision
import com.faceswap.core.RgbImage
import com.faceswap.core.SourceFace
import com.faceswap.core.SourceProvenance
import com.faceswap.studio.FaceSwapApp
import com.faceswap.studio.data.AppSettings
import com.faceswap.studio.data.LibraryFace
import com.faceswap.studio.data.ModelCatalog
import com.faceswap.studio.data.ModelSpec
import com.faceswap.studio.internal.InternalTesting
import com.faceswap.studio.util.Bitmaps
import com.faceswap.studio.util.Diagnostics
import com.faceswap.studio.video.VideoProgress
import com.faceswap.studio.video.VideoSwapper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** A face found in the current target (photo or video preview frame). */
data class TargetFace(val index: Int, val face: Face, val thumb: Bitmap, val embedding: FloatArray)

/** Shared source/assignment state for photo and video. */
data class SwapSetup(
    val targetFaces: List<TargetFace> = emptyList(),
    /** Quick mode: library ids combined into one source identity, applied per selector. */
    val quickSources: Set<String> = emptySet(),
    /** Per-face mode: target face index -> library id. Non-empty switches to per-face mapping. */
    val assignments: Map<Int, String> = emptyMap(),
) {
    val perFace get() = assignments.isNotEmpty()
    val ready get() = perFace || quickSources.isNotEmpty()
}

data class PhotoState(
    val target: Bitmap? = null,
    val result: Bitmap? = null,
    val showOriginal: Boolean = false,
    val busy: Boolean = false,
    val status: String? = null,
    val setup: SwapSetup = SwapSetup(),
    val savedUri: Uri? = null,
)

data class VideoState(
    val uri: Uri? = null,
    val preview: Bitmap? = null,
    val durationMs: Long = 0,
    val setup: SwapSetup = SwapSetup(),
    val progress: VideoProgress? = null,
    val running: Boolean = false,
    val output: File? = null,
    val status: String? = null,
    val savedUri: Uri? = null,
)

/** Pending gallery import: user picks which detected face to keep. */
data class ImportState(val image: RgbImage, val faces: List<TargetFace>, val live: Boolean, val photo: Bitmap)

class AppViewModel(app: Application) : AndroidViewModel(app) {
    private val a = app as FaceSwapApp
    val models = a.models
    val library = a.library
    val settings: StateFlow<AppSettings> = a.settings.settings

    private val _photo = MutableStateFlow(PhotoState()); val photo = _photo.asStateFlow()
    private val _video = MutableStateFlow(VideoState()); val video = _video.asStateFlow()
    private val _import = MutableStateFlow<ImportState?>(null); val import = _import.asStateFlow()
    private val _toast = MutableStateFlow<String?>(null); val toast = _toast.asStateFlow()
    private val _internalUnlocked = MutableStateFlow(false); val internalUnlocked = _internalUnlocked.asStateFlow()
    private val _keepScreenOn = MutableStateFlow(false); val keepScreenOn = _keepScreenOn.asStateFlow()
    private val _engineLoading = MutableStateFlow(false); val engineLoading = _engineLoading.asStateFlow()
    private val _importing = MutableStateFlow(false); val importing = _importing.asStateFlow()
    private val _crashReport = MutableStateFlow(Diagnostics.pendingReport()); val crashReport = _crashReport.asStateFlow()

    fun dismissCrashReport() { Diagnostics.markSeen(); _crashReport.value = null }
    fun showLastReport() { _crashReport.value = Diagnostics.lastReport() ?: "No crash reports recorded.\n\nRecent steps:\n" + Diagnostics.currentCrumbs() }
    fun reportShareUri(): Uri? {
        val f = Diagnostics.reportFile().takeIf { it.exists() } ?: return null
        val out = File(getApplication<Application>().cacheDir, "outputs/faceswap_crash_report.txt").apply { parentFile?.mkdirs() }
        f.copyTo(out, overwrite = true)
        return FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", out)
    }

    private val downloadJobs = HashMap<String, Job>()
    private var videoJob: Job? = null
    private var photoTargetRgb: RgbImage? = null
    private var pendingCapture: File? = null

    val internalAvailable = InternalTesting.AVAILABLE

    init {
        viewModelScope.launch { library.load() }
        viewModelScope.launch { models.installBundled() }
    }

    fun consumeToast() { _toast.value = null }
    private fun toast(msg: String) { _toast.value = msg }

    // ---------------------------------------------------------------- settings
    fun updateSettings(t: (AppSettings) -> AppSettings) = a.settings.update(t)

    // ---------------------------------------------------------------- models
    fun download(spec: ModelSpec) {
        if (downloadJobs[spec.id]?.isActive == true) return
        downloadJobs[spec.id] = viewModelScope.launch { models.download(spec) }
    }
    fun downloadRequired() = ModelCatalog.all.filter { it.required && !models.file(it).exists() }.forEach { download(it) }
    fun importModels(uris: List<Uri>) = viewModelScope.launch {
        val names = models.importFromUris(uris)
        toast(if (names.isEmpty()) "No recognised model files (expected names like inswapper_128_fp16.onnx)" else "Imported ${names.joinToString()}")
    }
    fun cancelDownload(spec: ModelSpec) { downloadJobs.remove(spec.id)?.cancel() }
    fun deleteModel(spec: ModelSpec) = viewModelScope.launch { a.engine.release(); models.delete(spec) }

    private suspend fun engine() = try {
        _engineLoading.value = true
        a.engine.get(settings.value)
    } catch (oom: OutOfMemoryError) {
        Diagnostics.crumb("engine: out of memory")
        a.engine.release()
        throw IllegalStateException("Not enough memory to load the models – close other apps and try again")
    } finally { _engineLoading.value = false }

    // ---------------------------------------------------------------- internal testing
    fun unlockInternal(pass: String): Boolean {
        val ok = InternalTesting.AVAILABLE && InternalTesting.checkPassphrase(pass)
        if (ok) { _internalUnlocked.value = true; a.engine.internalLiveCaptureMode = true }
        toast(if (ok) "Internal testing mode unlocked" else "Wrong passphrase")
        return ok
    }
    fun lockInternal() { _internalUnlocked.value = false; a.engine.internalLiveCaptureMode = false }

    // ---------------------------------------------------------------- library
    /** Returns a content uri for the camera app to write into. */
    fun newCaptureUri(): Uri {
        val f = File(getApplication<Application>().cacheDir, "captures/capture_${System.currentTimeMillis()}.jpg")
        f.parentFile?.mkdirs()
        pendingCapture = f
        return FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", f)
    }

    fun onCaptured(success: Boolean) {
        // pendingCapture is lost if Android killed the app while the camera was open: fall back to the newest capture.
        val f = pendingCapture ?: File(getApplication<Application>().cacheDir, "captures").listFiles()
            ?.filter { it.length() > 0 && System.currentTimeMillis() - it.lastModified() < 10 * 60_000 }?.maxByOrNull { it.lastModified() }
        Diagnostics.crumb("camera returned ok=$success file=${f?.name} size=${f?.length()}")
        pendingCapture = null
        if (f == null) { toast("Capture was lost – please try again"); return }
        if (!success || !f.exists() || f.length() == 0L) { f.delete(); return }
        startImport(Uri.fromFile(f), live = true, deleteAfter = f)
    }

    fun importFromGallery(uri: Uri) = startImport(uri, live = false, deleteAfter = null)

    private fun startImport(uri: Uri, live: Boolean, deleteAfter: File?) = viewModelScope.launch {
        _importing.value = true
        try {
            Diagnostics.crumb("import: decoding image (live=$live)")
            val (bmp, rgb) = withContext(Dispatchers.IO) {
                val bmp = Bitmaps.decode(getApplication(), uri, 1600)
                bmp to Bitmaps.toRgb(bmp)
            }
            Diagnostics.crumb("import: decoded ${bmp.width}x${bmp.height}, loading engine")
            val e = engine()
            val faces = withContext(Dispatchers.Default) {
                Diagnostics.crumb("import: detecting faces")
                val found = e.detect(rgb, settings.value.toConfig()).sortedByDescending { it.area }
                Diagnostics.crumb("import: ${found.size} face(s), computing identity")
                found.mapIndexed { i, f -> TargetFace(i, f, Bitmaps.faceThumb(rgb, f), e.embed(rgb, f)) }
            }
            Diagnostics.crumb("import: identity ok")
            if (faces.isEmpty()) { toast("No face found in that photo"); return@launch }
            if (faces.size == 1) saveImported(ImportState(rgb, faces, live, bmp), faces[0], defaultLabel(live))
            else _import.value = ImportState(rgb, faces, live, bmp)
        } catch (t: Throwable) {
            Diagnostics.crumb("import failed: $t")
            toast("Import failed: ${t.message ?: t.javaClass.simpleName}")
        } finally {
            _importing.value = false
            deleteAfter?.delete()
        }
    }

    private fun defaultLabel(live: Boolean) = (if (live) "Live " else "Face ") + (library.faces.value.size + 1)

    fun chooseImportFace(face: TargetFace, label: String) {
        val st = _import.value ?: return
        _import.value = null
        viewModelScope.launch { saveImported(st, face, label.ifBlank { defaultLabel(st.live) }) }
    }
    fun cancelImport() { _import.value = null }

    private suspend fun saveImported(st: ImportState, face: TargetFace, label: String) {
        Diagnostics.crumb("import: saving to library (live=${st.live})")
        val saved = if (st.live) library.addLive(label, listOf(face.embedding), st.photo, face.thumb)
        else library.addGallery(label, listOf(face.embedding), face.thumb)
        Diagnostics.crumb("import: saved ${saved.id}")
        toast("Saved “${saved.label}” to library" + if (saved.isLive) " (live)" else "")
    }

    fun renameFace(id: String, label: String) = viewModelScope.launch { library.rename(id, label) }
    fun deleteFace(id: String) = viewModelScope.launch {
        library.delete(id)
        _photo.update { it.copy(setup = it.setup.without(id)) }
        _video.update { it.copy(setup = it.setup.without(id)) }
    }

    private fun SwapSetup.without(id: String) = copy(quickSources = quickSources - id, assignments = assignments.filterValues { it != id })

    // ---------------------------------------------------------------- shared setup editing
    fun toggleQuickSource(video: Boolean, id: String) = editSetup(video) {
        it.copy(quickSources = if (id in it.quickSources) it.quickSources - id else it.quickSources + id)
    }
    fun assign(video: Boolean, faceIndex: Int, id: String?) = editSetup(video) {
        it.copy(assignments = if (id == null) it.assignments - faceIndex else it.assignments + (faceIndex to id))
    }
    fun clearAssignments(video: Boolean) = editSetup(video) { it.copy(assignments = emptyMap()) }

    private fun editSetup(video: Boolean, f: (SwapSetup) -> SwapSetup) {
        if (video) _video.update { it.copy(setup = f(it.setup)) } else _photo.update { it.copy(setup = f(it.setup), result = null) }
    }

    private fun buildMappings(setup: SwapSetup): List<FaceMapping> {
        val lib = library.faces.value.associateBy { it.id }
        if (setup.perFace) {
            return setup.assignments.mapNotNull { (idx, id) ->
                val lf = lib[id] ?: return@mapNotNull null
                val tf = setup.targetFaces.firstOrNull { it.index == idx } ?: return@mapNotNull null
                FaceMapping(lf.toSource(), tf.embedding)
            }
        }
        val chosen = setup.quickSources.mapNotNull { lib[it] }
        if (chosen.isEmpty()) return emptyList()
        return listOf(FaceMapping(combine(chosen)))
    }

    /** Several library entries of the same person → one averaged identity (better likeness). */
    private fun combine(faces: List<LibraryFace>): SourceFace =
        if (faces.size == 1) faces[0].toSource()
        else SourceFace(
            faces.joinToString("+") { it.id }, faces.joinToString(" + ") { it.label }, faces.flatMap { it.embeddings },
            if (faces.all { it.isLive }) SourceProvenance.LIVE_CAMERA else SourceProvenance.GALLERY,
        )

    private suspend fun detectTargets(rgb: RgbImage): List<TargetFace> {
        val e = engine()
        return withContext(Dispatchers.Default) {
            e.detect(rgb, settings.value.toConfig()).mapIndexed { i, f -> TargetFace(i, f, Bitmaps.faceThumb(rgb, f, 160), e.embed(rgb, f)) }
        }
    }

    fun policyMessage(d: PolicyDecision): String = when (d) {
        PolicyDecision.Allowed -> ""
        PolicyDecision.ExplicitRequiresVerification ->
            if (internalAvailable && !_internalUnlocked.value) "This target needs identity verification. In this internal build, unlock internal testing in Settings and use a live-captured face."
            else "This target needs identity verification. Verify your identity to use your own face on it."
        is PolicyDecision.ExplicitIdentityMismatch -> "Only your own verified face can be used on this target."
        PolicyDecision.ExplicitRequiresLiveCapture -> "Internal build: this target needs a face captured with the in-app camera (marked LIVE in the library)."
    }

    // ---------------------------------------------------------------- photo
    fun setPhotoTarget(uri: Uri) = viewModelScope.launch {
        _photo.update { PhotoState(busy = true, status = "Detecting faces…", setup = SwapSetup(quickSources = it.setup.quickSources)) }
        try {
            val bmp = withContext(Dispatchers.IO) { Bitmaps.decode(getApplication(), uri) }
            val rgb = Bitmaps.toRgb(bmp)
            photoTargetRgb = rgb
            _photo.update { it.copy(target = bmp) }
            val faces = detectTargets(rgb)
            _photo.update { it.copy(busy = false, status = if (faces.isEmpty()) "No faces found in this photo" else null, setup = it.setup.copy(targetFaces = faces)) }
        } catch (t: Throwable) {
            _photo.update { it.copy(busy = false, status = t.message ?: "Could not open image") }
        }
    }

    fun swapPhoto() {
        val rgb = photoTargetRgb ?: return
        val st = _photo.value
        val mappings = buildMappings(st.setup)
        if (mappings.isEmpty()) { toast("Pick a source face first"); return }
        viewModelScope.launch {
            _photo.update { it.copy(busy = true, status = "Swapping…", result = null, savedUri = null) }
            Diagnostics.crumb("photo: swap ${rgb.width}x${rgb.height}, ${mappings.size} mapping(s), preset=${settings.value.preset}")
            try {
                val e = engine()
                val r = withContext(Dispatchers.Default) { e.process(rgb, mappings, settings.value.toConfig()) }
                if (!r.decision.allowed) _photo.update { it.copy(busy = false, status = policyMessage(r.decision)) }
                else _photo.update {
                    it.copy(busy = false, result = Bitmaps.toBitmap(r.image), showOriginal = false,
                        status = "Swapped ${r.facesSwapped} of ${r.facesFound} face(s) in ${"%.1f".format(r.millis / 1000f)}s")
                }
            } catch (t: Throwable) {
                Diagnostics.crumb("photo: failed $t")
                _photo.update { it.copy(busy = false, status = "Error: ${t.message}") }
            }
        }
    }

    fun toggleOriginal() = _photo.update { it.copy(showOriginal = !it.showOriginal) }

    fun savePhoto() {
        val bmp = _photo.value.result ?: return
        viewModelScope.launch {
            val uri = withContext(Dispatchers.IO) { Bitmaps.saveToGallery(getApplication(), bmp) }
            _photo.update { it.copy(savedUri = uri) }
            toast(if (uri != null) "Saved to Pictures/FaceSwapStudio" else "Save failed")
        }
    }

    fun photoShareUri(): Uri? {
        val bmp = _photo.value.result ?: return null
        val f = File(getApplication<Application>().cacheDir, "outputs/share.jpg").apply { parentFile?.mkdirs() }
        f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        return FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", f)
    }

    // ---------------------------------------------------------------- video
    fun setVideoTarget(uri: Uri) = viewModelScope.launch {
        if (_video.value.running) { toast("A video is already processing"); return@launch }
        _video.update { VideoState(uri = uri, status = "Reading video…", setup = SwapSetup(quickSources = it.setup.quickSources)) }
        try {
            val (frame, dur) = withContext(Dispatchers.IO) {
                MediaMetadataRetriever().use { r ->
                    r.setDataSource(getApplication(), uri)
                    val d = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                    val bmp = r.getFrameAtTime(minOf(1_000_000L, d * 500), MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    bmp?.copy(Bitmap.Config.ARGB_8888, false) to d
                }
            }
            if (frame == null) { _video.update { it.copy(status = "Could not read video frames") }; return@launch }
            val faces = detectTargets(Bitmaps.toRgb(frame))
            _video.update { it.copy(preview = frame, durationMs = dur, status = if (faces.isEmpty()) "No faces in the preview frame – quick mode will still swap faces found later" else null, setup = it.setup.copy(targetFaces = faces)) }
        } catch (t: Throwable) {
            _video.update { it.copy(status = t.message ?: "Could not open video") }
        }
    }

    fun startVideo() {
        val st = _video.value
        val uri = st.uri ?: return
        val mappings = buildMappings(st.setup)
        if (mappings.isEmpty()) { toast("Pick a source face first"); return }
        val s = settings.value
        val out = File(getApplication<Application>().cacheDir, "outputs/faceswap_${System.currentTimeMillis()}.mp4")
        videoJob = viewModelScope.launch {
            _keepScreenOn.value = true
            _video.update { it.copy(running = true, progress = null, output = null, savedUri = null, status = "Loading models…") }
            Diagnostics.crumb("video: start maxSide=${s.videoMaxSide} enhance=${s.videoEnhance}")
            try {
                val e = engine()
                _video.update { it.copy(status = null) }
                val r = VideoSwapper(getApplication()).run(e, uri, out, mappings, s.toConfig(forVideo = true), s.videoMaxSide, s.keepAudio) { p ->
                    _video.update { it.copy(progress = p) }
                }
                val swapped = _video.value.progress?.facesSwapped ?: 0
                _video.update {
                    it.copy(running = false, output = r.file,
                        status = "Done: ${r.frames} frames in ${"%.0f".format(r.seconds)}s" + (if (swapped == 0) " – no faces were swapped (check the source / content policy)" else "") +
                            if (s.keepAudio && !r.audioCopied) " (no compatible audio track)" else "")
                }
            } catch (c: kotlinx.coroutines.CancellationException) {
                out.delete(); _video.update { it.copy(running = false, status = "Cancelled") }
            } catch (t: Throwable) {
                Diagnostics.crumb("video: failed $t")
                out.delete(); _video.update { it.copy(running = false, status = "Error: ${t.message}") }
            } finally {
                _keepScreenOn.value = false
            }
        }
    }

    fun cancelVideo() { videoJob?.cancel() }

    fun saveVideo() {
        val f = _video.value.output ?: return
        viewModelScope.launch {
            val uri = withContext(Dispatchers.IO) { Bitmaps.saveVideoToGallery(getApplication(), f) }
            _video.update { it.copy(savedUri = uri) }
            toast(if (uri != null) "Saved to Movies/FaceSwapStudio" else "Save failed")
        }
    }

    fun videoShareUri(): Uri? = _video.value.output?.let {
        FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", it)
    }
}
