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
import com.faceswap.studio.data.ProcessingMode
import com.faceswap.studio.engine.ServerEndpoint
import com.faceswap.studio.engine.ServerStatus
import com.faceswap.studio.internal.InternalTesting
import com.faceswap.studio.util.Bitmaps
import com.faceswap.studio.util.Diagnostics
import com.faceswap.studio.video.VideoProgress
import com.faceswap.studio.video.VideoSwapper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
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
    /** Undo stack for studio tools (previous results). */
    val history: List<Bitmap> = emptyList(),
    val toolBusy: String? = null,
    val showOriginal: Boolean = false,
    val busy: Boolean = false,
    val status: String? = null,
    val setup: SwapSetup = SwapSetup(),
    val savedUri: Uri? = null,
    /** Last swap's per-stage times (detect, swap, enhance, …) in ms. */
    val timings: Map<String, Long> = emptyMap(),
    /** True while the instant draft is shown and the full-quality render is still running. */
    val refining: Boolean = false,
)

/** One row of the per-model performance table (Settings → Performance). */
data class PerfRow(val id: String, val title: String, val backend: String, val ms: Double?, val speedup: Double?, val loadedOn: String?, val twins: List<String>)

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
    /** One-shot navigation requests (e.g. an image shared into the app opens the Photo tab). */
    private val _navigate = MutableStateFlow<String?>(null); val navigate = _navigate.asStateFlow()
    fun consumeNavigation() { _navigate.value = null }

    fun dismissCrashReport() { Diagnostics.markSeen(); _crashReport.value = null }
    fun showLastReport() { _crashReport.value = Diagnostics.lastReport() ?: "No crash reports recorded.\n\nRecent steps:\n" + Diagnostics.currentCrumbs() }
    fun reportShareUri(): Uri? {
        val f = Diagnostics.reportFile().takeIf { it.exists() } ?: return null
        val out = File(getApplication<Application>().cacheDir, "outputs/faceswap_crash_report.txt").apply { parentFile?.mkdirs() }
        f.copyTo(out, overwrite = true)
        return FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", out)
    }

    /** Any failure in a background action becomes a message + breadcrumb instead of an app crash. */
    private val errors = kotlinx.coroutines.CoroutineExceptionHandler { _, t ->
        Diagnostics.crumb("action failed: $t")
        _toast.value = "Error: ${t.message ?: t.javaClass.simpleName}"
        _photo.update { if (it.busy || it.toolBusy != null) it.copy(busy = false, toolBusy = null, refining = false, status = "Error: ${t.message}") else it }
    }
    private fun launch(block: suspend kotlinx.coroutines.CoroutineScope.() -> Unit) = viewModelScope.launch(errors, block = block)

    private val downloadJobs = HashMap<String, Job>()
    private var videoJob: Job? = null
    private var photoTargetRgb: RgbImage? = null
    private var pendingCapture: File? = null

    val internalAvailable = InternalTesting.AVAILABLE
    val rootAvailable get() = a.perf.rootAvailable
    val adpfSupported get() = a.perf.adpfSupported
    val qnnAvailable: Boolean by lazy { com.faceswap.studio.engine.QnnPlugin.available(app) }

    fun modelTitle(id: String) = ModelCatalog.byId(id)?.title ?: id
    fun swapperLabel(id: String) = modelTitle(id).removeSuffix(" 128")

    // ---------------------------------------------------------------- processing mode (local device / remote server)
    val remote = a.remote
    val serverStatus: StateFlow<ServerStatus> = remote.status

    /** True if model [id] can actually be used right now, on whichever backend the current settings select. */
    fun modelAvailable(id: String): Boolean = if (settings.value.processingMode == ProcessingMode.REMOTE) remote.has(id) else models.has(id)

    /** Reactive form of "are the models this app needs to swap a face available" – drives the Root.kt gate screen. */
    val readyToSwap: StateFlow<Boolean> = combine(settings, models.states, remote.status) { s, _, _ ->
        if (s.processingMode == ProcessingMode.REMOTE) remote.requiredReady else models.requiredReady
    }.stateIn(viewModelScope, SharingStarted.Eagerly, models.requiredReady)

    /** Non-reactive snapshot for one-off checks (e.g. inside a button's onClick). */
    val processingReady: Boolean get() = if (settings.value.processingMode == ProcessingMode.REMOTE) remote.requiredReady else models.requiredReady

    fun connectToServer() {
        if (jobRunning()) { toast("Finish or cancel the running job first"); return }
        val s = settings.value
        launch {
            a.engine.release()
            remote.connect(ServerEndpoint(s.serverHost.trim(), s.serverPort, s.serverUseTls, s.serverApiKey.trim()))
        }
    }

    fun disconnectServer() = launch { a.engine.release(); remote.disconnect() }

    /** Short status for the header chip: backend mode + whether tuning ran (state – reading tuning results hits disk). */
    private val _engineStatus by lazy { MutableStateFlow(computeEngineStatus()) }
    val engineStatus: StateFlow<String> get() = _engineStatus
    private fun computeEngineStatus(): String {
        val s = settings.value
        if (s.accelerator != com.faceswap.studio.data.Accelerator.AUTO) return s.accelerator.label
        val best = a.tuning.best()
        if (best.isEmpty()) return "CPU · not tuned"
        val accel = best.values.count { it != com.faceswap.studio.engine.Backend.CPU && it != com.faceswap.studio.engine.Backend.XNNPACK }
        return if (accel > 0) "Accelerated · $accel models" else "CPU · tuned"
    }

    init {
        launch { library.load() }
        launch { models.installBundled() }
    }

    fun consumeToast() { _toast.value = null }
    private fun toast(msg: String) { _toast.value = msg }

    // ---------------------------------------------------------------- settings
    fun updateSettings(t: (AppSettings) -> AppSettings) {
        val before = settings.value.accelerator
        a.settings.update(t)
        if (settings.value.accelerator != before) _engineStatus.value = computeEngineStatus()
    }

    // ---------------------------------------------------------------- models
    fun download(spec: ModelSpec) {
        if (downloadJobs[spec.id]?.isActive == true) return
        downloadJobs[spec.id] = launch { models.download(spec) }
    }
    fun downloadRequired() = ModelCatalog.all.filter { it.required && !models.file(it).exists() }.forEach { download(it) }
    fun importModels(uris: List<Uri>) = launch {
        val names = models.importFromUris(uris)
        toast(if (names.isEmpty()) "No recognised model files (expected names like inswapper_128_fp16.onnx)" else "Imported ${names.joinToString()}")
    }
    fun cancelDownload(spec: ModelSpec) { downloadJobs.remove(spec.id)?.cancel() }
    private fun jobRunning() = _video.value.running || _photo.value.busy || _photo.value.toolBusy != null || _live.value.faceId != null

    fun deleteModel(spec: ModelSpec) {
        if (jobRunning()) { toast("Finish or cancel the running job first"); return }
        launch { a.engine.release(); models.delete(spec); refreshPerf() }
    }

    private suspend fun engine() = try {
        _engineLoading.value = true
        a.engine.get(settings.value)
    } catch (oom: OutOfMemoryError) {
        Diagnostics.crumb("engine: out of memory")
        a.engine.release()
        throw IllegalStateException("Not enough memory to load the models – close other apps and try again")
    } finally { _engineLoading.value = false }

    // ---------------------------------------------------------------- device tuning
    val tuning = a.tuning
    private val _tuningSummary = MutableStateFlow(summary()); val tuningSummary = _tuningSummary.asStateFlow()
    private var tunerLaunches = 0

    private val _perfRows = MutableStateFlow(perfRows()); val perfRows = _perfRows.asStateFlow()
    fun refreshPerf() { _perfRows.value = perfRows(); _tuningSummary.value = summary(); _engineStatus.value = computeEngineStatus(); _tuned.value = a.tuning.best().isNotEmpty() to a.tuning.isComplete() }
    /** (has results, last run complete) */
    private val _tuned = MutableStateFlow(a.tuning.best().isNotEmpty() to a.tuning.isComplete()); val tuned = _tuned.asStateFlow()

    private fun perfRows(): List<PerfRow> {
        val best = a.tuning.best()
        val trials = a.tuning.trials()
        return ModelCatalog.all.filter { models.has(it.id) }.map { spec ->
            val b = best[spec.id]
            val ms = b?.let { bb -> trials.firstOrNull { it.model == spec.id && it.backend == bb }?.avgMs }
            val cpu = trials.firstOrNull { it.model == spec.id && it.backend == com.faceswap.studio.engine.Backend.CPU && it.ok }?.avgMs
            PerfRow(spec.id, spec.title, b?.label ?: "CPU (not tuned)", ms, if (cpu != null && ms != null && ms > 0) cpu / ms else null,
                a.engine.loadedBackends[spec.id], models.twins(spec.id))
        }
    }

    private fun summary(): String? {
        val best = a.tuning.best()
        if (best.isEmpty()) return null
        val trials = a.tuning.trials()
        return best.entries.joinToString("\n") { (m, b) ->
            val ms = trials.firstOrNull { it.model == m && it.backend == b }?.avgMs
            val cpu = trials.firstOrNull { it.model == m && it.backend == com.faceswap.studio.engine.Backend.CPU && it.status == "ok" }?.avgMs
            "$m → ${b.label}" + (ms?.let { " · ${"%.0f".format(it)} ms" } ?: "") +
                (if (cpu != null && ms != null && b != com.faceswap.studio.engine.Backend.CPU) " (${"%.1f".format(cpu / ms)}× vs CPU)" else "")
        } + if (!a.tuning.isComplete()) "\n(incomplete)" else ""
    }

    /** Called before (re)starting the tuner. Returns false when it should stop retrying. */
    /** Models the current settings actually use (quick tune). */
    fun modelsInUse(): Set<String> {
        val s = settings.value
        val ids = mutableSetOf(s.detectorId, com.faceswap.core.ModelIds.ARCFACE, s.swapperId, com.faceswap.core.ModelIds.OPEN_NSFW)
        if (s.landmarks68) ids += com.faceswap.core.ModelIds.HRFFA
        if (com.faceswap.core.MaskType.REGION in s.maskTypes) ids += com.faceswap.core.ModelIds.BISENET
        s.enhancerFor(false)?.let { ids += it }
        s.enhancerFor(true)?.let { ids += it }
        if (s.expressionRestore > 0f) ids += listOf(com.faceswap.core.ModelIds.LP_FEATURE, com.faceswap.core.ModelIds.LP_MOTION, com.faceswap.core.ModelIds.LP_GENERATOR)
        if (liveAvailable()) ids += listOf(com.faceswap.core.ModelIds.MFS_RUNTIME, com.faceswap.core.ModelIds.MFS_BUILDER, com.faceswap.core.ModelIds.MFS_ARCFACE)
        return ids.filter { modelAvailable(it) }.toSet()
    }

    /** [all] = every installed model (slow); otherwise only the models in use, keeping other results. */
    fun beginTuning(fresh: Boolean, all: Boolean = false): Boolean {
        if (fresh && jobRunning()) { toast("Finish or cancel the running job first"); return false }
        if (fresh) {
            tunerLaunches = 0
            if (all) { a.tuning.reset(); a.tuning.scope = null }
            else modelsInUse().let { a.tuning.resetModels(it); a.tuning.scope = it }
        }
        if (tunerLaunches++ >= 10) { a.tuning.scope = null; toast("Tuning stopped after repeated crashes – results so far are used"); return false }
        launch { a.engine.release() } // free memory for the tuner process
        return true
    }

    fun onTunerFinished(ok: Boolean): Boolean {
        refreshPerf()
        if (ok || a.tuning.isComplete()) { a.tuning.scope = null; toast("Device tuning complete"); return false }
        return true // crashed / closed: caller relaunches to continue
    }

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

    private fun startImport(uri: Uri, live: Boolean, deleteAfter: File?) = launch {
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
        launch { saveImported(st, face, label.ifBlank { defaultLabel(st.live) }) }
    }
    fun cancelImport() { _import.value = null }

    private suspend fun saveImported(st: ImportState, face: TargetFace, label: String) {
        Diagnostics.crumb("import: saving to library (live=${st.live})")
        val saved = if (st.live) library.addLive(label, listOf(face.embedding), st.photo, face.thumb)
        else library.addGallery(label, listOf(face.embedding), face.thumb)
        Diagnostics.crumb("import: saved ${saved.id}")
        toast("Saved “${saved.label}” to library" + if (saved.isLive) " (live)" else "")
    }

    fun renameFace(id: String, label: String) = launch { library.rename(id, label) }
    fun deleteFace(id: String) = launch {
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
    fun openShared(uri: Uri, video: Boolean) {
        _navigate.value = if (video) "video" else "photo"
        if (video) setVideoTarget(uri) else setPhotoTarget(uri)
    }

    fun setPhotoTarget(uri: Uri) = launch {
        photoJob?.cancel()
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

    private var photoJob: Job? = null

    fun swapPhoto() {
        val rgb = photoTargetRgb ?: return
        val st = _photo.value
        val mappings = buildMappings(st.setup)
        if (mappings.isEmpty()) { toast("Pick a source face first"); return }
        photoJob?.cancel()
        photoJob = launch {
            _photo.update { it.copy(busy = true, status = "Swapping…", result = null, savedUri = null, timings = emptyMap(), refining = false) }
            Diagnostics.crumb("photo: swap ${rgb.width}x${rgb.height}, ${mappings.size} mapping(s), preset=${settings.value.preset}")
            val s = settings.value
            val cfg = s.toConfig()
            try {
                val e = engine()
                val t0 = System.currentTimeMillis()
                val plan = withContext(Dispatchers.Default) { e.analyze(rgb, mappings, cfg) }
                if (!plan.decision.allowed) { _photo.update { it.copy(busy = false, status = policyMessage(plan.decision)) }; return@launch }
                if (plan.assignments.isEmpty()) { _photo.update { it.copy(busy = false, status = if (plan.faces.isEmpty()) "No faces found" else "No face matched the selection") }; return@launch }
                // Instant draft (native resolution, no enhancer) while the full-quality pass runs – same plan, no re-detection.
                val heavy = cfg.pixelBoost > (if (cfg.swapperId.startsWith("inswapper")) 128 else 256) || cfg.enhancerId != null || cfg.expressionRestore > 0f
                if (s.draftPreview && heavy) {
                    val draft = withContext(Dispatchers.Default) { e.render(rgb, plan, cfg, draft = true) }
                    _photo.update { it.copy(result = Bitmaps.toBitmap(draft), showOriginal = false, refining = true,
                        status = "Preview in ${"%.1f".format((System.currentTimeMillis() - t0) / 1000f)}s · refining…") }
                }
                val img = withContext(Dispatchers.Default) { e.render(rgb, plan, cfg) }
                val ms = System.currentTimeMillis() - t0
                _photo.update {
                    it.copy(busy = false, refining = false, result = Bitmaps.toBitmap(img), showOriginal = false, timings = plan.timings,
                        status = "Swapped ${plan.assignments.size} of ${plan.faces.size} face(s) in ${"%.1f".format(ms / 1000f)}s")
                }
            } catch (c: kotlinx.coroutines.CancellationException) {
                throw c
            } catch (t: Throwable) {
                Diagnostics.crumb("photo: failed $t")
                _photo.update { it.copy(busy = false, refining = false, status = "Error: ${t.message}") }
            }
        }
    }

    fun toggleOriginal() = _photo.update { it.copy(showOriginal = !it.showOriginal) }

    // ---------------------------------------------------------------- studio tools (FaceFusion processors)
    sealed class StudioOp(val label: String) {
        data class Age(val current: Float, val target: Float) : StudioOp("Changing age…")
        data class Background(val mode: com.faceswap.core.Processors.Background) : StudioOp("Processing background…")
        data class Upscale(val modelId: String) : StudioOp("Upscaling…")
        data class Edit(val edit: com.faceswap.core.Processors.FaceEdit) : StudioOp("Editing face…")
        data class Enhance(val modelId: String) : StudioOp("Enhancing faces…")
    }

    /** True if the current result has transparent pixels (background removed) → saved/shared as PNG. */
    fun photoHasAlpha(): Boolean {
        val bmp = _photo.value.result ?: return false
        val row = IntArray(bmp.width)
        for (y in 0 until bmp.height step 2) { bmp.getPixels(row, 0, bmp.width, 0, y, bmp.width, 1); if (row.any { (it ushr 24) != 0xFF }) return true }
        return false
    }

    fun applyStudio(op: StudioOp) {
        val base = _photo.value.result ?: _photo.value.target ?: return
        launch {
            _photo.update { it.copy(toolBusy = op.label, status = null) }
            Diagnostics.crumb("studio: $op")
            try {
                val e = engine()
                val s = settings.value
                val out = withContext(Dispatchers.Default) {
                    val rgb = Bitmaps.toRgb(base)
                    val faces by lazy { e.detect(rgb, s.toConfig()) }
                    val needsFaces = op is StudioOp.Age || op is StudioOp.Edit || op is StudioOp.Enhance
                    if (needsFaces && faces.isEmpty()) throw IllegalStateException("No face found in this picture")
                    when (op) {
                        is StudioOp.Age -> faces.fold(rgb) { img, f -> e.processors.modifyAge(img, f, op.current, op.target, s.maskBlur) }
                        is StudioOp.Background -> e.processors.replaceBackground(rgb, op.mode)
                        is StudioOp.Upscale -> {
                            // cap the output at 64 MP (8192 px side): larger results don't fit in a Bitmap / memory
                            val scale = e.processors.frameScale(op.modelId)
                            val maxIn = 8192 / scale
                            val input = if (maxOf(rgb.width, rgb.height) > maxIn) {
                                val k = maxIn.toFloat() / maxOf(rgb.width, rgb.height)
                                com.faceswap.core.ImageOps.resize(rgb, (rgb.width * k).toInt(), (rgb.height * k).toInt())
                            } else rgb
                            e.processors.enhanceFrame(input, op.modelId)
                        }
                        is StudioOp.Edit -> faces.fold(rgb) { img, f -> e.processors.editFace(img, f, op.edit, s.maskBlur) }
                        is StudioOp.Enhance -> {
                            enhanceAll(e, rgb, faces, op.modelId, s)
                        }
                    }
                }
                _photo.update { it.copy(result = Bitmaps.toBitmap(out), history = budgeted(it.history + base), toolBusy = null, showOriginal = false, savedUri = null) }
                // heavy processors (FRAN 1 GB+ of activations, LivePortrait) are unloaded right away
                if (op is StudioOp.Age || op is StudioOp.Edit || op is StudioOp.Upscale) a.engine.trim()
            } catch (t: Throwable) {
                Diagnostics.crumb("studio failed: $t")
                _photo.update { it.copy(toolBusy = null, status = "Error: ${t.message}") }
            }
        }
    }

    private fun enhanceAll(e: com.faceswap.core.FaceSwapEngine, rgb: RgbImage, faces: List<com.faceswap.core.Face>, id: String, s: AppSettings): RgbImage {
        // enhance only (no swap): run the restorer directly on each detected face
        if (faces.isEmpty()) return rgb
        val out = rgb.copy()
        val enh = e.restorer(id)
        for (f in faces) {
            val m = com.faceswap.core.Geometry.estimateSimilarity(f.landmarks, enh.template, enh.size.toFloat())
            val crop = com.faceswap.core.ImageOps.warpAffine(out, m, enh.size, enh.size)
            val mixed = com.faceswap.core.ImageOps.mix(crop, enh.enhance(crop, s.codeformerFidelity), s.enhanceBlend)
            com.faceswap.core.ImageOps.pasteBack(out, mixed, com.faceswap.core.ImageOps.boxMask(enh.size, enh.size, 0.3f), m)
        }
        return out
    }

    /** Background replacement with a picked image (decoded off the main thread). */
    fun replaceBackgroundWith(uri: Uri) = launch {
        val img = withContext(Dispatchers.IO) { Bitmaps.toRgb(Bitmaps.decode(getApplication(), uri, 2048)) }
        applyStudio(StudioOp.Background(com.faceswap.core.Processors.Background.Image(img)))
    }

    /** Undo steps are full-size bitmaps: keep at most 8 and ~320 MB. */
    private fun budgeted(list: List<Bitmap>): List<Bitmap> {
        val out = ArrayList<Bitmap>()
        var bytes = 0L
        for (b in list.asReversed()) {
            bytes += b.allocationByteCount
            if (out.size >= 8 || (out.isNotEmpty() && bytes > 320L shl 20)) break
            out.add(0, b)
        }
        return out
    }

    fun undoStudio() = _photo.update { st ->
        if (st.history.isEmpty()) st else st.copy(result = st.history.last(), history = st.history.dropLast(1))
    }

    /** Use the current result as the next target (chain swaps / tools). */
    fun resultAsTarget() {
        val r = _photo.value.result ?: return
        photoJob?.cancel()
        launch {
            _photo.update { it.copy(busy = true, status = "Detecting faces…") }
            val rgb = withContext(Dispatchers.Default) { Bitmaps.toRgb(r) }
            val faces = try { detectTargets(rgb) } finally { _photo.update { it.copy(busy = false) } }
            photoTargetRgb = rgb
            _photo.update { it.copy(target = r, result = null, history = emptyList(), timings = emptyMap(), savedUri = null, status = null,
                setup = it.setup.copy(targetFaces = faces, assignments = emptyMap())) }
        }
    }

    // ---------------------------------------------------------------- live preview (MobileFaceSwap)
    data class LiveState(val faceId: String? = null, val ready: Boolean = false, val fps: Float = 0f, val ms: Float = 0f, val faces: Int = 0, val message: String? = null)
    private val _live = MutableStateFlow(LiveState()); val live = _live.asStateFlow()
    private val _liveFrame = MutableStateFlow<Bitmap?>(null); val liveFrame = _liveFrame.asStateFlow()
    @Volatile private var liveIdentity: com.faceswap.core.MobileFaceSwap.Identity? = null
    @Volatile private var liveEngine: com.faceswap.core.FaceSwapEngine? = null
    private var liveSources: List<SourceFace> = emptyList()
    private var liveFrames = 0
    private var liveExplicit = false
    private var liveT0 = 0L
    private var liveWorkMs = 0f
    private var liveJob: Job? = null
    private var liveActive = false

    /**
     * Camera live preview needs a per-frame model call fast enough for ~15-30 fps – only realistic on-device.
     * Over a network, even a fast server, round-trip latency per frame makes it unusable, so live preview stays
     * local-only; photo and video swaps (one request per job, not per frame) work the same in both modes.
     */
    fun liveAvailable() = settings.value.processingMode == ProcessingMode.LOCAL &&
        models.has(com.faceswap.core.ModelIds.MFS_RUNTIME) && models.has(com.faceswap.core.ModelIds.MFS_BUILDER) && models.has(com.faceswap.core.ModelIds.MFS_ARCFACE)

    /** Camera screen visible: sustained NPU clocks for the duration. */
    fun startLiveSession() { if (!liveActive) { liveActive = true; a.perf.beginJob(useRoot = false); a.engine.pin() } }

    /** Builds the MobileFaceSwap identity from the selected library face (once per face). */
    fun prepareLive(faceId: String) {
        if (_live.value.faceId == faceId && (_live.value.ready || liveJob?.isActive == true)) return
        liveJob?.cancel()
        liveIdentity = null
        _live.value = LiveState(faceId = faceId, message = "Preparing identity…")
        liveJob = launch {
            try {
                val e = engine()
                val lf = library.faces.value.firstOrNull { it.id == faceId } ?: error("Face not found")
                val id = withContext(Dispatchers.Default) {
                    val photo = (lf.photoFile ?: lf.thumbFile).let { Bitmaps.decodeFile(it) } ?: error("Missing face image")
                    val rgb = Bitmaps.toRgb(photo)
                    val f = e.detect(rgb, settings.value.toConfig()).maxByOrNull { it.area } ?: error("No face in library image")
                    e.mobileFaceSwap()?.buildIdentity(rgb, f) ?: error("Live preview models not installed")
                }
                liveEngine = e; liveIdentity = id; liveSources = listOf(lf.toSource()); liveFrames = 0; liveT0 = System.currentTimeMillis()
                _live.value = LiveState(faceId = faceId, ready = true)
            } catch (c: kotlinx.coroutines.CancellationException) {
                throw c
            } catch (t: Throwable) {
                _live.value = LiveState(faceId = faceId, message = t.message ?: "Could not start live preview")
            }
        }
    }

    fun stopLive() {
        liveJob?.cancel(); liveIdentity = null; liveEngine = null
        _live.value = LiveState(); _liveFrame.value = null
        if (liveActive) { liveActive = false; a.perf.endJob(); a.engine.unpin() }
    }

    /** Camera analyzer thread: swaps the newest frame and publishes it. Unchanged frames pass through until ready. */
    fun onLiveFrame(bmp: Bitmap) {
        val id = liveIdentity
        val e = liveEngine
        if (id == null || e == null) { _liveFrame.value = bmp; return }
        val t = System.nanoTime()
        val rgb = Bitmaps.toRgb(bmp)
        if (liveFrames % 30 == 0) liveExplicit = runCatching { e.policy.isExplicit(rgb) }.getOrDefault(false)
        liveFrames++
        val decision = e.policy.evaluate(liveExplicit, liveSources)
        if (!decision.allowed) { _live.update { it.copy(message = policyMessage(decision)) }; _liveFrame.value = bmp; return }
        val cfg = settings.value.toConfig().copy(detectorSize = 320, landmarks68 = false)
        val faces = try {
            e.detect(rgb, cfg).also { e.previewSwap(rgb, faces = it, identity = id, blur = settings.value.maskBlur) }
        } catch (t: Throwable) {
            // engine rebuilt / models deleted underneath: ask for the face again instead of failing every frame
            liveIdentity = null; liveEngine = null
            _live.update { it.copy(ready = false, faceId = null, message = "Live preview stopped (${t.message ?: "engine changed"}) – tap a face to restart") }
            _liveFrame.value = bmp; return
        }
        _liveFrame.value = Bitmaps.toBitmap(rgb)
        val work = (System.nanoTime() - t) / 1e6f
        liveWorkMs = if (liveWorkMs == 0f) work else liveWorkMs * 0.9f + work * 0.1f
        val el = (System.currentTimeMillis() - liveT0).coerceAtLeast(1)
        if (liveFrames % 5 == 0) _live.update { it.copy(fps = liveFrames * 1000f / el, ms = liveWorkMs, faces = faces.size, message = null) }
        if (liveFrames > 300) { liveFrames = 0; liveT0 = System.currentTimeMillis() }
    }

    fun savePhoto() {
        val bmp = _photo.value.result ?: return
        launch {
            val uri = withContext(Dispatchers.IO) { Bitmaps.saveToGallery(getApplication(), bmp, png = photoHasAlpha()) }
            _photo.update { it.copy(savedUri = uri) }
            toast(if (uri != null) "Saved to Pictures/FaceSwapStudio" else "Save failed")
        }
    }

    fun photoShareUri(): Uri? {
        val bmp = _photo.value.result ?: return null
        val png = photoHasAlpha()
        val f = File(getApplication<Application>().cacheDir, "outputs/share." + if (png) "png" else "jpg").apply { parentFile?.mkdirs() }
        f.outputStream().use { bmp.compress(if (png) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG, 95, it) }
        return FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", f)
    }

    // ---------------------------------------------------------------- video
    fun setVideoTarget(uri: Uri) = launch {
        if (_video.value.running) { toast("A video is already processing"); return@launch }
        _video.update { VideoState(uri = uri, status = "Reading video…", setup = SwapSetup(quickSources = it.setup.quickSources)) }
        try {
            val (frame, dur) = withContext(Dispatchers.IO) {
                val r = MediaMetadataRetriever() // not AutoCloseable before API 29
                try {
                    r.setDataSource(getApplication(), uri)
                    val d = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                    val bmp = r.getFrameAtTime(minOf(1_000_000L, d * 500), MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    bmp?.copy(Bitmap.Config.ARGB_8888, false) to d
                } finally { r.release() }
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
        // previous render is replaced (Save copies it to the gallery) – the cache would otherwise grow by a video per run
        File(getApplication<Application>().cacheDir, "outputs").listFiles()?.filter { it.name.startsWith("faceswap_") && it.extension == "mp4" }?.forEach { it.delete() }
        val out = File(getApplication<Application>().cacheDir, "outputs/faceswap_${System.currentTimeMillis()}.mp4")
        videoJob = launch {
            _keepScreenOn.value = true
            _video.update { it.copy(running = true, progress = null, output = null, savedUri = null, status = "Loading models…") }
            Diagnostics.crumb("video: start maxSide=${s.videoMaxSide} enhance=${s.videoEnhance}")
            a.engine.pin()
            try {
                val e = engine()
                _video.update { it.copy(status = null) }
                a.perf.beginJob(useRoot = s.rootBoost)
                val r = VideoSwapper(getApplication()).run(e, uri, out, mappings, s.toConfig(forVideo = true), s.videoMaxSide, s.keepAudio, s.videoWorkers, a.perf) { p ->
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
                a.perf.endJob()
                a.engine.unpin()
                _keepScreenOn.value = false
            }
        }
    }

    fun cancelVideo() { videoJob?.cancel() }

    fun saveVideo() {
        val f = _video.value.output ?: return
        launch {
            val uri = withContext(Dispatchers.IO) { Bitmaps.saveVideoToGallery(getApplication(), f) }
            _video.update { it.copy(savedUri = uri) }
            toast(if (uri != null) "Saved to Movies/FaceSwapStudio" else "Save failed")
        }
    }

    fun videoShareUri(): Uri? = _video.value.output?.let {
        FileProvider.getUriForFile(getApplication(), getApplication<Application>().packageName + ".files", it)
    }
}
