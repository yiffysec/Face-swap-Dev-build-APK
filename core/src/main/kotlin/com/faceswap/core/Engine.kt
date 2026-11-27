package com.faceswap.core

import java.io.Closeable
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/** One identity built from one or more photos (embeddings averaged for a more stable likeness). */
class SourceFace(
    val id: String,
    val label: String,
    embeddings: List<FloatArray>,
    val provenance: SourceProvenance = SourceProvenance.GALLERY,
    /**
     * One representative photo of this identity (first sample passed to [FaceSwapEngine.buildSource]), kept
     * alongside the ArcFace embeddings above. The ONNX swappers never touch this – they work purely from the
     * averaged embedding – but a diffusion swapper (see [ModelIds.DIFFUSION_SWAPPERS]) needs an actual source
     * image to condition on, not just a 512-float vector, since its own identity encoder (InstantID/PuLID's
     * insightface pass, server-side) runs on real pixels. Null for a source built with no samples.
     */
    val referenceImage: RgbImage? = null,
) {
    val embeddings: List<FloatArray> = embeddings.map { Geometry.normalize(it) }
    val averaged: FloatArray = Geometry.normalize(FloatArray(512) { i -> this.embeddings.sumOf { it[i].toDouble() }.toFloat() })
    /** Per-swapper latent cache (keyed by swapper id). */
    internal val latents = ConcurrentHashMap<String, FloatArray>()
}

/** Maps a source identity onto target faces. [reference] = embedding of the target person to replace (null = selector rule). */
data class FaceMapping(val source: SourceFace, val reference: FloatArray? = null)

enum class FaceSelector { ALL, LARGEST, LEFTMOST, RIGHTMOST, CENTER }

/** Mask components, combined with min() like FaceFusion. */
enum class MaskType { BOX, MODEL, REGION, AREA, ELLIPSE }

/** Stable model ids (= file name without .onnx). */
object ModelIds {
    const val SCRFD = "scrfd_2.5g"
    const val YUNET = "yunet_2023_mar"
    const val HRFFA = "hrffa"
    const val ARCFACE = "arcface_w600k_r50"
    const val HYPERSWAP_1A = "hyperswap_1a_256"
    const val HYPERSWAP_1B = "hyperswap_1b_256"
    const val HYPERSWAP_1C = "hyperswap_1c_256"
    const val INSWAPPER = "inswapper_128_fp16"
    const val BISENET = "bisenet_resnet_18"
    const val GPEN_256 = "gpen_bfr_256"
    const val GPEN_512 = "gpen_bfr_512"
    const val GFPGAN = "gfpgan_1.4"
    const val RESTOREFORMER = "restoreformer_plus_plus"
    const val CODEFORMER = "codeformer"
    const val OPEN_NSFW = "open_nsfw"
    const val FRAN = "fran"
    const val MODNET = "modnet"
    const val U2NETP = "u2netp"
    const val ESRGAN_X2 = "real_esrgan_x2_fp16"
    const val ESRGAN_X4 = "real_esrgan_x4_fp16"
    const val SPAN_X4 = "span_kendata_x4"
    /** Qualcomm AI Hub Real-ESRGAN-General-x4v3 (BSD-3): 1.2 M params, NPU-friendly. */
    const val ESRGAN_GENERAL_X4 = "real_esrgan_general_x4v3"
    const val LP_FEATURE = "live_portrait_feature_extractor"
    const val LP_MOTION = "live_portrait_motion_extractor"
    const val LP_GENERATOR = "live_portrait_generator"
    const val LP_STITCHER = "live_portrait_stitcher"
    const val LP_EYE = "live_portrait_eye_retargeter"
    const val LP_LIP = "live_portrait_lip_retargeter"
    const val MFS_RUNTIME = "MobileFaceSwapUNetRuntime"
    const val MFS_BUILDER = "BuildFaceSwap"
    const val MFS_ARCFACE = "mobilefaceswap_arcface"

    // Diffusion-based swappers (InstantID / PuLID / REFace) – server-only, never bundled or run on-device: there is
    // no local file these ever resolve to (see ModelManager/RemoteCatalog's `has()`), so they only ever appear in
    // Settings when ProcessingMode.REMOTE is active and the connected server's /v1/diffusion/health lists them.
    // See server/diffusion_manifest.yaml for what each one actually is.
    const val INSTANTID_SDXL = "instantid_sdxl"
    const val PULID_SDXL = "pulid_sdxl"
    const val REFACE_SD14 = "reface_sd14"
    val DIFFUSION_SWAPPERS = listOf(INSTANTID_SDXL, PULID_SDXL, REFACE_SD14)

    val SWAPPERS = listOf(HYPERSWAP_1A, HYPERSWAP_1B, HYPERSWAP_1C, INSWAPPER) + DIFFUSION_SWAPPERS
    val ENHANCERS = listOf(GPEN_256, GFPGAN, GPEN_512, RESTOREFORMER, CODEFORMER)
    val FRAME_ENHANCERS = listOf(ESRGAN_GENERAL_X4, ESRGAN_X2, ESRGAN_X4, SPAN_X4)
}

/**
 * A [FaceSwapModel] whose actual swap happens server-side on a whole source photo (not a per-crop embedding) –
 * implemented by [com.faceswap.studio.engine] since it needs network access, which this module (`core`) doesn't
 * depend on. [FaceSwapEngine.render] calls [setActiveSource] with the current source's [SourceFace.referenceImage]
 * right before invoking [FaceSwapModel.swap] on each assignment, since that method's own signature (crop + a
 * 512-float latent) has no room for a whole source image – this is the narrow side channel that gets it there
 * without changing [FaceSwapModel] for every other swapper.
 */
interface DiffusionSourceAware {
    fun setActiveSource(image: RgbImage?)
}

data class SwapConfig(
    val swapperId: String = ModelIds.HYPERSWAP_1A,
    val detectorId: String = ModelIds.YUNET,
    val selector: FaceSelector = FaceSelector.ALL,
    /** Cosine similarity needed to match a target face to a mapping reference. */
    val referenceThreshold: Float = 0.35f,
    /** Crop resolution: model size = native; 2×/4× model size = pixel boost (k² sub-swaps, sharper). */
    val pixelBoost: Int = 256,
    val detectorSize: Int = 640,
    val minDetectionScore: Float = 0.5f,
    val minFaceSize: Int = 24,
    /** Refine alignment with 68-point landmarks (HRFFA) when available. */
    val landmarks68: Boolean = true,
    val maskTypes: Set<MaskType> = setOf(MaskType.BOX, MaskType.MODEL),
    val maskRegions: Set<FaceRegion> = FaceRegion.DEFAULT,
    val maskAreas: Set<MaskOps.Area> = setOf(MaskOps.Area.UPPER_FACE, MaskOps.Area.LOWER_FACE, MaskOps.Area.MOUTH),
    val maskBlur: Float = 0.3f,
    /** top, right, bottom, left as fraction of crop. */
    val maskPadding: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
    val swapOpacity: Float = 1f,
    val colorMatch: Float = 0f,
    val sharpen: Float = 0f,
    /** Face enhancer model id, or null for none. */
    val enhancerId: String? = null,
    val enhanceBlend: Float = 0.8f,
    val codeformerFidelity: Float = 0.5f,
    /** LivePortrait expression restore (0 = off, 1 = full target expression). */
    val expressionRestore: Float = 0f,
    val expressionAreas: Set<MaskOps.Area> = setOf(MaskOps.Area.UPPER_FACE, MaskOps.Area.LOWER_FACE),
    /** Draw boxes / landmarks / masks over the result (developer overlay). */
    val debugOverlay: Boolean = false,
    /**
     * Skip the face enhancer for a face whose swap is already at least as sharp as the original target face
     * (typical for small faces in video). The enhancer can't add real detail there, it only costs time.
     */
    val adaptiveEnhance: Boolean = false,
)

data class SwapResult(
    val image: RgbImage,
    val facesFound: Int,
    val facesSwapped: Int,
    val decision: PolicyDecision,
    val millis: Long,
    val timings: Map<String, Long> = emptyMap(),
)

/**
 * Lightweight tracker for video: associates detections across frames by IoU and smooths landmarks with an
 * exponential moving average, which removes most frame-to-frame jitter/flicker.
 */
class FaceTracker(private val smoothing: Float = 0.55f) {
    private class Track(var face: Face, var misses: Int = 0, var embedding: FloatArray? = null)
    private val tracks = ArrayList<Track>()

    fun update(faces: List<Face>): List<Face> {
        val out = ArrayList<Face>()
        val used = HashSet<Track>()
        for (f in faces) {
            val best = tracks.filter { it !in used }.maxByOrNull { Geometry.iou(it.face, f) }
            if (best != null && Geometry.iou(best.face, f) > 0.3f) {
                val a = smoothing
                val move = kotlin.math.abs(best.face.centerX - f.centerX) + kotlin.math.abs(best.face.centerY - f.centerY)
                val lm = if (move > f.width * 0.15f) f.landmarks else FloatArray(10) { best.face.landmarks[it] * a + f.landmarks[it] * (1 - a) }
                val nf = f.copyWith(lm).also { it.embedding = best.embedding }
                best.face = nf; best.misses = 0; used += best
                out += nf
            } else {
                val t = Track(f); tracks += t; used += t; out += f
            }
        }
        tracks.forEach { if (it !in used) it.misses++ }
        tracks.removeAll { it.misses > 5 }
        return out
    }

    /** Cache identity embedding on the track so recognition only runs once per tracked person. */
    fun remember(face: Face, embedding: FloatArray) {
        tracks.firstOrNull { it.face === face }?.embedding = embedding
    }

    fun reset() = tracks.clear()
}

/**
 * Orchestrates all models. Models are loaded lazily by id from [resolve] (id → file, null if not installed)
 * through [loader], which lets the app pick the backend (CPU / XNNPACK / NNAPI / QNN / LiteRT GPU) per model.
 */
class FaceSwapEngine(
    private val resolve: (String) -> File?,
    private val loader: (String, File) -> InferenceModel,
    verifier: IdentityVerifier = NoIdentityVerifier,
    private val emapCache: File? = null,
    /** Builds a [FaceSwapModel] for one of [ModelIds.DIFFUSION_SWAPPERS]; null (the local-engine default) means
     *  those ids are simply unavailable – see [EngineHolder.buildRemoteEngine] for the remote-engine wiring. */
    private val diffusionSwapper: ((String) -> FaceSwapModel)? = null,
) : Closeable {
    private val models = ConcurrentHashMap<String, Closeable>()

    val policy = ContentPolicy({ ContentAnalyser(load(ModelIds.OPEN_NSFW)) }, verifier)

    fun has(id: String) = resolve(id)?.exists() == true

    private fun load(id: String): InferenceModel {
        val f = resolve(id)?.takeIf { it.exists() } ?: throw MissingModelException(id)
        return loader(id, f)
    }

    @Suppress("UNCHECKED_CAST")
    @Synchronized
    private fun <T : Closeable> cached(key: String, make: () -> T): T = models.getOrPut(key) { make() } as T

    fun detector(id: String): FaceDetector = cached("det:$id") {
        if (id == ModelIds.YUNET) YunetDetector(load(id)) else ScrfdDetector(load(id))
    }
    val recognizer: FaceRecognizer get() = cached("rec") { FaceRecognizer(load(ModelIds.ARCFACE)) }
    fun landmarker(): Landmarker68? = if (has(ModelIds.HRFFA)) cached("lm68") { Landmarker68(load(ModelIds.HRFFA)) } else null
    fun parser(): FaceParser? = if (has(ModelIds.BISENET)) cached("parser") { FaceParser(load(ModelIds.BISENET)) } else null
    fun swapper(id: String): FaceSwapModel = cached("swap:$id") {
        if (id in ModelIds.DIFFUSION_SWAPPERS) {
            diffusionSwapper?.invoke(id) ?: throw MissingModelException(id)
        } else {
            val f = resolve(id)?.takeIf { it.exists() } ?: throw MissingModelException(id)
            if (id.startsWith("inswapper")) InSwapper(loader(id, f), f, emapCache) else HyperSwap(loader(id, f), f)
        }
    }
    fun restorer(id: String): FaceRestorer = cached("enh:$id") { FaceRestorer(load(id), id) }

    /** MobileFaceSwap (live preview swapper), null if its three model files are not installed. */
    fun mobileFaceSwap(): MobileFaceSwap? =
        if (has(ModelIds.MFS_RUNTIME) && has(ModelIds.MFS_BUILDER) && has(ModelIds.MFS_ARCFACE))
            cached("mfs") { MobileFaceSwap(load(ModelIds.MFS_RUNTIME), load(ModelIds.MFS_BUILDER), load(ModelIds.MFS_ARCFACE)) }
        else null

    /**
     * Real-time preview swap with MobileFaceSwap (~15× faster than HyperSwap, lower fidelity). Modifies [frame] in place.
     * Explicit-content policy is enforced by the caller per frame (same rule as [process]).
     */
    fun previewSwap(frame: RgbImage, faces: List<Face>, identity: MobileFaceSwap.Identity, blur: Float = 0.3f) {
        val mfs = mobileFaceSwap() ?: return
        val original = frame.copy()
        for (f in faces) {
            val m = Geometry.estimateSimilarity(f.landmarks, mfs.template, mfs.size.toFloat())
            val crop = ImageOps.warpAffine(original, m, mfs.size, mfs.size)
            val r = mfs.swap(crop, identity)
            val mask = MaskOps.minimum(listOf(r.mask!!, ImageOps.boxMask(mfs.size, mfs.size, blur)))
            ImageOps.pasteBack(frame, r.image, mask, m)
        }
    }
    private var processorsInstance: Processors? = null
    val processors: Processors
        @Synchronized get() = processorsInstance ?: Processors(::has, ::load).also { processorsInstance = it }

    /** Unloads one cached model group (free memory, e.g. before the tuner runs). */
    fun unloadAll() {
        models.values.forEach { runCatching { it.close() } }; models.clear()
        processorsInstance?.close(); processorsInstance = null
    }

    // ------------------------------------------------------------------------------------------------- analysis

    fun detect(img: RgbImage, config: SwapConfig = SwapConfig()): List<Face> {
        val det = if (has(config.detectorId)) detector(config.detectorId) else detector(ModelIds.SCRFD)
        val faces = det.detect(img, config.detectorSize, config.minDetectionScore)
            .filter { it.width >= config.minFaceSize && it.height >= config.minFaceSize }
            .sortedBy { it.x1 }
        if (config.landmarks68) landmarker()?.let { lm -> faces.forEach { refine(img, it, lm) } }
        return faces
    }

    private fun refine(img: RgbImage, f: Face, lm: Landmarker68) {
        val l68 = lm.detect(img, f)
        f.landmarks68 = l68
        f.landmarks = Geometry.fiveFrom68(l68)
    }

    fun embed(img: RgbImage, face: Face): FloatArray = face.embedding ?: recognizer.embed(img, face).also { face.embedding = it }

    fun buildSource(id: String, label: String, samples: List<Pair<RgbImage, Face>>): SourceFace =
        SourceFace(id, label, samples.map { (img, f) -> embed(img, f) }, referenceImage = samples.firstOrNull()?.first)

    private fun latentFor(src: SourceFace, swapper: FaceSwapModel): FloatArray =
        src.latents.getOrPut(swapper.id) { swapper.sourceLatent(src.averaged) }

    /** Decides which target faces receive which source. */
    fun plan(img: RgbImage, faces: List<Face>, mappings: List<FaceMapping>, config: SwapConfig, tracker: FaceTracker? = null): List<Pair<Face, SourceFace>> {
        if (faces.isEmpty() || mappings.isEmpty()) return emptyList()
        val referenced = mappings.filter { it.reference != null }
        if (referenced.isNotEmpty()) {
            val result = ArrayList<Pair<Face, SourceFace>>()
            for (f in faces) {
                val e = embed(img, f)
                tracker?.remember(f, e)
                val best = referenced.maxBy { Geometry.cosine(e, it.reference!!) }
                if (Geometry.cosine(e, best.reference!!) >= config.referenceThreshold) result += f to best.source
            }
            return result
        }
        val src = mappings.first().source
        val chosen = when (config.selector) {
            FaceSelector.ALL -> faces
            FaceSelector.LARGEST -> listOf(faces.maxBy { it.area })
            FaceSelector.LEFTMOST -> listOf(faces.minBy { it.centerX })
            FaceSelector.RIGHTMOST -> listOf(faces.maxBy { it.centerX })
            FaceSelector.CENTER -> listOf(faces.minBy { kotlin.math.abs(it.centerX - img.width / 2f) + kotlin.math.abs(it.centerY - img.height / 2f) })
        }
        return chosen.map { it to src }
    }

    // ------------------------------------------------------------------------------------------------- pipeline

    /** Result of the sequential analysis stage (detection, tracking, mapping). Rendering it is thread-safe. */
    class FramePlan internal constructor(
        val decision: PolicyDecision,
        val faces: List<Face>,
        val assignments: List<Pair<Face, SourceFace>>,
        internal val timingsMap: MutableMap<String, Long>,
    ) {
        /** Per-stage times in ms accumulated by analyze + render (detect, swap, enhance, …). */
        val timings: Map<String, Long> get() = synchronized(timingsMap) { LinkedHashMap(timingsMap) }
    }

    /**
     * Stage 1 (must run in frame order for video because of the tracker): content policy, detection, tracking,
     * source assignment. [explicitOverride] lets video callers reuse a cached content check.
     */
    fun analyze(target: RgbImage, mappings: List<FaceMapping>, config: SwapConfig, tracker: FaceTracker? = null, explicitOverride: Boolean? = null): FramePlan {
        val timings = LinkedHashMap<String, Long>()
        fun <T> timed(k: String, b: () -> T): T { val s = System.currentTimeMillis(); return b().also { timings[k] = (timings[k] ?: 0) + System.currentTimeMillis() - s } }
        val explicit = explicitOverride ?: timed("policy") { policy.isExplicit(target) }
        val decision = policy.evaluate(explicit, mappings.map { it.source }.distinct())
        if (!decision.allowed) return FramePlan(decision, emptyList(), emptyList(), timings)
        var faces = timed("detect") { detect(target, config) }
        if (tracker != null) faces = tracker.update(faces)
        val plan = timed("plan") { plan(target, faces, mappings, config, tracker) }
        return FramePlan(decision, faces, plan, timings)
    }

    /**
     * Stage 2 (thread-safe, can run on several frames in parallel): swap, expression restore, enhance, overlay.
     * [draft] = fast preview of the same plan: native swapper resolution, no expression restore, no enhancer.
     */
    fun render(target: RgbImage, fp: FramePlan, config: SwapConfig, draft: Boolean = false): RgbImage {
        if (!fp.decision.allowed || fp.assignments.isEmpty()) return if (config.debugOverlay) target.copy().also { Debugger.draw(it, fp.faces, emptySet()) } else target
        val timings: MutableMap<String, Long> = if (draft) LinkedHashMap() else fp.timingsMap
        fun <T> timed(k: String, b: () -> T): T { val s = System.currentTimeMillis(); return b().also { synchronized(timings) { timings[k] = (timings[k] ?: 0) + System.currentTimeMillis() - s } } }
        val cfg = if (draft) config.copy(pixelBoost = 0, maskTypes = config.maskTypes - MaskType.REGION) else config
        val out = target.copy()
        val swapper = swapper(cfg.swapperId)
        for ((face, src) in fp.assignments) timed("swap") {
            (swapper as? DiffusionSourceAware)?.setActiveSource(src.referenceImage)
            swapFace(out, target, face, latentFor(src, swapper), swapper, cfg)
        }
        if (draft) return out
        if (cfg.expressionRestore > 0f && processors.hasLivePortrait) for ((face, _) in fp.assignments) timed("expression") {
            processors.restoreExpression(target, out, face, cfg.expressionRestore, cfg.expressionAreas, cfg.maskBlur)
        }
        if (cfg.enhancerId != null && has(cfg.enhancerId)) {
            val enh = restorer(cfg.enhancerId)
            for ((face, _) in fp.assignments) timed("enhance") { enhanceFace(out, target, face, enh, cfg) }
        }
        if (cfg.debugOverlay) Debugger.draw(out, fp.faces, fp.assignments.map { it.first }.toSet())
        return out
    }

    /** Full pipeline on one image / frame. */
    fun process(
        target: RgbImage,
        mappings: List<FaceMapping>,
        config: SwapConfig,
        tracker: FaceTracker? = null,
        explicitOverride: Boolean? = null,
    ): SwapResult {
        val t0 = System.currentTimeMillis()
        val fp = analyze(target, mappings, config, tracker, explicitOverride)
        val img = render(target, fp, config)
        return SwapResult(img, fp.faces.size, fp.assignments.size, fp.decision, System.currentTimeMillis() - t0, fp.timings)
    }

    private fun buildMask(size: Int, crop: RgbImage, swapped: RgbImage, modelMask: FloatArray?, face: Face, m: Affine, config: SwapConfig): FloatArray {
        val masks = ArrayList<FloatArray>()
        val types = config.maskTypes
        if (MaskType.BOX in types || types.isEmpty()) masks += ImageOps.boxMask(size, size, config.maskBlur, config.maskPadding)
        if (MaskType.ELLIPSE in types) masks += ImageOps.ellipseMask(size, size, config.maskBlur)
        if (MaskType.MODEL in types && modelMask != null) {
            val mm = if (modelMask.size == size * size) modelMask else MaskOps.resize(modelMask, kotlin.math.sqrt(modelMask.size.toFloat()).toInt(), kotlin.math.sqrt(modelMask.size.toFloat()).toInt(), size, size)
            masks += MaskOps.gaussian(mm, size, size, size / 128f)
        }
        if (MaskType.REGION in types) parser()?.let { p ->
            // regions of the ORIGINAL crop (like FaceFusion): hair / hands / objects in front of the face stay untouched.
            // One parser pass per face – parsing the swapped crop as well doubled the cost for no visible gain.
            masks += p.regionMask(crop, config.maskRegions)
        }
        if (MaskType.AREA in types) face.landmarks68?.let { l68 -> masks += MaskOps.areaMask(Geometry.transformPoints(l68, m), config.maskAreas, size, size) }
        return MaskOps.minimum(masks)
    }

    private fun swapFace(out: RgbImage, original: RgbImage, face: Face, latent: FloatArray, swapper: FaceSwapModel, config: SwapConfig) {
        val ms = swapper.size
        val k = (config.pixelBoost / ms).coerceIn(1, 4) // pixelBoost 0 (draft) → native
        val size = ms * k
        val m = Geometry.estimateSimilarity(face.landmarks, swapper.template, size.toFloat())
        val crop = ImageOps.warpAffine(original, m, size, size)
        var modelMask: FloatArray? = null
        var swapped = if (k == 1) swapper.swap(crop, latent).also { modelMask = it.mask }.image else {
            // pixel boost: interleaved sub-sampling into k² native tiles, swap each, re-interleave
            val result = RgbImage(size, size)
            val maskAcc = FloatArray(size * size)
            var anyMask = false
            for (ti in 0 until k) for (tj in 0 until k) {
                val tile = RgbImage(ms, ms)
                for (y in 0 until ms) for (x in 0 until ms) tile.pixels[y * ms + x] = crop.pixels[(y * k + ti) * size + (x * k + tj)]
                val s = swapper.swap(tile, latent)
                for (y in 0 until ms) for (x in 0 until ms) {
                    val di = (y * k + ti) * size + (x * k + tj)
                    result.pixels[di] = s.image.pixels[y * ms + x]
                    s.mask?.let { maskAcc[di] = it[y * ms + x]; anyMask = true }
                }
            }
            if (anyMask) modelMask = maskAcc
            result
        }
        val mask = buildMask(size, crop, swapped, modelMask, face, m, config)
        if (config.colorMatch > 0f) swapped = ImageOps.colorTransfer(swapped, crop, mask, config.colorMatch)
        if (config.sharpen > 0f) swapped = ImageOps.sharpen(swapped, config.sharpen)
        ImageOps.pasteBack(out, swapped, mask, m, config.swapOpacity)
    }

    /** Faces skipped by [SwapConfig.adaptiveEnhance] (for stats / tests). */
    @Volatile var enhanceSkipped = 0; private set

    private fun enhanceFace(out: RgbImage, original: RgbImage, face: Face, enh: FaceRestorer, config: SwapConfig) {
        val m = Geometry.estimateSimilarity(face.landmarks, enh.template, enh.size.toFloat())
        val crop = ImageOps.warpAffine(out, m, enh.size, enh.size)
        if (config.adaptiveEnhance) {
            val before = ImageOps.sharpness(ImageOps.warpAffine(original, m, enh.size, enh.size))
            if (ImageOps.sharpness(crop) >= before * 0.9) { enhanceSkipped++; return }
        }
        val restored = enh.enhance(crop, config.codeformerFidelity)
        val mixed = ImageOps.mix(crop, restored, config.enhanceBlend)
        val box = ImageOps.boxMask(enh.size, enh.size, config.maskBlur.coerceAtLeast(0.2f), floatArrayOf(0.02f, 0.02f, 0.02f, 0.02f))
        ImageOps.pasteBack(out, mixed, box, m)
    }

    override fun close() {
        unloadAll()
        policy.close()
        processorsInstance?.close()
    }
}

class MissingModelException(val modelId: String) : IllegalStateException("Model not installed: $modelId")
