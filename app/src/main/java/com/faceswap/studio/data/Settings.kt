package com.faceswap.studio.data

import android.content.Context
import com.faceswap.core.FaceRegion
import com.faceswap.core.FaceSelector
import com.faceswap.core.MaskOps
import com.faceswap.core.MaskType
import com.faceswap.core.ModelIds
import com.faceswap.core.SwapConfig
import com.faceswap.studio.engine.CpuInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject

/** Where model inference actually runs. */
enum class ProcessingMode(val label: String, val hint: String) {
    LOCAL("On this device", "Every model runs on this phone (CPU / XNNPACK / NNAPI / QNN / GPU)"),
    REMOTE("Remote server", "Images are sent to your own server, which hosts every model and does all processing"),
}

enum class Accelerator(val label: String, val hint: String) {
    AUTO("Auto (tuned)", "Fastest verified backend per model"),
    CPU("CPU", "Always works, slowest"),
    XNNPACK("XNNPACK", "Optimised CPU kernels"),
    NNAPI("NNAPI", "Tensor TPU via NNAPI (not on Snapdragon)"),
    QNN("NPU (QNN)", "Snapdragon Hexagon NPU"),
    GPU("GPU (LiteRT)", "Models with a GPU twin"),
}

/** [boost] = pixel-boost factor over the swapper's native size; [enhance] = run the face enhancer. */
enum class QualityPreset(val label: String, val boost: Int, val enhance: Boolean, val hint: String) {
    FAST("Fast", 1, false, "Native resolution, no enhancer"),
    BALANCED("Balanced", 1, true, "Native + face enhancer"),
    HIGH("High", 2, true, "2× pixel boost + enhancer"),
    ULTRA("Ultra", 4, true, "4× pixel boost + enhancer"),
}

data class AppSettings(
    val processingMode: ProcessingMode = ProcessingMode.LOCAL,
    /** Hostname or IP of the remote inference server (no scheme, e.g. "192.168.1.50" or "faceswap.example.com"). */
    val serverHost: String = "",
    val serverPort: Int = 8443,
    val serverUseTls: Boolean = true,
    /** Sent as "Authorization: Bearer <key>" if non-blank. Matches the key the setup script generated on the server. */
    val serverApiKey: String = "",
    val preset: QualityPreset = QualityPreset.BALANCED,
    val swapperId: String = ModelIds.HYPERSWAP_1A,
    val detectorId: String = ModelIds.YUNET,
    /** "auto" = preset decides (GPEN-256), "none", or a model id. */
    val enhancer: String = "auto",
    val selector: FaceSelector = FaceSelector.ALL,
    val landmarks68: Boolean = true,
    val maskTypes: Set<MaskType> = setOf(MaskType.BOX, MaskType.MODEL),
    val maskRegions: Set<FaceRegion> = FaceRegion.DEFAULT,
    val maskBlur: Float = 0.3f,
    val maskPaddingTop: Float = 0f,
    val maskPaddingBottom: Float = 0f,
    val swapOpacity: Float = 1f,
    val colorMatch: Float = 0f,
    val sharpen: Float = 0f,
    val enhanceBlend: Float = 0.8f,
    val codeformerFidelity: Float = 0.5f,
    val expressionRestore: Float = 0f,
    val debugOverlay: Boolean = false,
    val referenceThreshold: Float = 0.35f,
    val detectorSize: Int = 640,
    val accelerator: Accelerator = Accelerator.AUTO,
    val threads: Int = CpuInfo.bigCores,
    val rootBoost: Boolean = false,
    val videoMaxSide: Int = 1280,
    val videoEnhance: Boolean = false,
    val videoWorkers: Int = 2,
    val keepAudio: Boolean = true,
    val frameEnhancerId: String = ModelIds.SPAN_X4,
    /** Show an instant native-resolution preview while the full-quality photo renders. */
    val draftPreview: Boolean = true,
    /** Video: skip the enhancer on faces whose swap is already as sharp as the original frame. */
    val smartEnhance: Boolean = true,
) {
    fun nativeSize(): Int = if (swapperId.startsWith("inswapper")) 128 else 256

    fun enhancerFor(forVideo: Boolean): String? = when {
        forVideo && !videoEnhance -> null
        enhancer == "none" -> null
        enhancer == "auto" -> if (forVideo || preset.enhance) ModelIds.GPEN_256 else null
        else -> enhancer
    }

    fun toConfig(forVideo: Boolean = false) = SwapConfig(
        swapperId = swapperId,
        detectorId = detectorId,
        selector = selector,
        referenceThreshold = referenceThreshold,
        pixelBoost = nativeSize() * (if (forVideo) 1 else preset.boost),
        detectorSize = detectorSize,
        landmarks68 = landmarks68,
        maskTypes = maskTypes,
        maskRegions = maskRegions,
        maskBlur = maskBlur,
        maskPadding = floatArrayOf(maskPaddingTop, 0f, maskPaddingBottom, 0f),
        swapOpacity = swapOpacity,
        colorMatch = colorMatch,
        sharpen = sharpen,
        enhancerId = enhancerFor(forVideo),
        enhanceBlend = enhanceBlend,
        codeformerFidelity = codeformerFidelity,
        expressionRestore = expressionRestore,
        expressionAreas = setOf(MaskOps.Area.UPPER_FACE, MaskOps.Area.LOWER_FACE),
        debugOverlay = debugOverlay,
        adaptiveEnhance = forVideo && smartEnhance,
    )
}

/** Persists settings as one JSON blob (adding fields never breaks old installs). */
class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _settings = MutableStateFlow(read())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private fun read(): AppSettings {
        val d = AppSettings()
        val j = runCatching { JSONObject(prefs.getString("json", "{}")!!) }.getOrDefault(JSONObject())
        fun f(k: String, v: Float) = j.optDouble(k, v.toDouble()).toFloat()
        fun i(k: String, v: Int) = j.optInt(k, v)
        fun b(k: String, v: Boolean) = j.optBoolean(k, v)
        fun s(k: String, v: String) = j.optString(k, v)
        fun <T> set(k: String, v: Set<T>, parse: (String) -> T?): Set<T> =
            j.optJSONArray(k)?.let { a -> (0 until a.length()).mapNotNull { runCatching { parse(a.getString(it)) }.getOrNull() }.toSet() } ?: v
        return AppSettings(
            processingMode = enumOr(s("processingMode", d.processingMode.name), d.processingMode),
            serverHost = s("serverHost", d.serverHost),
            serverPort = i("serverPort", d.serverPort),
            serverUseTls = b("serverUseTls", d.serverUseTls),
            serverApiKey = s("serverApiKey", d.serverApiKey),
            preset = enumOr(s("preset", d.preset.name), d.preset),
            swapperId = s("swapperId", d.swapperId),
            detectorId = s("detectorId", d.detectorId),
            enhancer = s("enhancer", d.enhancer),
            selector = enumOr(s("selector", d.selector.name), d.selector),
            landmarks68 = b("landmarks68", d.landmarks68),
            maskTypes = set("maskTypes", d.maskTypes) { MaskType.valueOf(it) },
            maskRegions = set("maskRegions", d.maskRegions) { FaceRegion.valueOf(it) },
            maskBlur = f("maskBlur", d.maskBlur),
            maskPaddingTop = f("maskPaddingTop", d.maskPaddingTop),
            maskPaddingBottom = f("maskPaddingBottom", d.maskPaddingBottom),
            swapOpacity = f("swapOpacity", d.swapOpacity),
            colorMatch = f("colorMatch", d.colorMatch),
            sharpen = f("sharpen", d.sharpen),
            enhanceBlend = f("enhanceBlend", d.enhanceBlend),
            codeformerFidelity = f("codeformerFidelity", d.codeformerFidelity),
            expressionRestore = f("expressionRestore", d.expressionRestore),
            debugOverlay = b("debugOverlay", d.debugOverlay),
            referenceThreshold = f("referenceThreshold", d.referenceThreshold),
            detectorSize = i("detectorSize", d.detectorSize),
            accelerator = enumOr(s("accelerator", d.accelerator.name), d.accelerator),
            threads = i("threads", d.threads),
            rootBoost = b("rootBoost", d.rootBoost),
            videoMaxSide = i("videoMaxSide", d.videoMaxSide),
            videoEnhance = b("videoEnhance", d.videoEnhance),
            videoWorkers = i("videoWorkers", d.videoWorkers),
            keepAudio = b("keepAudio", d.keepAudio),
            frameEnhancerId = s("frameEnhancerId", d.frameEnhancerId),
            draftPreview = b("draftPreview", d.draftPreview),
            smartEnhance = b("smartEnhance", d.smartEnhance),
        )
    }

    private inline fun <reified T : Enum<T>> enumOr(v: String, d: T): T = runCatching { enumValueOf<T>(v) }.getOrDefault(d)

    fun update(transform: (AppSettings) -> AppSettings) {
        val s = transform(_settings.value)
        val j = JSONObject()
            .put("processingMode", s.processingMode.name).put("serverHost", s.serverHost).put("serverPort", s.serverPort)
            .put("serverUseTls", s.serverUseTls).put("serverApiKey", s.serverApiKey)
            .put("preset", s.preset.name).put("swapperId", s.swapperId).put("detectorId", s.detectorId).put("enhancer", s.enhancer)
            .put("selector", s.selector.name).put("landmarks68", s.landmarks68)
            .put("maskTypes", org.json.JSONArray(s.maskTypes.map { it.name })).put("maskRegions", org.json.JSONArray(s.maskRegions.map { it.name }))
            .put("maskBlur", s.maskBlur).put("maskPaddingTop", s.maskPaddingTop).put("maskPaddingBottom", s.maskPaddingBottom)
            .put("swapOpacity", s.swapOpacity).put("colorMatch", s.colorMatch).put("sharpen", s.sharpen).put("enhanceBlend", s.enhanceBlend)
            .put("codeformerFidelity", s.codeformerFidelity).put("expressionRestore", s.expressionRestore).put("debugOverlay", s.debugOverlay)
            .put("referenceThreshold", s.referenceThreshold).put("detectorSize", s.detectorSize).put("accelerator", s.accelerator.name)
            .put("threads", s.threads).put("rootBoost", s.rootBoost).put("videoMaxSide", s.videoMaxSide).put("videoEnhance", s.videoEnhance)
            .put("videoWorkers", s.videoWorkers).put("keepAudio", s.keepAudio).put("frameEnhancerId", s.frameEnhancerId)
            .put("draftPreview", s.draftPreview).put("smartEnhance", s.smartEnhance)
        prefs.edit().putString("json", j.toString()).apply()
        _settings.value = s
    }
}
