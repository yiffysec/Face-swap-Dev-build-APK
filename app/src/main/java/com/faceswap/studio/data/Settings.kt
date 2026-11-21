package com.faceswap.studio.data

import android.content.Context
import com.faceswap.core.FaceSelector
import com.faceswap.core.MaskShape
import com.faceswap.core.SwapConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class Accelerator(val label: String) { AUTO("Auto (tuned)"), CPU("CPU"), XNNPACK("XNNPACK"), NNAPI("NNAPI (NPU)") }

enum class QualityPreset(val label: String, val boost: Int, val enhance: Boolean) {
    FAST("Fast", 128, false), BALANCED("Balanced", 256, false), HIGH("High", 256, true), ULTRA("Ultra", 512, true)
}

data class AppSettings(
    val preset: QualityPreset = QualityPreset.BALANCED,
    val selector: FaceSelector = FaceSelector.ALL,
    val maskShape: MaskShape = MaskShape.BOX,
    val maskBlur: Float = 0.3f,
    val maskPaddingTop: Float = 0f,
    val maskPaddingBottom: Float = 0f,
    val swapOpacity: Float = 1f,
    val colorMatch: Float = 0f,
    val sharpen: Float = 0f,
    val enhanceBlend: Float = 0.8f,
    val referenceThreshold: Float = 0.35f,
    val detectorSize: Int = 640,
    val accelerator: Accelerator = Accelerator.AUTO,
    val threads: Int = com.faceswap.studio.engine.CpuInfo.bigCores,
    val videoMaxSide: Int = 1280,
    val videoEnhance: Boolean = false,
    val keepAudio: Boolean = true,
) {
    fun toConfig(forVideo: Boolean = false) = SwapConfig(
        selector = selector,
        referenceThreshold = referenceThreshold,
        pixelBoost = preset.boost,
        detectorSize = detectorSize,
        maskShape = maskShape,
        maskBlur = maskBlur,
        maskPadding = floatArrayOf(maskPaddingTop, 0f, maskPaddingBottom, 0f),
        swapOpacity = swapOpacity,
        colorMatch = colorMatch,
        sharpen = sharpen,
        enhance = if (forVideo) videoEnhance else preset.enhance,
        enhanceBlend = enhanceBlend,
    )
}

class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _settings = MutableStateFlow(read())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private fun read(): AppSettings {
        val d = AppSettings()
        return AppSettings(
            preset = enumOr(prefs.getString("preset", null), d.preset),
            selector = enumOr(prefs.getString("selector", null), d.selector),
            maskShape = enumOr(prefs.getString("maskShape", null), d.maskShape),
            maskBlur = prefs.getFloat("maskBlur", d.maskBlur),
            maskPaddingTop = prefs.getFloat("maskPaddingTop", d.maskPaddingTop),
            maskPaddingBottom = prefs.getFloat("maskPaddingBottom", d.maskPaddingBottom),
            swapOpacity = prefs.getFloat("swapOpacity", d.swapOpacity),
            colorMatch = prefs.getFloat("colorMatch", d.colorMatch),
            sharpen = prefs.getFloat("sharpen", d.sharpen),
            enhanceBlend = prefs.getFloat("enhanceBlend", d.enhanceBlend),
            referenceThreshold = prefs.getFloat("referenceThreshold", d.referenceThreshold),
            detectorSize = prefs.getInt("detectorSize", d.detectorSize),
            accelerator = enumOr(prefs.getString("accelerator", null), d.accelerator),
            threads = prefs.getInt("threads", d.threads),
            videoMaxSide = prefs.getInt("videoMaxSide", d.videoMaxSide),
            videoEnhance = prefs.getBoolean("videoEnhance", d.videoEnhance),
            keepAudio = prefs.getBoolean("keepAudio", d.keepAudio),
        )
    }

    private inline fun <reified T : Enum<T>> enumOr(v: String?, d: T): T = runCatching { enumValueOf<T>(v!!) }.getOrDefault(d)

    fun update(transform: (AppSettings) -> AppSettings) {
        val s = transform(_settings.value)
        prefs.edit()
            .putString("preset", s.preset.name).putString("selector", s.selector.name).putString("maskShape", s.maskShape.name)
            .putFloat("maskBlur", s.maskBlur).putFloat("maskPaddingTop", s.maskPaddingTop).putFloat("maskPaddingBottom", s.maskPaddingBottom)
            .putFloat("swapOpacity", s.swapOpacity).putFloat("colorMatch", s.colorMatch).putFloat("sharpen", s.sharpen)
            .putFloat("enhanceBlend", s.enhanceBlend).putFloat("referenceThreshold", s.referenceThreshold)
            .putInt("detectorSize", s.detectorSize).putString("accelerator", s.accelerator.name).putInt("threads", s.threads)
            .putInt("videoMaxSide", s.videoMaxSide).putBoolean("videoEnhance", s.videoEnhance).putBoolean("keepAudio", s.keepAudio)
            .apply()
        _settings.value = s
    }
}
