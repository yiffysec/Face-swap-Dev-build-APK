package com.faceswap.studio.ui

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Dns
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Masks
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.faceswap.core.FaceRegion
import com.faceswap.core.MaskType
import com.faceswap.core.ModelIds
import com.faceswap.studio.BuildConfig
import com.faceswap.studio.data.Accelerator
import com.faceswap.studio.data.AppSettings
import com.faceswap.studio.data.ProcessingMode
import com.faceswap.studio.data.QualityPreset
import com.faceswap.studio.engine.CpuInfo
import com.faceswap.studio.engine.ServerStatus
import com.faceswap.studio.engine.TunerActivity
import com.faceswap.studio.ui.theme.Good
import com.faceswap.studio.ui.theme.Warn

@Composable
fun SettingsRoot(vm: AppViewModel, page: SettingsPage, onPage: (SettingsPage) -> Unit) {
    when (page) {
        SettingsPage.HUB -> SettingsHub(vm, onPage)
        SettingsPage.SWAP -> SubPage("Swapping", { onPage(SettingsPage.HUB) }) { SwapSettings(vm) }
        SettingsPage.MASKS -> SubPage("Masks & blending", { onPage(SettingsPage.HUB) }) { MaskSettings(vm) }
        SettingsPage.PERFORMANCE -> SubPage("Performance", { onPage(SettingsPage.HUB) }) { PerformanceSettings(vm) }
        SettingsPage.SERVER -> SubPage("Processing", { onPage(SettingsPage.HUB) }) { ServerSettings(vm) }
        SettingsPage.MODELS -> ModelsScreen(vm, gate = false, onBack = { onPage(SettingsPage.HUB) })
        SettingsPage.DIAGNOSTICS -> SubPage("Diagnostics & testing", { onPage(SettingsPage.HUB) }) { DiagnosticsSettings(vm) }
    }
}

@Composable
fun SubPage(title: String, onBack: () -> Unit, content: @Composable () -> Unit) {
    Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Text(title, style = MaterialTheme.typography.titleLarge)
        }
        content()
        Spacer(Modifier.height(28.dp))
    }
}

@Composable
private fun SettingsHub(vm: AppViewModel, onPage: (SettingsPage) -> Unit) {
    val s by vm.settings.collectAsState()
    val states by vm.models.states.collectAsState()
    val installed = remember(states) { states.values.count { it is com.faceswap.studio.data.ModelState.Ready } }
    Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp)) {
        ScreenHeader("Settings", "FaceSwap Studio ${BuildConfig.VERSION_NAME}")
        Panel(padding = 6.dp) {
            val serverStatus by vm.serverStatus.collectAsState()
            NavRow(Icons.Default.Dns, "Processing", processingSubtitle(s.processingMode, serverStatus)) { onPage(SettingsPage.SERVER) }
            NavRow(Icons.Default.AutoFixHigh, "Swapping", "${s.preset.label} · ${vm.swapperLabel(s.swapperId)} · enhancer ${s.enhancerFor(false)?.let(vm::modelTitle) ?: "off"}") { onPage(SettingsPage.SWAP) }
            NavRow(Icons.Default.Masks, "Masks & blending", s.maskTypes.joinToString { it.name.lowercase() } + " · feather ${"%.2f".format(s.maskBlur)}") { onPage(SettingsPage.MASKS) }
            if (s.processingMode == ProcessingMode.LOCAL) {
                val engineStatus by vm.engineStatus.collectAsState()
                NavRow(Icons.Default.Speed, "Performance", engineStatus + " · ${CpuInfo.chipName}") { onPage(SettingsPage.PERFORMANCE) }
                NavRow(Icons.Default.Download, "Models", "$installed installed · ${mb(vm.models.totalBytes)} on device") { onPage(SettingsPage.MODELS) }
            }
            NavRow(Icons.Default.BugReport, "Diagnostics & testing", if (vm.internalAvailable) "Crash reports · internal testing mode" else "Crash reports") { onPage(SettingsPage.DIAGNOSTICS) }
        }
        Note(
            if (s.processingMode == ProcessingMode.LOCAL) "Everything runs offline on this phone: ONNX Runtime (CPU · XNNPACK · NNAPI · Qualcomm QNN NPU) and LiteRT GPU."
            else "Processing runs on your server – see Processing above for connection status.",
            Modifier.padding(top = 14.dp),
        )
    }
}

// ============================================================================================================ swapping

@Composable
private fun SwapSettings(vm: AppViewModel) {
    val s by vm.settings.collectAsState()
    val m = vm.models
    fun up(t: (AppSettings) -> AppSettings) = vm.updateSettings(t)

    SectionTitle("Quality preset")
    Panel {
        Segmented(QualityPreset.entries.map { it to it.label }, s.preset) { p -> up { it.copy(preset = p) } }
        Note(s.preset.hint)
        ToggleRow("Instant draft preview", s.draftPreview, "Shows a native-resolution swap in about a second while High/Ultra and the enhancer finish") { v -> up { it.copy(draftPreview = v) } }
    }

    SectionTitle("Face swapper")
    Panel {
        ChipRow(ModelIds.SWAPPERS.filter { vm.modelAvailable(it) }.map { it to vm.swapperLabel(it) }, { s.swapperId == it }) { id -> up { it.copy(swapperId = id) } }
        Note(
            if (s.processingMode == com.faceswap.studio.data.ProcessingMode.REMOTE) "HyperSwap 1a/1b/1c: 256 px with its own face mask. InSwapper isn't offered over a remote server."
            else "HyperSwap 1a/1b/1c: 256 px with its own face mask. InSwapper: 128 px, strongest likeness – pair it with High/Ultra pixel boost.",
        )
    }

    SectionTitle("Face enhancer")
    Panel {
        ChipRow(listOf("auto" to "Auto (GPEN-256)", "none" to "Off") + ModelIds.ENHANCERS.filter { vm.modelAvailable(it) }.map { it to vm.modelTitle(it) }, { s.enhancer == it }) { id -> up { it.copy(enhancer = id) } }
        SliderRow("Blend", s.enhanceBlend, 0f..1f) { v -> up { it.copy(enhanceBlend = v) } }
        if (s.enhancer == ModelIds.CODEFORMER) SliderRow("CodeFormer fidelity", s.codeformerFidelity, 0f..1f) { v -> up { it.copy(codeformerFidelity = v) } }
    }

    SectionTitle("Expression restorer")
    Panel {
        if (vm.modelAvailable(ModelIds.LP_GENERATOR) && vm.modelAvailable(ModelIds.LP_MOTION) && vm.modelAvailable(ModelIds.LP_FEATURE)) {
            SliderRow("Restore target expression", s.expressionRestore, 0f..1f) { v -> up { it.copy(expressionRestore = v) } }
            Note("LivePortrait re-applies the target's mouth and eyes after swapping. 0 = off.")
        } else Note("Install the LivePortrait models (Models → Studio tools) to restore the target's original expression after swapping.")
    }

    SectionTitle("Detection")
    Panel {
        ChipRow(listOf(ModelIds.YUNET to "YuNet (fast)", ModelIds.SCRFD to "SCRFD").filter { vm.modelAvailable(it.first) }, { s.detectorId == it }) { id -> up { it.copy(detectorId = id) } }
        Spacer(Modifier.height(6.dp))
        ToggleRow("68-point landmarks", s.landmarks68, "HRFFA refines alignment – sharper, steadier swaps") { v -> up { it.copy(landmarks68 = v) } }
        SliderRow("Per-face match tolerance", s.referenceThreshold, 0.2f..0.6f) { v -> up { it.copy(referenceThreshold = v) } }
        ToggleRow("Debug overlay", s.debugOverlay, "Draws boxes and landmarks on results") { v -> up { it.copy(debugOverlay = v) } }
    }
}

// ============================================================================================================ masks

@Composable
private fun MaskSettings(vm: AppViewModel) {
    val s by vm.settings.collectAsState()
    fun up(t: (AppSettings) -> AppSettings) = vm.updateSettings(t)
    SectionTitle("Mask layers (combined)")
    Panel {
        val maskOptions = listOf(MaskType.BOX to "Box", MaskType.MODEL to "Swapper mask", MaskType.REGION to "Face parser", MaskType.AREA to "Landmark area", MaskType.ELLIPSE to "Oval")
        ChipRow(maskOptions, { it in s.maskTypes }) { t -> up { it.copy(maskTypes = if (t in it.maskTypes) it.maskTypes - t else it.maskTypes + t) } }
        Note("Layers are intersected like FaceFusion. Face parser keeps hair, hands and glasses from the target.")
        if (MaskType.REGION in s.maskTypes) {
            Spacer(Modifier.height(8.dp))
            Text("Parser regions", style = MaterialTheme.typography.bodyMedium)
            ChipRow(FaceRegion.entries.map { it to it.name.lowercase().replace("_", " ") }, { it in s.maskRegions }) { r -> up { it.copy(maskRegions = if (r in it.maskRegions) it.maskRegions - r else it.maskRegions + r) } }
            Note("Unselected parts keep the original – e.g. untick glasses to keep the target's glasses.")
        }
    }
    SectionTitle("Blending")
    Panel {
        SliderRow("Edge feather", s.maskBlur, 0.05f..0.8f) { v -> up { it.copy(maskBlur = v) } }
        SliderRow("Trim top (forehead)", s.maskPaddingTop, 0f..0.3f) { v -> up { it.copy(maskPaddingTop = v) } }
        SliderRow("Trim bottom (chin)", s.maskPaddingBottom, 0f..0.3f) { v -> up { it.copy(maskPaddingBottom = v) } }
        SliderRow("Swap strength", s.swapOpacity, 0.3f..1f) { v -> up { it.copy(swapOpacity = v) } }
        SliderRow("Colour match to target", s.colorMatch, 0f..1f) { v -> up { it.copy(colorMatch = v) } }
        SliderRow("Sharpen", s.sharpen, 0f..1.5f) { v -> up { it.copy(sharpen = v) } }
    }
}

// ============================================================================================================ performance

@Composable
private fun PerformanceSettings(vm: AppViewModel) {
    val s by vm.settings.collectAsState()
    val rows by vm.perfRows.collectAsState()
    val ctx = LocalContext.current
    fun up(t: (AppSettings) -> AppSettings) = vm.updateSettings(t)
    LaunchedEffect(Unit) { vm.refreshPerf() }

    lateinit var launchTuner: () -> Unit
    val tuner = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        if (r.resultCode == TunerActivity.RESULT_USER_CANCELLED) { vm.onTunerFinished(false); return@rememberLauncherForActivityResult }
        if (vm.onTunerFinished(r.resultCode == android.app.Activity.RESULT_OK) && vm.beginTuning(fresh = false)) launchTuner()
    }
    launchTuner = { tuner.launch(Intent(ctx, TunerActivity::class.java)) }

    SectionTitle("This device")
    Panel {
        Text(CpuInfo.chipName, style = MaterialTheme.typography.titleSmall)
        Spacer(Modifier.height(6.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Pill("${CpuInfo.bigCores} big cores")
            Pill(if (vm.adpfSupported) "ADPF on" else "ADPF n/a")
            if (CpuInfo.isSnapdragon) Pill("QNN " + if (vm.qnnAvailable) "ready" else "missing", content = if (vm.qnnAvailable) Good else Warn)
            if (vm.rootAvailable) Pill("root", content = Good)
        }
        Spacer(Modifier.height(12.dp))
        val tunedState by vm.tuned.collectAsState()
        val tuned = tunedState.first
        Button(onClick = { if (vm.beginTuning(fresh = true)) launchTuner() }, modifier = Modifier.fillMaxWidth().height(48.dp)) {
            Icon(Icons.Default.Tune, null); Spacer(Modifier.size(8.dp)); Text("Tune models in use (${vm.modelsInUse().size})")
        }
        OutlinedButton(onClick = { if (vm.beginTuning(fresh = true, all = true)) launchTuner() }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp)) {
            Text("Tune all installed models (slow)")
        }
        Note(if (tuned) "Auto uses the fastest backend verified on a real face for each model." + (if (!tunedState.second) " Last run was incomplete." else "")
        else "Not tuned yet – Auto runs every model on CPU until you tune. Tuning the models in use takes a few minutes; re-run it after switching swapper or enhancer.")
    }

    SectionTitle("Backend")
    Panel {
        ChipRow(Accelerator.entries.map { it to it.label }, { s.accelerator == it }) { a -> up { it.copy(accelerator = a) } }
        Note(s.accelerator.hint + if (s.accelerator != Accelerator.AUTO) " – forced backends skip the correctness check; switch back to Auto if results look wrong." else "")
        SliderRow("CPU threads", s.threads.toFloat(), 1f..8f, steps = 6, fmt = { it.toInt().toString() }) { v -> up { it.copy(threads = v.toInt()) } }
        SliderRow("Video render workers", s.videoWorkers.toFloat(), 1f..4f, steps = 2, fmt = { it.toInt().toString() }) { v -> up { it.copy(videoWorkers = v.toInt()) } }
        Note("Workers render frames in parallel while the next frames are decoded and detected. 2 suits NPU/GPU backends, 3–4 helps on CPU.")
        if (vm.rootAvailable) ToggleRow("Root performance boost", s.rootBoost, "During video: CPU governors + memory clocks pinned high, restored afterwards. Thermal limits are never touched.") { v -> up { it.copy(rootBoost = v) } }
    }

    SectionTitle("Models · backend · speed")
    Panel(padding = 10.dp) {
        if (rows.isEmpty()) Note("No models installed.")
        rows.forEachIndexed { i, r ->
            if (i > 0) HorizontalDivider(Modifier.padding(vertical = 6.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(r.title, style = MaterialTheme.typography.bodyMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(r.backend + (r.loadedOn?.let { " · loaded: $it" } ?: ""), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                r.twins.forEach { Pill(it, Modifier.padding(start = 4.dp)) }
                Column(Modifier.width(76.dp), horizontalAlignment = Alignment.End) {
                    Text(r.ms?.let { if (it >= 100) "%.0f ms".format(it) else "%.1f ms".format(it) } ?: "–", style = MaterialTheme.typography.labelLarge)
                    r.speedup?.takeIf { it > 1.05 }?.let { Text("%.1f× CPU".format(it), style = MaterialTheme.typography.labelSmall, color = Good) }
                }
            }
        }
    }
}

// ============================================================================================================ diagnostics

@Composable
private fun DiagnosticsSettings(vm: AppViewModel) {
    val unlocked by vm.internalUnlocked.collectAsState()
    SectionTitle("Crash reports")
    Panel {
        Note("Reports stay on the device. If the app closes unexpectedly, the report opens automatically on the next launch.")
        OutlinedButton(onClick = vm::showLastReport, modifier = Modifier.padding(top = 8.dp)) { Text("View last crash report") }
    }
    if (vm.internalAvailable) {
        SectionTitle("Internal testing (debug build only)")
        Panel {
            if (unlocked) {
                Text("Unlocked", style = MaterialTheme.typography.titleSmall, color = Good)
                Note("Explicit targets accept faces captured with the in-app camera (LIVE in the library). Locks again when the app restarts.")
                OutlinedButton(onClick = vm::lockInternal, modifier = Modifier.padding(top = 8.dp)) { Text("Lock") }
            } else {
                var pass by remember { mutableStateOf("") }
                OutlinedTextField(pass, { pass = it }, label = { Text("Passphrase") }, singleLine = true,
                    visualTransformation = PasswordVisualTransformation(), keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    modifier = Modifier.fillMaxWidth())
                Button(onClick = { if (vm.unlockInternal(pass)) pass = "" }, modifier = Modifier.padding(top = 8.dp)) { Text("Unlock") }
            }
        }
    }
}

internal fun mb(b: Long) = if (b >= 1_000_000_000) "%.2f GB".format(b / 1e9) else if (b >= 1_000_000) "%.0f MB".format(b / 1e6) else "%.0f KB".format(b / 1e3)

private fun processingSubtitle(mode: ProcessingMode, status: ServerStatus): String = when (mode) {
    ProcessingMode.LOCAL -> "On this device"
    ProcessingMode.REMOTE -> "Remote server · " + when (status) {
        is ServerStatus.Connected -> "connected (${status.info.models.size} models)"
        is ServerStatus.Connecting -> "connecting…"
        is ServerStatus.Failed -> "not connected"
        is ServerStatus.Disconnected -> "not connected"
    }
}
