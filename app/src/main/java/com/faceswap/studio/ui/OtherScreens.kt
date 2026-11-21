package com.faceswap.studio.ui

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.faceswap.core.FaceSelector
import com.faceswap.core.MaskShape
import com.faceswap.studio.data.Accelerator
import com.faceswap.studio.data.LibraryFace
import com.faceswap.studio.data.ModelCatalog
import com.faceswap.studio.data.ModelState
import com.faceswap.studio.data.QualityPreset
import java.text.DateFormat
import java.util.Date

// ===================================================================== VIDEO
@Composable
fun VideoScreen(vm: AppViewModel, adders: FaceAdders) {
    val ctx = LocalContext.current
    val st by vm.video.collectAsState()
    val lib by vm.library.faces.collectAsState()
    val settings by vm.settings.collectAsState()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let(vm::setVideoTarget) }

    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("Video swap", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        SectionTitle("1 · Source face")
        SourcePicker(lib, st.setup.quickSources, { vm.toggleQuickSource(true, it) }, adders.gallery, adders.camera)

        SectionTitle("2 · Target video")
        val img = st.progress?.preview ?: st.preview
        if (img == null) {
            Box(
                Modifier.fillMaxWidth().height(200.dp).clip(RoundedCornerShape(14.dp)).background(MaterialTheme.colorScheme.surfaceVariant)
                    .clickable(enabled = !st.running) { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) },
                contentAlignment = Alignment.Center,
            ) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Icon(Icons.Default.Movie, null, Modifier.size(40.dp)); Text("Choose a video") } }
        } else {
            BitmapView(img)
            if (!st.running) OutlinedButton(onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) }, modifier = Modifier.padding(top = 8.dp)) { Text("Change") }
            if (st.durationMs > 0) Text("Length ${st.durationMs / 1000}s", style = MaterialTheme.typography.bodySmall)
        }

        TargetFacesPanel(st.setup.targetFaces, lib, st.setup.assignments, { i, id -> vm.assign(true, i, id) }, { vm.clearAssignments(true) })

        SectionTitle("Video options")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(720, 1080, 1280, 1920).forEach { s -> FilterChip(settings.videoMaxSide == s, { vm.updateSettings { it.copy(videoMaxSide = s) } }, { Text("${s}p") }) }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Quality: ${settings.preset.label}", Modifier.weight(1f))
        }
        ToggleRow("Face enhancer (much slower)", settings.videoEnhance) { v -> vm.updateSettings { it.copy(videoEnhance = v) } }
        ToggleRow("Keep original audio", settings.keepAudio) { v -> vm.updateSettings { it.copy(keepAudio = v) } }

        Spacer(Modifier.height(12.dp))
        if (st.running) {
            val p = st.progress
            LinearProgressIndicator(progress = { p?.fraction ?: 0f }, modifier = Modifier.fillMaxWidth())
            Text(
                if (p == null) "Preparing…" else "Frame ${p.frame}/${p.totalFrames} · ${"%.1f".format(p.fps)} fps · ~${p.etaSeconds / 60}m ${p.etaSeconds % 60}s left",
                Modifier.padding(top = 6.dp), style = MaterialTheme.typography.bodySmall,
            )
            OutlinedButton(onClick = vm::cancelVideo, modifier = Modifier.padding(top = 8.dp)) { Text("Cancel") }
        } else {
            Button(onClick = vm::startVideo, enabled = st.uri != null && st.setup.ready, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                Icon(Icons.Default.PlayArrow, null); Spacer(Modifier.size(8.dp)); Text("Process video")
            }
        }
        st.status?.let { Text(it, Modifier.padding(top = 8.dp)) }
        if (st.output != null && !st.running) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 10.dp)) {
                Button(onClick = vm::saveVideo, modifier = Modifier.weight(1f)) { Icon(Icons.Default.Save, null); Spacer(Modifier.size(6.dp)); Text("Save") }
                OutlinedButton(onClick = {
                    vm.videoShareUri()?.let { uri ->
                        ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                            type = "video/mp4"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }, "Share"))
                    }
                }, modifier = Modifier.weight(1f)) { Icon(Icons.Default.Share, null); Spacer(Modifier.size(6.dp)); Text("Share") }
            }
        }
        Text("Keep the app open while processing; the screen stays on automatically.", Modifier.padding(top = 12.dp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// ===================================================================== LIBRARY
@Composable
fun LibraryScreen(vm: AppViewModel, adders: FaceAdders) {
    val lib by vm.library.faces.collectAsState()
    var editing by remember { mutableStateOf<LibraryFace?>(null) }
    Column(Modifier.padding(16.dp)) {
        Text("Face library", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text("Saved faces are reused across photo and video swaps. Camera captures are stored as PNG with live metadata.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(vertical = 12.dp)) {
            Button(onClick = adders.camera, modifier = Modifier.weight(1f)) { Icon(Icons.Default.CameraAlt, null); Spacer(Modifier.size(6.dp)); Text("Live capture") }
            OutlinedButton(onClick = adders.gallery, modifier = Modifier.weight(1f)) { Icon(Icons.Default.Add, null); Spacer(Modifier.size(6.dp)); Text("From photo") }
        }
        if (lib.isEmpty()) Text("No faces yet.", Modifier.padding(top = 24.dp))
        LazyVerticalGrid(GridCells.Adaptive(150.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(lib, key = { it.id }) { f ->
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                    Column(Modifier.padding(10.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Box {
                            val img = remember(f.thumbFile) { loadThumb(f.thumbFile) }
                            if (img != null) Image(img, null, Modifier.size(110.dp).clip(CircleShape), contentScale = ContentScale.Crop)
                            if (f.isLive) LiveBadge(Modifier.align(Alignment.BottomCenter))
                        }
                        Text(f.label, fontWeight = FontWeight.SemiBold, modifier = Modifier.padding(top = 6.dp))
                        Text(DateFormat.getDateInstance(DateFormat.SHORT).format(Date(f.createdAt)) + if (f.isLive) " · camera" else " · photo", style = MaterialTheme.typography.bodySmall)
                        Row {
                            IconButton(onClick = { editing = f }) { Icon(Icons.Default.Edit, "Rename") }
                            IconButton(onClick = { vm.deleteFace(f.id) }) { Icon(Icons.Default.Delete, "Delete") }
                        }
                    }
                }
            }
        }
    }
    editing?.let { f ->
        var name by remember(f.id) { mutableStateOf(f.label) }
        AlertDialog(
            onDismissRequest = { editing = null },
            title = { Text("Rename face") },
            text = { OutlinedTextField(name, { name = it }, singleLine = true) },
            confirmButton = { TextButton(onClick = { vm.renameFace(f.id, name.trim().ifEmpty { f.label }); editing = null }) { Text("Save") } },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("Cancel") } },
        )
    }
}

// ===================================================================== MODELS
@Composable
fun ModelsScreen(vm: AppViewModel, gate: Boolean) {
    val states by vm.models.states.collectAsState()
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { if (it.isNotEmpty()) vm.importModels(it) }
    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text(if (gate) "Set up FaceSwap Studio" else "Models", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text(
            "Everything runs on your phone. Models download once from Hugging Face and are verified by checksum; no images ever leave the device.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        val missing = ModelCatalog.all.filter { it.required && states[it.id] !is ModelState.Ready }
        if (states.values.any { it is ModelState.Installing }) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 12.dp)) {
                androidx.compose.material3.CircularProgressIndicator(Modifier.size(22.dp)); Spacer(Modifier.size(10.dp))
                Text("Installing bundled models… (one time, about a minute)")
            }
        } else if (missing.isNotEmpty()) {
            Button(onClick = vm::downloadRequired, modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp).height(52.dp)) {
                Text("Download required models (${mb(missing.sumOf { it.sizeBytes })})")
            }
        }
        ModelCatalog.all.forEach { spec ->
            val s = states[spec.id] ?: ModelState.Missing
            Card(Modifier.fillMaxWidth().padding(vertical = 5.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(spec.title + if (spec.required) "" else "  (optional)", fontWeight = FontWeight.SemiBold)
                            Text("${spec.description} · ${mb(spec.sizeBytes)}", style = MaterialTheme.typography.bodySmall)
                            Text("License: ${spec.license}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        when (s) {
                            is ModelState.Ready -> TextButton(onClick = { vm.deleteModel(spec) }) { Text("Delete") }
                            is ModelState.Downloading -> TextButton(onClick = { vm.cancelDownload(spec) }) { Text("Pause") }
                            is ModelState.Verifying -> Text("Verifying…", style = MaterialTheme.typography.bodySmall)
                            is ModelState.Installing -> Text("Installing…", style = MaterialTheme.typography.bodySmall)
                            else -> TextButton(onClick = { vm.download(spec) }) { Text(if (s is ModelState.Failed) "Retry" else "Get") }
                        }
                    }
                    when (s) {
                        is ModelState.Downloading -> {
                            LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
                            Text("${mb(s.bytes)} / ${mb(s.total)}", style = MaterialTheme.typography.labelSmall)
                        }
                        is ModelState.Failed -> Text(s.message, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                        is ModelState.Ready -> Text("Ready", color = MaterialTheme.colorScheme.secondary, style = MaterialTheme.typography.labelSmall)
                        else -> Unit
                    }
                }
            }
        }
        OutlinedButton(onClick = { importer.launch(arrayOf("*/*")) }, modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
            Text("Import model files from storage (offline)")
        }
        Text("Downloads resume if interrupted. For offline devices, copy the .onnx files over USB and import them, or use a build with bundled models.", Modifier.padding(top = 8.dp), style = MaterialTheme.typography.bodySmall)
    }
}

private fun mb(b: Long) = if (b >= 1_000_000_000) "%.2f GB".format(b / 1e9) else "%.0f MB".format(b / 1e6)

// ===================================================================== SETTINGS
@Composable
fun ToggleRow(label: String, value: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(1f)); Switch(value, onChange)
    }
}

@Composable
private fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, fmt: (Float) -> String = { "%.2f".format(it) }, onChange: (Float) -> Unit) {
    Column(Modifier.padding(vertical = 2.dp)) {
        Row { Text(label, Modifier.weight(1f)); Text(fmt(value), color = MaterialTheme.colorScheme.onSurfaceVariant) }
        Slider(value, onChange, valueRange = range)
    }
}

@Composable
fun SettingsScreen(vm: AppViewModel) {
    val s by vm.settings.collectAsState()
    val unlocked by vm.internalUnlocked.collectAsState()
    fun up(t: (com.faceswap.studio.data.AppSettings) -> com.faceswap.studio.data.AppSettings) = vm.updateSettings(t)

    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        SectionTitle("Quality preset")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) { QualityPreset.entries.forEach { p -> FilterChip(s.preset == p, { up { it.copy(preset = p) } }, { Text(p.label) }) } }
        Text("Fast = native 128px · Balanced = 2× pixel boost · High = boost + GFPGAN · Ultra = 4× boost + GFPGAN", style = MaterialTheme.typography.bodySmall)

        SectionTitle("Blending")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(MaskShape.BOX to "Box", MaskShape.ELLIPSE to "Oval", MaskShape.BOX_AND_ELLIPSE to "Box ∩ Oval").forEach { (m, l) -> FilterChip(s.maskShape == m, { up { it.copy(maskShape = m) } }, { Text(l) }) }
        }
        SliderRow("Edge feather", s.maskBlur, 0.05f..0.8f) { v -> up { it.copy(maskBlur = v) } }
        SliderRow("Trim top (forehead)", s.maskPaddingTop, 0f..0.3f) { v -> up { it.copy(maskPaddingTop = v) } }
        SliderRow("Trim bottom (chin)", s.maskPaddingBottom, 0f..0.3f) { v -> up { it.copy(maskPaddingBottom = v) } }
        SliderRow("Swap strength", s.swapOpacity, 0.3f..1f) { v -> up { it.copy(swapOpacity = v) } }
        SliderRow("Colour match to target", s.colorMatch, 0f..1f) { v -> up { it.copy(colorMatch = v) } }
        SliderRow("Sharpen", s.sharpen, 0f..1.5f) { v -> up { it.copy(sharpen = v) } }
        SliderRow("Enhancer blend", s.enhanceBlend, 0f..1f) { v -> up { it.copy(enhanceBlend = v) } }

        SectionTitle("Detection")
        SliderRow("Per-face match tolerance", s.referenceThreshold, 0.2f..0.6f) { v -> up { it.copy(referenceThreshold = v) } }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(320 to "Fast", 480 to "Normal", 640 to "Accurate").forEach { (sz, l) -> FilterChip(s.detectorSize == sz, { up { it.copy(detectorSize = sz) } }, { Text("$l ($sz)") }) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(FaceSelector.ALL to "All faces", FaceSelector.LARGEST to "Largest", FaceSelector.CENTER to "Center").forEach { (sel, l) -> FilterChip(s.selector == sel, { up { it.copy(selector = sel) } }, { Text(l) }) }
        }

        SectionTitle("Performance")
        val ctx = LocalContext.current
        val tuneSummary by vm.tuningSummary.collectAsState()
        lateinit var launchTuner: () -> Unit
        val tuner = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
            if (r.resultCode == com.faceswap.studio.engine.TunerActivity.RESULT_USER_CANCELLED) { vm.onTunerFinished(false); return@rememberLauncherForActivityResult }
            if (vm.onTunerFinished(r.resultCode == android.app.Activity.RESULT_OK) && vm.beginTuning(fresh = false)) launchTuner()
        }
        launchTuner = { tuner.launch(Intent(ctx, com.faceswap.studio.engine.TunerActivity::class.java)) }
        Button(onClick = { if (vm.beginTuning(fresh = true)) launchTuner() }, modifier = Modifier.fillMaxWidth()) { Text(if (tuneSummary == null) "Tune for this device" else "Re-run device tuning") }
        Text(tuneSummary ?: "Not tuned yet – Auto uses CPU for every model until you run tuning (takes 2–5 minutes).",
            Modifier.padding(vertical = 6.dp), style = MaterialTheme.typography.bodySmall)
        Text("Big cores detected: ${com.faceswap.studio.engine.CpuInfo.bigCores} · SoC: ${com.faceswap.studio.engine.socModel()}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) { Accelerator.entries.forEach { a -> FilterChip(s.accelerator == a, { up { it.copy(accelerator = a) } }, { Text(a.label) }) } }
        SliderRow("CPU threads", s.threads.toFloat(), 1f..8f, { it.toInt().toString() }) { v -> up { it.copy(threads = v.toInt()) } }
        Text("Auto uses the fastest backend the tuner verified for each model. Forcing XNNPACK/NNAPI skips that safety check – switch back to Auto or CPU if the app closes.", style = MaterialTheme.typography.bodySmall)

        SectionTitle("Diagnostics")
        Text("Crash reports are kept on the device only. If the app closes unexpectedly, the report opens automatically on next launch.", style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = vm::showLastReport, modifier = Modifier.padding(top = 6.dp)) { Text("View last crash report") }

        if (vm.internalAvailable) {
            HorizontalDivider(Modifier.padding(vertical = 16.dp))
            SectionTitle("Internal testing (debug build only)")
            if (unlocked) {
                Text("Unlocked. Explicit targets accept faces captured with the in-app camera (LIVE in the library).", color = MaterialTheme.colorScheme.secondary)
                OutlinedButton(onClick = vm::lockInternal, modifier = Modifier.padding(top = 8.dp)) { Text("Lock") }
            } else {
                var pass by remember { mutableStateOf("") }
                OutlinedTextField(pass, { pass = it }, label = { Text("Passphrase") }, singleLine = true,
                    visualTransformation = PasswordVisualTransformation(), keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    modifier = Modifier.fillMaxWidth())
                Button(onClick = { if (vm.unlockInternal(pass)) pass = "" }, modifier = Modifier.padding(top = 8.dp)) { Text("Unlock") }
            }
        }
        Spacer(Modifier.height(24.dp))
        Text("FaceSwap Studio · on-device · ONNX Runtime", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    }
}
