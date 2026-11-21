package com.faceswap.studio.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.faceswap.core.ModelIds
import com.faceswap.core.Processors
import com.faceswap.studio.util.Bitmaps

private enum class Tool(val label: String) { EDIT("Face editor"), AGE("Age"), BACKGROUND("Background"), UPSCALE("Upscale"), ENHANCE("Enhance") }

/** FaceFusion processors on the current photo (result or target). Each apply pushes an undo step. */
@Composable
fun StudioTools(vm: AppViewModel) {
    val st by vm.photo.collectAsState()
    if (st.target == null) return
    val m = vm.models
    var tool by remember { mutableStateOf(Tool.EDIT) }
    val ctx = LocalContext.current
    val pickBg = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { runCatching { Bitmaps.toRgb(Bitmaps.decode(ctx, it, 2048)) }.getOrNull()?.let { img -> vm.applyStudio(AppViewModel.StudioOp.Background(Processors.Background.Image(img))) } }
    }

    Card(Modifier.fillMaxWidth().padding(top = 14.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Studio tools", fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                if (st.history.isNotEmpty()) TextButton(onClick = vm::undoStudio) { Text("Undo") }
                if (st.result != null) TextButton(onClick = vm::resultAsTarget) { Text("Use as target") }
            }
            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                items(Tool.entries.size) { i -> val t = Tool.entries[i]; FilterChip(tool == t, { tool = t }, { Text(t.label) }) }
            }
            st.toolBusy?.let {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 10.dp)) {
                    CircularProgressIndicator(Modifier.size(20.dp)); Text("  $it")
                }
                return@Column
            }
            when (tool) {
                Tool.EDIT -> {
                    if (!(m.has(ModelIds.LP_FEATURE) && m.has(ModelIds.LP_MOTION) && m.has(ModelIds.LP_GENERATOR) && m.has(ModelIds.LP_STITCHER))) {
                        Missing("LivePortrait models (Models → Studio tools)")
                    } else {
                        var smile by remember { mutableFloatStateOf(0f) }; var eyebrow by remember { mutableFloatStateOf(0f) }
                        var eyeOpen by remember { mutableFloatStateOf(0f) }; var lipOpen by remember { mutableFloatStateOf(0f) }
                        var gazeH by remember { mutableFloatStateOf(0f) }; var gazeV by remember { mutableFloatStateOf(0f) }
                        var pout by remember { mutableFloatStateOf(0f) }; var grim by remember { mutableFloatStateOf(0f) }
                        var yaw by remember { mutableFloatStateOf(0f) }; var pitch by remember { mutableFloatStateOf(0f) }; var roll by remember { mutableFloatStateOf(0f) }
                        Slider1("Smile", smile) { smile = it }; Slider1("Eyebrows", eyebrow) { eyebrow = it }
                        Slider1("Eyes open", eyeOpen) { eyeOpen = it }; Slider1("Lips open", lipOpen) { lipOpen = it }
                        Slider1("Gaze ←→", gazeH) { gazeH = it }; Slider1("Gaze ↑↓", gazeV) { gazeV = it }
                        Slider1("Pout", pout) { pout = it }; Slider1("Grim", grim) { grim = it }
                        Slider1("Head turn", yaw) { yaw = it }; Slider1("Head tilt", pitch) { pitch = it }; Slider1("Head roll", roll) { roll = it }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = {
                                vm.applyStudio(AppViewModel.StudioOp.Edit(Processors.FaceEdit(
                                    eyebrowDirection = eyebrow, eyeGazeHorizontal = gazeH, eyeGazeVertical = gazeV, eyeOpen = eyeOpen,
                                    lipOpen = lipOpen, mouthGrim = grim, mouthPout = pout, mouthSmile = smile,
                                    headPitch = pitch, headYaw = yaw, headRoll = roll,
                                )))
                            }) { Text("Apply") }
                            OutlinedButton(onClick = { smile = 0f; eyebrow = 0f; eyeOpen = 0f; lipOpen = 0f; gazeH = 0f; gazeV = 0f; pout = 0f; grim = 0f; yaw = 0f; pitch = 0f; roll = 0f }) { Text("Reset") }
                        }
                    }
                }
                Tool.AGE -> {
                    if (!m.has(ModelIds.FRAN)) Missing("FRAN (Models → Studio tools)") else {
                        var current by remember { mutableFloatStateOf(30f) }; var target by remember { mutableFloatStateOf(60f) }
                        Text("Current age ${current.toInt()}"); Slider(current, { current = it }, valueRange = 10f..90f)
                        Text("Target age ${target.toInt()}"); Slider(target, { target = it }, valueRange = 5f..95f)
                        Button(onClick = { vm.applyStudio(AppViewModel.StudioOp.Age(current, target)) }) { Text("Apply") }
                        Note("Runs at 1024² per face – a few seconds on the NPU/GPU.")
                    }
                }
                Tool.BACKGROUND -> {
                    if (!m.has(ModelIds.MODNET) && !m.has(ModelIds.U2NETP)) Missing("MODNet") else {
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 6.dp)) {
                            OutlinedButton(onClick = { vm.applyStudio(AppViewModel.StudioOp.Background(Processors.Background.Blur(0.7f))) }) { Text("Blur") }
                            OutlinedButton(onClick = { vm.applyStudio(AppViewModel.StudioOp.Background(Processors.Background.Transparent)) }) { Text("Remove") }
                            OutlinedButton(onClick = { pickBg.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) { Text("Replace") }
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 6.dp)) {
                            listOf("White" to 0xFFFFFF, "Black" to 0x000000, "Green" to 0x00B140, "Studio grey" to 0x8A8D93).forEach { (n, c) ->
                                OutlinedButton(onClick = { vm.applyStudio(AppViewModel.StudioOp.Background(Processors.Background.Color(c))) }) { Text(n) }
                            }
                        }
                        Note("Remove keeps transparency – the photo is saved as PNG.")
                    }
                }
                Tool.UPSCALE -> {
                    val options = listOf(ModelIds.SPAN_X4 to "SPAN ×4 (fast)", ModelIds.ESRGAN_X2 to "Real-ESRGAN ×2", ModelIds.ESRGAN_X4 to "Real-ESRGAN ×4").filter { m.has(it.first) }
                    if (options.isEmpty()) Missing("an upscaler") else Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        options.forEach { (id, l) -> OutlinedButton(onClick = { vm.applyStudio(AppViewModel.StudioOp.Upscale(id)) }) { Text(l) } }
                    }
                    Note("Tiled – any resolution. Large photos take a while.")
                }
                Tool.ENHANCE -> {
                    val options = ModelIds.ENHANCERS.filter { m.has(it) }
                    if (options.isEmpty()) Missing("a face enhancer") else Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        options.forEach { id -> OutlinedButton(onClick = { vm.applyStudio(AppViewModel.StudioOp.Enhance(id)) }) { Text(id.substringBefore("_bfr").replace("_", " ")) } }
                    }
                }
            }
        }
    }
}

@Composable
private fun Slider1(label: String, v: Float, on: (Float) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.weight(0.35f), style = MaterialTheme.typography.bodySmall)
        Slider(v, on, valueRange = -1f..1f, modifier = Modifier.weight(0.65f))
    }
}

@Composable private fun Missing(what: String) = Text("Install $what to use this tool.", color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp))
@Composable private fun Note(t: String) = Text(t, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp))
