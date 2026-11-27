package com.faceswap.studio.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.faceswap.core.FaceSelector
import com.faceswap.core.ModelIds
import com.faceswap.studio.ui.theme.Scrim

@Composable
fun VideoScreen(vm: AppViewModel, adders: FaceAdders, nav: Navigator) {
    val ctx = LocalContext.current
    val st by vm.video.collectAsState()
    val lib by vm.library.faces.collectAsState()
    val settings by vm.settings.collectAsState()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let(vm::setVideoTarget) }
    fun choose() = pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))

    Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp)) {
        ScreenHeader("Video", "Frame-by-frame swap with tracking, audio kept") {
            val engineStatus by vm.engineStatus.collectAsState()
            Pill(engineStatus, onClick = { nav.toSettings(SettingsPage.PERFORMANCE) })
        }

        val img = st.progress?.preview ?: st.preview
        if (img == null) {
            EmptyCanvas(Icons.Default.Movie, "Choose a video", "MP4 / MOV from your gallery", onClick = { if (!st.running) choose() })
        } else {
            Box {
                CompareCanvas(img, null) {
                    st.progress?.takeIf { st.running }?.let { p ->
                        Pill("${p.frame}/${p.totalFrames} · ${"%.1f".format(p.fps)} fps", Modifier.align(Alignment.TopStart).padding(10.dp), color = Scrim, content = Color.White)
                    }
                    if (st.durationMs > 0 && !st.running) Pill(formatDuration(st.durationMs / 1000), Modifier.align(Alignment.TopEnd).padding(10.dp), color = Scrim, content = Color.White)
                }
            }
            if (!st.running) OutlinedButton(onClick = ::choose, modifier = Modifier.padding(top = 8.dp)) {
                Icon(Icons.Default.Refresh, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text("Change")
            }
        }

        SectionTitle("Source face")
        SourcePicker(lib, st.setup.quickSources, { vm.toggleQuickSource(true, it) }, adders.gallery, adders.camera)

        TargetFacesPanel(st.setup.targetFaces, lib, st.setup.assignments, { i, id -> vm.assign(true, i, id) }, { vm.clearAssignments(true) })
        if (!st.setup.perFace && st.setup.targetFaces.size > 1) {
            SectionTitle("Apply to")
            ChipRow(listOf(FaceSelector.ALL to "All faces", FaceSelector.LARGEST to "Largest", FaceSelector.CENTER to "Center"), { settings.selector == it }) { sel -> vm.updateSettings { it.copy(selector = sel) } }
        }

        SectionTitle("Output")
        Panel {
            Text("Resolution (longest side)", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(6.dp))
            Segmented(listOf(720 to "720", 1080 to "1080", 1280 to "1280", 1920 to "1920"), settings.videoMaxSide) { s -> vm.updateSettings { it.copy(videoMaxSide = s) } }
            Spacer(Modifier.height(8.dp))
            ToggleRow("Face enhancer", settings.videoEnhance, "GPEN per face – sharper, several times slower") { v -> vm.updateSettings { it.copy(videoEnhance = v) } }
            if (settings.videoEnhance) ToggleRow("Smart enhance", settings.smartEnhance, "Skip faces that are already as sharp as the original") { v -> vm.updateSettings { it.copy(smartEnhance = v) } }
            ToggleRow("Keep original audio", settings.keepAudio) { v -> vm.updateSettings { it.copy(keepAudio = v) } }
            Note("Swapper: ${vm.swapperLabel(settings.swapperId)} at native resolution · ${settings.videoWorkers} render workers" + if (settings.rootBoost && vm.rootAvailable) " · root boost" else "")
            if (settings.swapperId in ModelIds.DIFFUSION_SWAPPERS) {
                Note("${vm.swapperLabel(settings.swapperId)} is a diffusion model for single photos — running it per frame on a video is very slow and will flicker. Pick a different swapper in Settings.", color = MaterialTheme.colorScheme.error)
            }
        }

        Spacer(Modifier.height(16.dp))
        if (st.running) {
            val p = st.progress
            Panel {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(if (p == null) (st.status ?: "Preparing…") else "Frame ${p.frame} of ${p.totalFrames}", Modifier.weight(1f), style = MaterialTheme.typography.titleSmall)
                    p?.let { Text("${(it.fraction * 100).toInt()}%", style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary) }
                }
                LinearProgressIndicator(progress = { p?.fraction ?: 0f }, modifier = Modifier.fillMaxWidth().padding(vertical = 10.dp))
                p?.let {
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Pill("${"%.1f".format(it.fps)} fps"); Pill("~${formatDuration(it.etaSeconds.toLong())} left"); Pill("${it.facesSwapped} faces swapped")
                    }
                }
                OutlinedButton(onClick = vm::cancelVideo, modifier = Modifier.padding(top = 10.dp)) { Icon(Icons.Default.Stop, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text("Cancel") }
            }
        } else {
            Button(
                onClick = vm::startVideo, enabled = st.uri != null && st.setup.ready,
                modifier = Modifier.fillMaxWidth().height(56.dp), shape = MaterialTheme.shapes.large,
                colors = ButtonDefaults.buttonColors(),
            ) { Icon(Icons.Default.PlayArrow, null); Spacer(Modifier.size(8.dp)); Text("Process video", style = MaterialTheme.typography.titleMedium) }
            if (st.uri != null && !st.setup.ready) Note("Pick a source face above (or assign faces individually).")
        }
        st.status?.takeIf { !st.running }?.let { Text(it, Modifier.padding(top = 10.dp), style = MaterialTheme.typography.bodyMedium) }
        if (st.output != null && !st.running) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 12.dp)) {
                FilledTonalButton(onClick = vm::saveVideo, modifier = Modifier.weight(1f).height(48.dp)) { Icon(Icons.Default.Save, null); Spacer(Modifier.size(6.dp)); Text(if (st.savedUri != null) "Saved" else "Save") }
                OutlinedButton(onClick = { share(ctx, vm.videoShareUri(), "video/mp4") }, modifier = Modifier.weight(1f).height(48.dp)) { Icon(Icons.Default.Share, null); Spacer(Modifier.size(6.dp)); Text("Share") }
            }
        }
        Note("Keep the app open while processing – the screen stays on automatically.", Modifier.padding(top = 12.dp))
        Spacer(Modifier.height(24.dp))
    }
}

fun formatDuration(seconds: Long): String = if (seconds >= 3600) "%d:%02d:%02d".format(seconds / 3600, seconds / 60 % 60, seconds % 60) else "%d:%02d".format(seconds / 60, seconds % 60)
