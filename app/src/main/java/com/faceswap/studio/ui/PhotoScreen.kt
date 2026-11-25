package com.faceswap.studio.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.SubdirectoryArrowRight
import androidx.compose.material3.Button
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.faceswap.core.FaceSelector
import com.faceswap.studio.data.QualityPreset

@Composable
fun PhotoScreen(vm: AppViewModel, adders: FaceAdders, nav: Navigator) {
    val ctx = LocalContext.current
    val st by vm.photo.collectAsState()
    val lib by vm.library.faces.collectAsState()
    val settings by vm.settings.collectAsState()
    val loading by vm.engineLoading.collectAsState()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let(vm::setPhotoTarget) }
    fun choose() = pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))

    Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp)) {
        ScreenHeader("Photo", "Swap faces in a picture – everything runs on this phone") {
            val engineStatus by vm.engineStatus.collectAsState()
            Pill(engineStatus, onClick = { nav.toSettings(SettingsPage.PERFORMANCE) })
        }

        val target = st.target
        if (target == null) {
            EmptyCanvas(Icons.Default.AddPhotoAlternate, "Choose a photo", "or share one to FaceSwap Studio", onClick = ::choose)
        } else {
            CompareCanvas(target, if (st.showOriginal) null else st.result) {
                BusyOverlay(
                    when { loading -> "Loading models…"; st.refining -> "Refining…"; st.busy -> st.status ?: "Working…"; st.toolBusy != null -> st.toolBusy; else -> null },
                    visible = st.busy || st.toolBusy != null,
                    dim = !st.refining,
                )
            }
            Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = ::choose, enabled = !st.busy) { Icon(Icons.Default.Refresh, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text("Change") }
                if (st.result != null) {
                    OutlinedButton(onClick = vm::resultAsTarget, enabled = !st.busy) { Icon(Icons.Default.SubdirectoryArrowRight, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text("Use as target") }
                }
                if (st.history.isNotEmpty()) OutlinedButton(onClick = vm::undoStudio) { Icon(Icons.AutoMirrored.Filled.Undo, null, Modifier.size(18.dp)) }
            }
            if (st.result != null) Note("Drag across the picture to compare, tap to flip between original and swapped.")
        }

        SectionTitle("Source face")
        SourcePicker(lib, st.setup.quickSources, { vm.toggleQuickSource(false, it) }, adders.gallery, adders.camera)

        TargetFacesPanel(st.setup.targetFaces, lib, st.setup.assignments, { i, id -> vm.assign(false, i, id) }, { vm.clearAssignments(false) })

        if (!st.setup.perFace && st.setup.targetFaces.size > 1) {
            SectionTitle("Apply to")
            ChipRow(listOf(FaceSelector.ALL to "All faces", FaceSelector.LARGEST to "Largest", FaceSelector.CENTER to "Center", FaceSelector.LEFTMOST to "Left", FaceSelector.RIGHTMOST to "Right"),
                { settings.selector == it }) { sel -> vm.updateSettings { it.copy(selector = sel) } }
        }

        SectionTitle("Quality")
        Segmented(QualityPreset.entries.map { it to it.label }, settings.preset) { p -> vm.updateSettings { it.copy(preset = p) } }
        Note(settings.preset.hint + " · " + vm.swapperLabel(settings.swapperId) + (settings.enhancerFor(false)?.let { " · " + vm.modelTitle(it) } ?: ""))
        val enh = settings.enhancerFor(false)
        if (enh != null && !vm.models.has(enh)) Note("${vm.modelTitle(enh)} isn't installed – results won't be enhanced.", color = MaterialTheme.colorScheme.error)
        if (!vm.models.has(settings.swapperId)) Note("${vm.modelTitle(settings.swapperId)} isn't installed – pick another swapper in Settings → Swapping.", color = MaterialTheme.colorScheme.error)

        Spacer(Modifier.height(16.dp))
        Button(
            onClick = vm::swapPhoto,
            enabled = target != null && st.setup.ready && !st.busy && st.toolBusy == null,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = MaterialTheme.shapes.large,
        ) {
            Icon(Icons.Default.AutoFixHigh, null); Spacer(Modifier.size(8.dp))
            Text(when { loading -> "Loading models…"; st.busy -> "Working…"; st.result != null -> "Swap again"; else -> "Swap faces" }, style = MaterialTheme.typography.titleMedium)
        }
        if (target != null && !st.setup.ready) Note("Pick a source face above (or assign faces individually).")
        st.status?.let { Text(it, Modifier.padding(top = 10.dp), style = MaterialTheme.typography.bodyMedium) }
        TimingChips(st.timings)

        if (st.result != null && !st.busy) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 12.dp)) {
                FilledTonalButton(onClick = vm::savePhoto, modifier = Modifier.weight(1f).height(48.dp)) { Icon(Icons.Default.Save, null); Spacer(Modifier.size(6.dp)); Text(if (st.savedUri != null) "Saved" else "Save") }
                OutlinedButton(onClick = { share(ctx, vm.photoShareUri(), if (vm.photoHasAlpha()) "image/png" else "image/jpeg") }, modifier = Modifier.weight(1f).height(48.dp)) {
                    Icon(Icons.Default.Share, null); Spacer(Modifier.size(6.dp)); Text("Share")
                }
            }
        }

        if (target != null) StudioTools(vm, nav)
        Spacer(Modifier.height(24.dp))
    }
}

/** Small helper so composables can align icons in a row. */
@Composable
internal fun IconText(icon: androidx.compose.ui.graphics.vector.ImageVector, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) { Icon(icon, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text(text) }
}
