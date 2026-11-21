package com.faceswap.studio.ui

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.faceswap.core.FaceSelector
import com.faceswap.studio.data.QualityPreset

@Composable
fun PhotoScreen(vm: AppViewModel, adders: FaceAdders) {
    val ctx = LocalContext.current
    val st by vm.photo.collectAsState()
    val lib by vm.library.faces.collectAsState()
    val settings by vm.settings.collectAsState()
    val loading by vm.engineLoading.collectAsState()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let(vm::setPhotoTarget) }

    Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("Photo swap", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        SectionTitle("1 · Source face")
        SourcePicker(lib, st.setup.quickSources, { vm.toggleQuickSource(false, it) }, adders.gallery, adders.camera)

        SectionTitle("2 · Target photo")
        val shown = if (st.showOriginal || st.result == null) st.target else st.result
        if (shown == null) {
            Box(
                Modifier.fillMaxWidth().height(220.dp).clip(RoundedCornerShape(14.dp)).background(MaterialTheme.colorScheme.surfaceVariant)
                    .clickable { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Default.AddPhotoAlternate, null, Modifier.size(40.dp))
                    Text("Choose a photo")
                }
            }
        } else {
            Box {
                BitmapView(shown, Modifier.clickable { if (st.result != null) vm.toggleOriginal() })
                if (st.result != null) Text(
                    if (st.showOriginal) "ORIGINAL" else "SWAPPED",
                    Modifier.align(Alignment.TopStart).padding(8.dp).background(Color(0x99000000), RoundedCornerShape(6.dp)).padding(horizontal = 6.dp, vertical = 2.dp),
                    color = Color.White, style = MaterialTheme.typography.labelSmall,
                )
                if (st.busy) CircularProgressIndicator(Modifier.align(Alignment.Center))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 8.dp)) {
                OutlinedButton(onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) { Text("Change") }
                if (st.result != null) {
                    OutlinedButton(onClick = vm::toggleOriginal) { Icon(Icons.Default.Compare, null); Spacer(Modifier.size(4.dp)); Text("Compare") }
                }
            }
        }

        TargetFacesPanel(st.setup.targetFaces, lib, st.setup.assignments, { i, id -> vm.assign(false, i, id) }, { vm.clearAssignments(false) })

        if (!st.setup.perFace && st.setup.targetFaces.size > 1) {
            SectionTitle("Apply to")
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(FaceSelector.ALL to "All", FaceSelector.LARGEST to "Largest", FaceSelector.LEFTMOST to "Left", FaceSelector.RIGHTMOST to "Right").forEach { (sel, l) ->
                    FilterChip(settings.selector == sel, { vm.updateSettings { it.copy(selector = sel) } }, { Text(l) })
                }
            }
        }

        SectionTitle("Quality")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            QualityPreset.entries.forEach { p -> FilterChip(settings.preset == p, { vm.updateSettings { it.copy(preset = p) } }, { Text(p.label) }) }
        }
        val enh = settings.enhancerFor(false)
        if (enh != null && !vm.models.has(enh))
            Text("Enhancer $enh isn't installed (Models) – results won't be enhanced.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        if (!vm.models.has(settings.swapperId))
            Text("Swapper ${settings.swapperId} isn't installed – pick another in Settings or install it in Models.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)

        Spacer(Modifier.height(14.dp))
        Button(
            onClick = vm::swapPhoto,
            enabled = st.target != null && st.setup.ready && !st.busy,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            Icon(Icons.Default.AutoFixHigh, null); Spacer(Modifier.size(8.dp))
            Text(if (loading) "Loading models…" else if (st.busy) "Working…" else "Swap faces")
        }
        st.status?.let { Text(it, Modifier.padding(top = 8.dp), style = MaterialTheme.typography.bodyMedium) }

        if (st.result != null) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 10.dp)) {
                Button(onClick = vm::savePhoto, modifier = Modifier.weight(1f)) { Icon(Icons.Default.Save, null); Spacer(Modifier.size(6.dp)); Text("Save") }
                OutlinedButton(onClick = {
                    vm.photoShareUri()?.let { uri ->
                        ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                            type = "image/jpeg"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }, "Share"))
                    }
                }, modifier = Modifier.weight(1f)) { Icon(Icons.Default.Share, null); Spacer(Modifier.size(6.dp)); Text("Share") }
            }
        }
        StudioTools(vm)
        Spacer(Modifier.height(24.dp))
    }
}
