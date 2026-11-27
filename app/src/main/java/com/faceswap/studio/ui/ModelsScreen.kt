package com.faceswap.studio.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.unit.dp
import com.faceswap.studio.data.ModelCatalog
import com.faceswap.studio.data.ModelGroup
import com.faceswap.studio.data.ModelSpec
import com.faceswap.studio.data.ModelState
import com.faceswap.studio.ui.theme.Good

@Composable
fun ModelsScreen(vm: AppViewModel, gate: Boolean, onBack: (() -> Unit)?) {
    val states by vm.models.states.collectAsState()
    val settings by vm.settings.collectAsState()
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { if (it.isNotEmpty()) vm.importModels(it) }
    var confirmDelete by remember { mutableStateOf<ModelSpec?>(null) }
    Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = if (onBack != null) 8.dp else 12.dp)) {
        if (onBack != null) Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Text("Models", style = MaterialTheme.typography.titleLarge)
        } else ScreenHeader(if (gate) "Set up FaceSwap Studio" else "Models")

        if (settings.processingMode == com.faceswap.studio.data.ProcessingMode.REMOTE) {
            Panel {
                Text("Remote mode is on", style = MaterialTheme.typography.titleSmall)
                Note("Models run on your server, not this list. Manage the connection in Settings → Processing. These downloads only matter if you switch back to on-device processing.")
            }
        }

        Panel {
            Text(if (gate) "One-time setup" else "Storage", style = MaterialTheme.typography.titleSmall)
            Note("Every model runs on this phone. Core models ship inside the APK, extras come as offline model packs (Import) or a one-time download from Hugging Face. GPU / NPU int8 badges mark accelerator twins built in CI.")
            Note("${mb(vm.models.totalBytes)} used on this device")
            val installing = states.values.any { it is ModelState.Installing }
            val missing = ModelCatalog.all.filter { it.required && states[it.id] !is ModelState.Ready }
            if (installing) Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 12.dp)) {
                CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp); Spacer(Modifier.size(10.dp)); Text("Installing bundled models… (one time)")
            } else if (missing.isNotEmpty()) Button(onClick = vm::downloadRequired, modifier = Modifier.fillMaxWidth().padding(top = 12.dp).height(50.dp)) {
                Text("Download required models (${mb(missing.sumOf { it.sizeBytes })})")
            }
            OutlinedButton(onClick = { importer.launch(arrayOf("*/*")) }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                Icon(Icons.Default.FolderOpen, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text("Import files or model pack .zip")
            }
        }

        ModelGroup.entries.forEach { group ->
            val specs = ModelCatalog.all.filter { it.group == group }
            val absent = specs.filter { states[it.id] !is ModelState.Ready }
            SectionTitle(group.title + " · ${specs.size - absent.size}/${specs.size}", action = {
                if (absent.isNotEmpty() && group != ModelGroup.REQUIRED) TextButton(onClick = { absent.forEach(vm::download) }) { Text("Get all · ${mb(absent.sumOf { it.sizeBytes })}") }
            })
            Panel(padding = 10.dp) {
                specs.forEachIndexed { i, spec ->
                    if (i > 0) HorizontalDivider(Modifier.padding(vertical = 4.dp), color = MaterialTheme.colorScheme.outlineVariant)
                    ModelRow(vm, spec, states[spec.id] ?: ModelState.Missing) { confirmDelete = spec }
                }
            }
        }
        Note("Downloads resume if interrupted. Deleting a model also removes its GPU / NPU twins and compiled NPU graphs.", Modifier.padding(top = 10.dp))
        Spacer(Modifier.height(24.dp))
    }
    confirmDelete?.let { spec ->
        AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text("Delete ${spec.title}?") },
            text = { Text(if (spec.required) "This model is required – the app needs it again before swapping." else "You can import or download it again later.") },
            confirmButton = { TextButton(onClick = { vm.deleteModel(spec); confirmDelete = null }) { Text("Delete", color = MaterialTheme.colorScheme.error) } },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ModelRow(vm: AppViewModel, spec: ModelSpec, s: ModelState, onDelete: () -> Unit) {
    val twins = remember(s) { vm.models.twins(spec.id) }
    Column(Modifier.padding(vertical = 4.dp, horizontal = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (s is ModelState.Ready) Icon(Icons.Default.CheckCircle, null, Modifier.size(18.dp), tint = Good) else Spacer(Modifier.size(18.dp))
            Column(Modifier.weight(1f).padding(horizontal = 10.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(spec.title, style = MaterialTheme.typography.bodyMedium)
                    twins.forEach { Pill(it, Modifier.padding(start = 6.dp)) }
                }
                Text("${spec.description} · ${mb(if (s is ModelState.Ready) vm.models.bytesOnDisk(spec.id) else spec.sizeBytes)} · ${spec.license}",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            when (s) {
                is ModelState.Ready -> TextButton(onClick = onDelete) { Text("Delete") }
                is ModelState.Downloading -> TextButton(onClick = { vm.cancelDownload(spec) }) { Text("Pause") }
                is ModelState.Verifying -> Text("Verifying…", style = MaterialTheme.typography.bodySmall)
                is ModelState.Installing -> Text("Installing…", style = MaterialTheme.typography.bodySmall)
                else -> TextButton(onClick = { vm.download(spec) }) { Text(if (s is ModelState.Failed) "Retry" else "Get") }
            }
        }
        when (s) {
            is ModelState.Downloading -> {
                LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp, start = 28.dp))
                Text("${mb(s.bytes)} / ${mb(s.total)}", Modifier.padding(start = 28.dp), style = MaterialTheme.typography.labelSmall)
            }
            is ModelState.Failed -> Text(s.message, Modifier.padding(start = 28.dp), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            else -> Unit
        }
    }
}
