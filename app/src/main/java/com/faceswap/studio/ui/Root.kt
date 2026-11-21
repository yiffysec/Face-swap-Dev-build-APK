package com.faceswap.studio.ui

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

enum class Tab(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    PHOTO("Photo", Icons.Default.Image), VIDEO("Video", Icons.Default.Movie), LIVE("Live", Icons.Default.Videocam), LIBRARY("Faces", Icons.Default.Face),
    MODELS("Models", Icons.Default.CloudDownload), SETTINGS("Settings", Icons.Default.Settings),
}

/** Launchers shared by every screen that can add faces to the library. */
class FaceAdders(val gallery: () -> Unit, val camera: () -> Unit)

@Composable
fun FaceSwapRoot(vm: AppViewModel) {
    val ctx = LocalContext.current
    val states by vm.models.states.collectAsState()
    val requiredReady = remember(states) { vm.models.requiredReady }
    var tab by rememberSaveable { mutableStateOf(if (vm.models.requiredReady) Tab.PHOTO else Tab.MODELS) }

    val toast by vm.toast.collectAsState()
    LaunchedEffect(toast) { toast?.let { Toast.makeText(ctx, it, Toast.LENGTH_SHORT).show(); vm.consumeToast() } }

    val pickFace = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let { vm.importFromGallery(it) } }
    val camera = rememberLauncherForActivityResult(TakeSelfie()) { ok -> vm.onCaptured(ok) }
    // The app declares CAMERA (for live preview), so Android requires the grant before ACTION_IMAGE_CAPTURE too.
    val cameraPerm = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) camera.launch(vm.newCaptureUri()) else Toast.makeText(ctx, "Camera permission is needed for live capture", Toast.LENGTH_SHORT).show()
    }
    val adders = remember {
        FaceAdders(
            gallery = {
                if (!vm.models.requiredReady) Toast.makeText(ctx, "Download the required models first", Toast.LENGTH_SHORT).show()
                else pickFace.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            camera = {
                if (!vm.models.requiredReady) Toast.makeText(ctx, "Download the required models first", Toast.LENGTH_SHORT).show()
                else if (androidx.core.content.ContextCompat.checkSelfPermission(ctx, android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) camera.launch(vm.newCaptureUri())
                else cameraPerm.launch(android.Manifest.permission.CAMERA)
            },
        )
    }

    Scaffold(
        bottomBar = {
            NavigationBar {
                Tab.entries.forEach { t ->
                    NavigationBarItem(selected = tab == t, onClick = { tab = t }, icon = { Icon(t.icon, t.label) }, label = { Text(t.label, maxLines = 1) })
                }
            }
        },
    ) { pad ->
        Box(Modifier.fillMaxSize().padding(pad)) {
            when {
                !requiredReady && tab in setOf(Tab.PHOTO, Tab.VIDEO, Tab.LIVE, Tab.LIBRARY) -> ModelsScreen(vm, gate = true)
                tab == Tab.LIVE -> LiveScreen(vm)
                tab == Tab.PHOTO -> PhotoScreen(vm, adders)
                tab == Tab.VIDEO -> VideoScreen(vm, adders)
                tab == Tab.LIBRARY -> LibraryScreen(vm, adders)
                tab == Tab.MODELS -> ModelsScreen(vm, gate = false)
                tab == Tab.SETTINGS -> SettingsScreen(vm)
            }
        }
    }

    // Processing indicator after a capture / photo import
    val importing by vm.importing.collectAsState()
    if (importing) {
        AlertDialog(
            onDismissRequest = {},
            title = { Text("Processing face…") },
            text = {
                androidx.compose.foundation.layout.Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                    androidx.compose.material3.CircularProgressIndicator(Modifier.size(24.dp))
                    androidx.compose.foundation.layout.Spacer(Modifier.size(12.dp))
                    Text("Loading models and reading the face. The first time takes a little longer.")
                }
            },
            confirmButton = {},
        )
    }

    // Crash report from the previous run (or opened from Settings → Diagnostics)
    val report by vm.crashReport.collectAsState()
    report?.let { text ->
        val clipboard = androidx.compose.ui.platform.LocalClipboardManager.current
        AlertDialog(
            onDismissRequest = { vm.dismissCrashReport() },
            title = { Text("Crash report") },
            text = {
                Column {
                    Text("The app closed unexpectedly last time. Copy or share this report so it can be fixed.", style = MaterialTheme.typography.bodySmall)
                    androidx.compose.foundation.layout.Spacer(Modifier.size(8.dp))
                    Text(
                        text,
                        Modifier.heightIn(max = 360.dp).verticalScroll(rememberScrollState()).padding(top = 4.dp),
                        style = MaterialTheme.typography.bodySmall.copy(fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace),
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { clipboard.setText(androidx.compose.ui.text.AnnotatedString(text)); Toast.makeText(ctx, "Copied", Toast.LENGTH_SHORT).show() }) { Text("Copy") }
            },
            dismissButton = {
                androidx.compose.foundation.layout.Row {
                    TextButton(onClick = {
                        vm.reportShareUri()?.let { uri ->
                            ctx.startActivity(android.content.Intent.createChooser(android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                                type = "text/plain"; putExtra(android.content.Intent.EXTRA_STREAM, uri); addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }, "Share report"))
                        }
                    }) { Text("Share") }
                    TextButton(onClick = { vm.dismissCrashReport() }) { Text("Close") }
                }
            },
        )
    }

    // Choose which face to save when an imported photo has several.
    val imp by vm.import.collectAsState()
    imp?.let { st ->
        var label by remember { mutableStateOf("") }
        AlertDialog(
            onDismissRequest = { vm.cancelImport() },
            title = { Text(if (st.live) "Which face is yours?" else "Which face should be saved?") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedTextField(label, { label = it }, label = { Text("Name (optional)") }, singleLine = true)
                    Text("Tap a face to save it", style = MaterialTheme.typography.bodySmall)
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(st.faces) { f ->
                            Image(f.thumb.asImageBitmap(), null, Modifier.size(80.dp).clip(RoundedCornerShape(10.dp)).clickable { vm.chooseImportFace(f, label) }, contentScale = ContentScale.Crop)
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { vm.cancelImport() }) { Text("Cancel") } },
        )
    }
}
