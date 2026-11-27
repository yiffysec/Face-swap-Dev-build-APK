package com.faceswap.studio.ui

import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

enum class Tab(val label: String, val icon: ImageVector) {
    PHOTO("Photo", Icons.Default.Image), VIDEO("Video", Icons.Default.Movie), LIVE("Live", Icons.Default.Videocam),
    LIBRARY("Faces", Icons.Default.Face), SETTINGS("Settings", Icons.Default.Settings),
}

/** Sub-pages of the Settings tab. */
enum class SettingsPage { HUB, SWAP, MASKS, PERFORMANCE, SERVER, MODELS, DIAGNOSTICS }

/** Launchers shared by every screen that can add faces to the library. */
class FaceAdders(val gallery: () -> Unit, val camera: () -> Unit)

/** Lets any screen jump to a Settings sub-page (e.g. "install models", "tune device"). */
class Navigator(val toSettings: (SettingsPage) -> Unit)

private fun notReadyMessage(mode: com.faceswap.studio.data.ProcessingMode) =
    if (mode == com.faceswap.studio.data.ProcessingMode.REMOTE) "Connect to a server first (Settings → Processing)" else "Install the required models first"

@Composable
fun FaceSwapRoot(vm: AppViewModel) {
    val ctx = LocalContext.current
    val settings by vm.settings.collectAsState()
    val requiredReady by vm.readyToSwap.collectAsState()
    var tab by rememberSaveable { mutableStateOf(Tab.PHOTO) }
    var page by rememberSaveable { mutableStateOf(SettingsPage.HUB) }
    val nav = remember { Navigator { p -> page = p; tab = Tab.SETTINGS } }

    val toast by vm.toast.collectAsState()
    LaunchedEffect(toast) { toast?.let { Toast.makeText(ctx, it, Toast.LENGTH_SHORT).show(); vm.consumeToast() } }
    val navigate by vm.navigate.collectAsState()
    LaunchedEffect(navigate) {
        when (navigate) { "photo" -> tab = Tab.PHOTO; "video" -> tab = Tab.VIDEO }
        if (navigate != null) vm.consumeNavigation()
    }

    val pickFace = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let { vm.importFromGallery(it) } }
    val camera = rememberLauncherForActivityResult(TakeSelfie()) { ok -> vm.onCaptured(ok) }
    // The app declares CAMERA (for live preview), so Android requires the grant before ACTION_IMAGE_CAPTURE too.
    val cameraPerm = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) camera.launch(vm.newCaptureUri()) else Toast.makeText(ctx, "Camera permission is needed for live capture", Toast.LENGTH_SHORT).show()
    }
    val adders = remember {
        FaceAdders(
            gallery = {
                if (!vm.processingReady) Toast.makeText(ctx, notReadyMessage(settings.processingMode), Toast.LENGTH_SHORT).show()
                else pickFace.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            camera = {
                if (!vm.processingReady) Toast.makeText(ctx, notReadyMessage(settings.processingMode), Toast.LENGTH_SHORT).show()
                else if (androidx.core.content.ContextCompat.checkSelfPermission(ctx, android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) camera.launch(vm.newCaptureUri())
                else cameraPerm.launch(android.Manifest.permission.CAMERA)
            },
        )
    }

    BackHandler(enabled = tab == Tab.SETTINGS && page != SettingsPage.HUB) { page = SettingsPage.HUB }
    BackHandler(enabled = tab != Tab.PHOTO && !(tab == Tab.SETTINGS && page != SettingsPage.HUB)) { tab = Tab.PHOTO }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainerLow, tonalElevation = 0.dp) {
                Tab.entries.forEach { t ->
                    NavigationBarItem(
                        selected = tab == t,
                        onClick = { if (tab == t && t == Tab.SETTINGS) page = SettingsPage.HUB; tab = t },
                        icon = { Icon(t.icon, t.label) },
                        label = { Text(t.label, maxLines = 1) },
                        colors = NavigationBarItemDefaults.colors(indicatorColor = MaterialTheme.colorScheme.primaryContainer),
                    )
                }
            }
        },
    ) { pad ->
        Box(Modifier.fillMaxSize().padding(pad)) {
            val gated = !requiredReady && tab != Tab.SETTINGS
            AnimatedContent(targetState = if (gated) null else tab, transitionSpec = { fadeIn() togetherWith fadeOut() }, label = "tab") { t ->
                when (t) {
                    null -> if (settings.processingMode == com.faceswap.studio.data.ProcessingMode.REMOTE) ServerGateScreen(vm) else ModelsScreen(vm, gate = true, onBack = null)
                    Tab.PHOTO -> PhotoScreen(vm, adders, nav)
                    Tab.VIDEO -> VideoScreen(vm, adders, nav)
                    Tab.LIVE -> LiveScreen(vm, adders, nav)
                    Tab.LIBRARY -> LibraryScreen(vm, adders)
                    Tab.SETTINGS -> SettingsRoot(vm, page) { page = it }
                }
            }
        }
    }

    // Processing indicator after a capture / photo import
    val importing by vm.importing.collectAsState()
    if (importing) {
        AlertDialog(
            onDismissRequest = {},
            title = { Text("Reading face…") },
            text = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(24.dp))
                    Spacer(Modifier.size(12.dp))
                    Text("Detecting the face and computing its identity. The first time also loads the models.")
                }
            },
            confirmButton = {},
        )
    }

    // Crash report from the previous run (or opened from Settings → Diagnostics)
    val report by vm.crashReport.collectAsState()
    report?.let { text ->
        val clipboard = LocalClipboardManager.current
        AlertDialog(
            onDismissRequest = { vm.dismissCrashReport() },
            title = { Text("Crash report") },
            text = {
                Column {
                    Text("Copy or share this report so the problem can be fixed. It never leaves the device unless you share it.", style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.size(8.dp))
                    Text(
                        text,
                        Modifier.heightIn(max = 360.dp).verticalScroll(rememberScrollState()).padding(top = 4.dp),
                        style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { clipboard.setText(AnnotatedString(text)); Toast.makeText(ctx, "Copied", Toast.LENGTH_SHORT).show() }) { Text("Copy") }
            },
            dismissButton = {
                Row {
                    TextButton(onClick = { share(ctx, vm.reportShareUri(), "text/plain") }) { Text("Share") }
                    TextButton(onClick = { vm.dismissCrashReport() }) { Text("Close") }
                }
            },
        )
    }

    // Choose which face to save when an imported photo has several.
    val imp by vm.import.collectAsState()
    imp?.let { st ->
        var label by remember(st) { mutableStateOf("") }
        AlertDialog(
            onDismissRequest = { vm.cancelImport() },
            title = { Text(if (st.live) "Which face is yours?" else "Which face should be saved?") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedTextField(label, { label = it }, label = { Text("Name (optional)") }, singleLine = true)
                    Text("Tap a face to save it", style = MaterialTheme.typography.bodySmall)
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(st.faces) { f ->
                            Image(f.thumb.asImageBitmap(), null, Modifier.size(84.dp).clip(RoundedCornerShape(12.dp)).clickable { vm.chooseImportFace(f, label) }, contentScale = ContentScale.Crop)
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { vm.cancelImport() }) { Text("Cancel") } },
        )
    }
}
