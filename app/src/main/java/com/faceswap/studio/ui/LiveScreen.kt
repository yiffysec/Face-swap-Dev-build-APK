package com.faceswap.studio.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.faceswap.studio.ui.theme.Scrim
import java.util.concurrent.Executors

/** Real-time camera face swap with MobileFaceSwap (preview quality). */
@Composable
fun LiveScreen(vm: AppViewModel, adders: FaceAdders, nav: Navigator) {
    val ctx = LocalContext.current
    val lib by vm.library.faces.collectAsState()
    val live by vm.live.collectAsState()
    val frame by vm.liveFrame.collectAsState()
    var front by rememberSaveable { mutableStateOf(true) }
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val perm = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }

    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 12.dp)) {
        ScreenHeader("Live", "Real-time preview swap (MobileFaceSwap)")
        if (!vm.liveAvailable()) {
            Panel {
                Text("Live preview models aren't installed", style = MaterialTheme.typography.titleSmall)
                Note("Import the models-live pack (or download the three MobileFaceSwap models, ~290 MB) to swap on the camera feed in real time.")
                Button(onClick = { nav.toSettings(SettingsPage.MODELS) }, modifier = Modifier.padding(top = 10.dp)) {
                    Icon(Icons.Default.Download, null, Modifier.size(18.dp)); Spacer(Modifier.size(6.dp)); Text("Open models")
                }
            }
            return@Column
        }
        if (!granted) {
            EmptyCanvas(Icons.Default.Videocam, "Allow camera", "Needed for the live preview", onClick = { perm.launch(Manifest.permission.CAMERA) })
            return@Column
        }

        Box(Modifier.fillMaxWidth().weight(1f).clip(MaterialTheme.shapes.large).background(Color.Black), contentAlignment = Alignment.Center) {
            frame?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit) }
            if (frame == null) CircularProgressIndicator(color = Color.White)
            Row(Modifier.align(Alignment.TopStart).padding(10.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (live.ready) {
                    Pill("%.0f fps".format(live.fps), color = Scrim, content = Color.White)
                    Pill("%.0f ms".format(live.ms), color = Scrim, content = Color.White)
                    Pill("${live.faces} face${if (live.faces == 1) "" else "s"}", color = Scrim, content = Color.White)
                } else Pill(if (live.faceId == null) "Pick a face below" else "Preparing…", color = Scrim, content = Color.White)
            }
            FilledIconButton(onClick = { front = !front }, modifier = Modifier.align(Alignment.TopEnd).padding(10.dp),
                colors = IconButtonDefaults.filledIconButtonColors(containerColor = Scrim, contentColor = Color.White)) {
                Icon(Icons.Default.Cameraswitch, "Switch camera")
            }
            live.message?.let { Pill(it, Modifier.align(Alignment.BottomCenter).padding(12.dp), color = Scrim, content = Color.White) }
        }
        Spacer(Modifier.height(12.dp))
        SourcePicker(lib, setOfNotNull(live.faceId), { id -> vm.prepareLive(id) }, adders.gallery, adders.camera)
        Note("Preview quality for framing – use Photo or Video for final results.")
    }

    if (granted && vm.liveAvailable()) CameraFrames(front) { bmp -> vm.onLiveFrame(bmp) }
    DisposableEffect(Unit) {
        vm.startLiveSession()
        onDispose { vm.stopLive() }
    }
}

/** CameraX ImageAnalysis → upright RGBA bitmaps (≈640 px), newest frame only, on a background thread. */
@Composable
private fun CameraFrames(front: Boolean, onFrame: (Bitmap) -> Unit) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    DisposableEffect(front) {
        val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "live-camera").apply { priority = Thread.MAX_PRIORITY } }
        val providerFuture = ProcessCameraProvider.getInstance(ctx)
        var provider: ProcessCameraProvider? = null
        var analysis: ImageAnalysis? = null
        providerFuture.addListener({
            provider = providerFuture.get()
            analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(android.util.Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build().also { a ->
                    a.setAnalyzer(executor) { proxy ->
                        try {
                            val bmp = proxy.toBitmap()
                            val m = Matrix().apply {
                                postRotate(proxy.imageInfo.rotationDegrees.toFloat())
                                if (front) postScale(-1f, 1f)
                            }
                            onFrame(Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true))
                        } catch (_: Throwable) {
                        } finally { proxy.close() }
                    }
                }
            runCatching {
                provider?.unbindAll()
                provider?.bindToLifecycle(owner, if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA, analysis!!)
            }
        }, ContextCompat.getMainExecutor(ctx))
        onDispose {
            analysis?.clearAnalyzer()
            runCatching { provider?.unbindAll() }
            executor.shutdown()
        }
    }
}
