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
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
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
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.faceswap.studio.util.Bitmaps
import java.util.concurrent.Executors

/** Real-time camera face swap with MobileFaceSwap (preview quality). */
@Composable
fun LiveScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    val lib by vm.library.faces.collectAsState()
    val live by vm.live.collectAsState()
    var selected by remember { mutableStateOf<String?>(null) }
    var front by remember { mutableStateOf(true) }
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val perm = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    var frame by remember { mutableStateOf<Bitmap?>(null) }

    Column(Modifier.padding(16.dp)) {
        Text("Live preview", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        if (!vm.liveAvailable()) {
            Text("Install the three MobileFaceSwap models (Models → Live preview) to use the real-time camera preview.", color = MaterialTheme.colorScheme.error)
            return@Column
        }
        Text("Pick a face, then point the camera. Preview quality – use Photo/Video for final results.", style = MaterialTheme.typography.bodySmall)
        SourcePicker(lib, setOfNotNull(selected), { id -> selected = id; vm.prepareLive(id) }, {}, {})
        Row {
            FilterChip(front, { front = true }, { Text("Front") }); FilterChip(!front, { front = false }, { Text("Back") }, Modifier.padding(start = 6.dp))
        }
        live.message?.let { Text(it, Modifier.padding(vertical = 6.dp)) }
        if (!granted) { Button(onClick = { perm.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }; return@Column }
        Box(Modifier.fillMaxWidth().weight(1f).padding(top = 8.dp).clip(RoundedCornerShape(14.dp)).background(Color.Black), contentAlignment = Alignment.Center) {
            frame?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit) }
            if (live.ready) Text("%.1f fps · %d face(s)".format(live.fps, live.faces), Modifier.align(Alignment.TopStart).padding(8.dp)
                .background(Color(0x99000000), RoundedCornerShape(6.dp)).padding(horizontal = 6.dp, vertical = 2.dp), color = Color.White)
        }
    }

    if (granted && vm.liveAvailable()) CameraFrames(front) { bmp ->
        val rgb = Bitmaps.toRgb(bmp)
        val out = vm.processLiveFrame(rgb)
        frame = Bitmaps.toBitmap(out)
    }
    DisposableEffect(Unit) { onDispose { vm.stopLive() } }
}

/** CameraX ImageAnalysis → upright RGBA bitmaps (≈640 px), newest frame only. */
@Composable
private fun CameraFrames(front: Boolean, onFrame: (Bitmap) -> Unit) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    DisposableEffect(front) {
        val executor = Executors.newSingleThreadExecutor()
        val providerFuture = ProcessCameraProvider.getInstance(ctx)
        var provider: ProcessCameraProvider? = null
        providerFuture.addListener({
            provider = providerFuture.get()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(android.util.Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { proxy ->
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
            runCatching {
                provider?.unbindAll()
                provider?.bindToLifecycle(owner, if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            }
        }, ContextCompat.getMainExecutor(ctx))
        onDispose { runCatching { provider?.unbindAll() }; executor.shutdown() }
    }
}
