package com.faceswap.studio.ui

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.LruCache
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.faceswap.studio.data.LibraryFace
import com.faceswap.studio.ui.theme.Accent
import com.faceswap.studio.ui.theme.LiveRed
import com.faceswap.studio.ui.theme.Scrim
import java.io.File

/** Camera capture that asks the camera app for the front camera. */
class TakeSelfie : ActivityResultContracts.TakePicture() {
    override fun createIntent(context: Context, input: Uri): Intent = super.createIntent(context, input).apply {
        putExtra("android.intent.extras.CAMERA_FACING", 1)
        putExtra("android.intent.extras.LENS_FACING_FRONT", 1)
        putExtra("android.intent.extra.USE_FRONT_CAMERA", true)
    }
}

/** Library thumbnails, bounded (≈ 24 MB) so a big library can't exhaust memory. */
private val thumbCache = object : LruCache<String, ImageBitmap>(24 * 1024 * 1024) {
    override fun sizeOf(key: String, value: ImageBitmap) = value.width * value.height * 4
}

fun loadThumb(f: File): ImageBitmap? {
    val key = f.absolutePath + f.lastModified()
    thumbCache.get(key)?.let { return it }
    return BitmapFactory.decodeFile(f.absolutePath)?.asImageBitmap()?.also { thumbCache.put(key, it) }
}

fun share(ctx: Context, uri: Uri?, mime: String) {
    uri ?: return
    ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
        type = mime; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }, "Share"))
}

// ===================================================================================================== layout blocks

@Composable
fun ScreenHeader(title: String, subtitle: String? = null, trailing: @Composable (() -> Unit)? = null) {
    Row(Modifier.fillMaxWidth().padding(top = 4.dp, bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.headlineSmall)
            subtitle?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
        }
        trailing?.invoke()
    }
}

@Composable
fun SectionTitle(text: String, modifier: Modifier = Modifier, action: (@Composable () -> Unit)? = null) {
    Row(modifier.fillMaxWidth().padding(top = 18.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(text.uppercase(), Modifier.weight(1f), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        action?.invoke()
    }
}

/** Rounded surface used for every grouped block. */
@Composable
fun Panel(modifier: Modifier = Modifier, padding: Dp = 14.dp, content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
    ) { Column(Modifier.padding(padding), content = content) }
}

@Composable
fun Note(text: String, modifier: Modifier = Modifier, color: Color = MaterialTheme.colorScheme.onSurfaceVariant) =
    Text(text, modifier.padding(top = 6.dp), style = MaterialTheme.typography.bodySmall, color = color)

@Composable
fun Pill(text: String, modifier: Modifier = Modifier, color: Color = MaterialTheme.colorScheme.surfaceContainerHighest, content: Color = MaterialTheme.colorScheme.onSurface, onClick: (() -> Unit)? = null) {
    Box(
        modifier.clip(CircleShape).background(color).let { if (onClick != null) it.clickable { onClick() } else it }.padding(horizontal = 10.dp, vertical = 5.dp),
    ) { Text(text, style = MaterialTheme.typography.labelSmall, color = content, maxLines = 1) }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun TimingChips(timings: Map<String, Long>) {
    if (timings.isEmpty()) return
    FlowRow(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        timings.filterValues { it > 0 }.forEach { (k, v) -> Pill("$k ${if (v >= 1000) "%.1fs".format(v / 1000f) else "${v}ms"}") }
    }
}

@Composable
fun NavRow(icon: ImageVector, title: String, subtitle: String? = null, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium).clickable(onClick = onClick).padding(horizontal = 6.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(38.dp).clip(RoundedCornerShape(12.dp)).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
            Icon(icon, null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(20.dp))
        }
        Column(Modifier.weight(1f).padding(horizontal = 14.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            subtitle?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis) }
        }
        Icon(Icons.Default.ChevronRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// ===================================================================================================== controls

@Composable
fun ToggleRow(label: String, value: Boolean, hint: String? = null, enabled: Boolean = true, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).clickable(enabled = enabled) { onChange(!value) }.padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            hint?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
        }
        Switch(value, onChange, enabled = enabled)
    }
}

@Composable
fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, steps: Int = 0, fmt: (Float) -> String = { "%.2f".format(it) }, onChange: (Float) -> Unit) {
    // local state while dragging: persisting settings on every drag frame would rebuild configs and write prefs 60×/s
    var local by remember(value) { mutableFloatStateOf(value) }
    Column(Modifier.padding(vertical = 2.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            Text(fmt(local), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        }
        Slider(local, { local = it }, valueRange = range, steps = steps, onValueChangeFinished = { onChange(local) })
    }
}

@Composable
fun <T> ChipRow(options: List<Pair<T, String>>, selected: (T) -> Boolean, onPick: (T) -> Unit) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(options.size) { i -> val (v, l) = options[i]; FilterChip(selected(v), { onPick(v) }, { Text(l) }) }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun <T> Segmented(options: List<Pair<T, String>>, selected: T, modifier: Modifier = Modifier, onPick: (T) -> Unit) {
    SingleChoiceSegmentedButtonRow(modifier.fillMaxWidth()) {
        options.forEachIndexed { i, (v, l) ->
            SegmentedButton(selected = v == selected, onClick = { onPick(v) }, shape = SegmentedButtonDefaults.itemShape(i, options.size), label = { Text(l, maxLines = 1) })
        }
    }
}

// ===================================================================================================== faces

@Composable
fun LiveBadge(modifier: Modifier = Modifier) = Box(
    modifier.clip(RoundedCornerShape(4.dp)).background(LiveRed).padding(horizontal = 4.dp, vertical = 1.dp),
) { Text("LIVE", color = Color.White, fontSize = 9.sp, fontWeight = FontWeight.Bold) }

@Composable
fun FaceAvatar(face: LibraryFace, size: Int = 64, selected: Boolean = false, onClick: (() -> Unit)? = null) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width((size + 10).dp)) {
        Box {
            val img = remember(face.thumbFile) { loadThumb(face.thumbFile) }
            val mod = Modifier.size(size.dp).clip(CircleShape)
                .border(if (selected) 3.dp else 1.dp, if (selected) Accent else MaterialTheme.colorScheme.outline, CircleShape)
                .let { if (onClick != null) it.clickable { onClick() } else it }
            if (img != null) Image(img, face.label, mod, contentScale = ContentScale.Crop) else Box(mod.background(MaterialTheme.colorScheme.surfaceVariant))
            if (face.isLive) LiveBadge(Modifier.align(Alignment.BottomCenter))
            if (selected) Box(Modifier.align(Alignment.TopEnd).size(20.dp).clip(CircleShape).background(Accent), contentAlignment = Alignment.Center) {
                Icon(Icons.Default.Check, null, tint = Color.White, modifier = Modifier.size(14.dp))
            }
        }
        Text(face.label, fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 4.dp),
            color = if (selected) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Horizontal library strip for picking source faces (multi-select = averaged identity). */
@Composable
fun SourcePicker(
    faces: List<LibraryFace>,
    selected: Set<String>,
    onToggle: (String) -> Unit,
    onAddGallery: () -> Unit,
    onAddCamera: () -> Unit,
) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp), contentPadding = PaddingValues(end = 8.dp)) {
        item { AddTile(Icons.Default.CameraAlt, "Capture", onAddCamera) }
        item { AddTile(Icons.Default.Add, "Photo", onAddGallery) }
        items(faces, key = { it.id }) { f -> FaceAvatar(f, selected = f.id in selected) { onToggle(f.id) } }
    }
    when {
        faces.isEmpty() -> Note("Add your face: Capture uses the camera (marked LIVE), Photo picks one from the gallery.")
        selected.size > 1 -> Note("${selected.size} photos combined into one identity – more angles, better likeness.")
        selected.isEmpty() -> Note("Tap a face to use it as the source. Tap several photos of the same person to combine them.")
    }
}

@Composable
private fun AddTile(icon: ImageVector, label: String, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(74.dp)) {
        Box(
            Modifier.size(64.dp).clip(CircleShape).background(MaterialTheme.colorScheme.surfaceContainerHighest)
                .border(1.dp, MaterialTheme.colorScheme.outline, CircleShape).clickable { onClick() },
            contentAlignment = Alignment.Center,
        ) { Icon(icon, label, tint = MaterialTheme.colorScheme.primary) }
        Text(label, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Target faces with per-face source assignment. */
@Composable
fun TargetFacesPanel(
    targets: List<TargetFace>,
    library: List<LibraryFace>,
    assignments: Map<Int, String>,
    onAssign: (Int, String?) -> Unit,
    onClear: () -> Unit,
) {
    if (targets.isEmpty()) return
    SectionTitle("Faces in target · ${targets.size}", action = { if (assignments.isNotEmpty()) TextButton(onClick = onClear) { Text("Reset") } })
    LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        items(targets, key = { it.index }) { t ->
            var open by remember { mutableStateOf(false) }
            val assigned = library.firstOrNull { it.id == assignments[t.index] }
            Box {
                Column(
                    Modifier.clip(MaterialTheme.shapes.medium).background(MaterialTheme.colorScheme.surfaceContainerHigh)
                        .border(if (assigned != null) 2.dp else 1.dp, if (assigned != null) Accent else MaterialTheme.colorScheme.outlineVariant, MaterialTheme.shapes.medium)
                        .clickable { open = true }.padding(8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Box {
                        Image(t.thumb.asImageBitmap(), null, Modifier.size(68.dp).clip(RoundedCornerShape(10.dp)), contentScale = ContentScale.Crop)
                        if (assigned != null) {
                            val img = remember(assigned.thumbFile) { loadThumb(assigned.thumbFile) }
                            if (img != null) Image(img, null, Modifier.size(28.dp).align(Alignment.BottomEnd).offset(6.dp, 6.dp).clip(CircleShape).border(2.dp, Accent, CircleShape), contentScale = ContentScale.Crop)
                        }
                    }
                    Spacer(Modifier.height(6.dp))
                    Text(assigned?.label ?: "Face ${t.index + 1}", fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.width(72.dp),
                        color = if (assigned != null) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant)
                }
                DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                    DropdownMenuItem(text = { Text("Keep original") }, onClick = { onAssign(t.index, null); open = false })
                    library.forEach { lf ->
                        DropdownMenuItem(text = { Text(lf.label + if (lf.isLive) "  · LIVE" else "") }, onClick = { onAssign(t.index, lf.id); open = false })
                    }
                }
            }
        }
    }
    Note(if (assignments.isNotEmpty()) "Per-face mode: only assigned faces are swapped (matched by identity across video frames)."
    else "Tap a face to give it its own source. Otherwise the selected source applies to the faces chosen below.")
}

// ===================================================================================================== canvas

/**
 * Image canvas. With [after] it becomes a before/after comparison: drag anywhere to move the divider, tap to toggle.
 * [overlay] draws on top (busy states, badges).
 */
@Composable
fun CompareCanvas(before: Bitmap, after: Bitmap?, modifier: Modifier = Modifier, overlay: @Composable BoxScope.() -> Unit = {}) {
    var split by remember(after) { mutableFloatStateOf(if (after != null) 1f else 0f) }
    var width by remember { mutableStateOf(IntSize.Zero) }
    val ratio = (before.width.toFloat() / before.height).coerceIn(0.4f, 2.5f)
    Box(
        modifier.fillMaxWidth().heightIn(max = 560.dp).aspectRatio(ratio).clip(MaterialTheme.shapes.large)
            .background(MaterialTheme.colorScheme.surfaceContainerLowest)
            .onSizeChanged { width = it }
            .let { m ->
                if (after == null) m else m
                    .pointerInput(after) { detectHorizontalDragGestures { change, _ -> split = (change.position.x / width.width.coerceAtLeast(1)).coerceIn(0f, 1f) } }
                    .pointerInput(after) { detectTapGestures { split = if (split > 0.5f) 0f else 1f } }
            },
    ) {
        val b = remember(before) { before.asImageBitmap() }
        Image(b, null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        if (after != null) {
            val a = remember(after) { after.asImageBitmap() }
            Image(a, null, Modifier.fillMaxSize().drawWithContent { clipRect(right = size.width * split) { this@drawWithContent.drawContent() } }, contentScale = ContentScale.Fit)
            if (split in 0.01f..0.99f) {
                val x = with(LocalDensity.current) { (width.width * split).toDp() }
                Box(Modifier.offset(x = x - 1.dp).width(2.dp).fillMaxHeight().background(Color.White))
                Box(Modifier.align(Alignment.CenterStart).offset(x = x - 16.dp).size(32.dp).clip(CircleShape).background(Color.White), contentAlignment = Alignment.Center) {
                    Icon(Icons.Default.SwapHoriz, null, tint = Color.Black, modifier = Modifier.size(20.dp))
                }
            }
            Pill(if (split > 0.5f) "SWAPPED" else "ORIGINAL", Modifier.align(Alignment.TopStart).padding(10.dp), color = Scrim, content = Color.White)
        }
        overlay()
    }
}

@Composable
fun BusyOverlay(text: String?, visible: Boolean, modifier: Modifier = Modifier) {
    AnimatedVisibility(visible, modifier) {
        Box(Modifier.fillMaxSize().background(Color(0x66000000)), contentAlignment = Alignment.Center) {
            Surface(shape = CircleShape, color = Scrim) {
                Row(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp, color = Color.White)
                    text?.let { Text(it, Modifier.padding(start = 10.dp), color = Color.White, style = MaterialTheme.typography.labelLarge) }
                }
            }
        }
    }
}

@Composable
fun EmptyCanvas(icon: ImageVector, title: String, subtitle: String, height: Dp = 240.dp, onClick: () -> Unit) {
    Box(
        Modifier.fillMaxWidth().height(height).clip(MaterialTheme.shapes.large).background(MaterialTheme.colorScheme.surfaceContainer)
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, MaterialTheme.shapes.large).clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(Modifier.size(56.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
                Icon(icon, null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(28.dp))
            }
            Text(title, Modifier.padding(top = 12.dp), style = MaterialTheme.typography.titleMedium)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
fun BitmapView(bmp: Bitmap, modifier: Modifier = Modifier) =
    Image(bmp.asImageBitmap(), null, modifier.fillMaxWidth().clip(MaterialTheme.shapes.large), contentScale = ContentScale.Fit)
