package com.faceswap.studio.ui

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.faceswap.studio.data.LibraryFace
import com.faceswap.studio.ui.theme.Accent
import com.faceswap.studio.ui.theme.LiveRed
import java.io.File

/** Camera capture that asks the camera app for the front camera. */
class TakeSelfie : ActivityResultContracts.TakePicture() {
    override fun createIntent(context: Context, input: Uri): Intent = super.createIntent(context, input).apply {
        putExtra("android.intent.extras.CAMERA_FACING", 1)
        putExtra("android.intent.extras.LENS_FACING_FRONT", 1)
        putExtra("android.intent.extra.USE_FRONT_CAMERA", true)
    }
}

private val thumbCache = HashMap<String, ImageBitmap>()

fun loadThumb(f: File): ImageBitmap? {
    val key = f.absolutePath + f.lastModified()
    thumbCache[key]?.let { return it }
    return BitmapFactory.decodeFile(f.absolutePath)?.asImageBitmap()?.also { thumbCache[key] = it }
}

@Composable
fun SectionTitle(text: String, modifier: Modifier = Modifier) =
    Text(text, modifier = modifier.padding(top = 14.dp, bottom = 6.dp), style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

@Composable
fun LiveBadge(modifier: Modifier = Modifier) = Box(
    modifier.clip(RoundedCornerShape(4.dp)).background(LiveRed).padding(horizontal = 4.dp, vertical = 1.dp),
) { Text("LIVE", color = Color.White, fontSize = 9.sp, fontWeight = FontWeight.Bold) }

@Composable
fun FaceAvatar(face: LibraryFace, size: Int = 64, selected: Boolean = false, onClick: (() -> Unit)? = null) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width((size + 8).dp)) {
        Box {
            val img = remember(face.thumbFile) { loadThumb(face.thumbFile) }
            val mod = Modifier.size(size.dp).clip(CircleShape)
                .border(if (selected) 3.dp else 1.dp, if (selected) Accent else Color(0x33FFFFFF), CircleShape)
                .let { if (onClick != null) it.clickable { onClick() } else it }
            if (img != null) Image(img, face.label, mod, contentScale = ContentScale.Crop) else Box(mod.background(Color.DarkGray))
            if (face.isLive) LiveBadge(Modifier.align(Alignment.BottomCenter))
            if (selected) Box(Modifier.align(Alignment.TopEnd).size(20.dp).clip(CircleShape).background(Accent), contentAlignment = Alignment.Center) {
                Icon(Icons.Default.Check, null, tint = Color.White, modifier = Modifier.size(14.dp))
            }
        }
        Text(face.label, fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 2.dp))
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
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        item { AddTile(Icons.Default.CameraAlt, "Camera", onAddCamera) }
        item { AddTile(Icons.Default.Add, "Photo", onAddGallery) }
        items(faces, key = { it.id }) { f -> FaceAvatar(f, selected = f.id in selected) { onToggle(f.id) } }
    }
    if (selected.size > 1) Text("${selected.size} photos combined into one identity", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun AddTile(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(72.dp)) {
        Box(
            Modifier.size(64.dp).clip(CircleShape).border(1.dp, Color(0x55FFFFFF), CircleShape).clickable { onClick() },
            contentAlignment = Alignment.Center,
        ) { Icon(icon, label) }
        Text(label, fontSize = 11.sp, modifier = Modifier.padding(top = 2.dp))
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
    Row(verticalAlignment = Alignment.CenterVertically) {
        SectionTitle("Faces in target (${targets.size}) – tap to assign individually", Modifier.weight(1f))
        if (assignments.isNotEmpty()) TextButton(onClick = onClear) { Text("Reset") }
    }
    LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        items(targets, key = { it.index }) { t ->
            var open by remember { mutableStateOf(false) }
            val assigned = library.firstOrNull { it.id == assignments[t.index] }
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                border = if (assigned != null) BorderStroke(2.dp, Accent) else null,
                modifier = Modifier.clickable { open = true },
            ) {
                Column(Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Image(t.thumb.asImageBitmap(), null, Modifier.size(72.dp).clip(RoundedCornerShape(8.dp)), contentScale = ContentScale.Crop)
                    Spacer(Modifier.height(4.dp))
                    Text(assigned?.label?.let { "→ $it" } ?: "Face ${t.index + 1}", fontSize = 11.sp, maxLines = 1)
                }
                DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                    DropdownMenuItem(text = { Text("Keep original") }, onClick = { onAssign(t.index, null); open = false })
                    library.forEach { lf ->
                        DropdownMenuItem(text = { Text(lf.label + if (lf.isLive) "  • LIVE" else "") }, onClick = { onAssign(t.index, lf.id); open = false })
                    }
                }
            }
        }
    }
    if (assignments.isNotEmpty()) Text("Per-face mode: only assigned faces are swapped (tracked by identity in videos).", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
fun BitmapView(bmp: Bitmap, modifier: Modifier = Modifier) =
    Image(bmp.asImageBitmap(), null, modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)), contentScale = ContentScale.Fit)

@Composable
fun PillButton(text: String, onClick: () -> Unit, enabled: Boolean = true, modifier: Modifier = Modifier) =
    OutlinedButton(onClick = onClick, enabled = enabled, modifier = modifier) { Text(text) }
