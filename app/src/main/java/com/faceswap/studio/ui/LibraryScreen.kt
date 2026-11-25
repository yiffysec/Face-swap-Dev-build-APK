package com.faceswap.studio.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Face
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.faceswap.studio.data.LibraryFace
import java.text.DateFormat
import java.util.Date

@Composable
fun LibraryScreen(vm: AppViewModel, adders: FaceAdders) {
    val lib by vm.library.faces.collectAsState()
    var editing by remember { mutableStateOf<LibraryFace?>(null) }
    var deleting by remember { mutableStateOf<LibraryFace?>(null) }

    LazyVerticalGrid(
        GridCells.Adaptive(156.dp), Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item(span = { GridItemSpan(maxLineSpan) }) {
            Column {
                ScreenHeader("Faces", "${lib.size} saved · reused by photo, video and live")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = adders.camera, modifier = Modifier.weight(1f).height(48.dp)) { Icon(Icons.Default.CameraAlt, null); Spacer(Modifier.size(6.dp)); Text("Live capture") }
                    OutlinedButton(onClick = adders.gallery, modifier = Modifier.weight(1f).height(48.dp)) { Icon(Icons.Default.Add, null); Spacer(Modifier.size(6.dp)); Text("From photo") }
                }
                Note("Camera captures are stored as PNG with LIVE metadata. Several photos of the same person can be combined when swapping.")
                Spacer(Modifier.height(6.dp))
            }
        }
        if (lib.isEmpty()) item(span = { GridItemSpan(maxLineSpan) }) {
            EmptyCanvas(Icons.Default.Face, "No faces yet", "Capture or pick a photo to start", height = 200.dp, onClick = adders.camera)
        }
        items(lib, key = { it.id }) { f ->
            Card(shape = MaterialTheme.shapes.large, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Box(Modifier.fillMaxWidth().aspectRatio(1f)) {
                        val img = remember(f.thumbFile) { loadThumb(f.thumbFile) }
                        if (img != null) Image(img, f.label, Modifier.fillMaxSize().clip(MaterialTheme.shapes.large), contentScale = ContentScale.Crop)
                        else Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.surfaceVariant))
                        if (f.isLive) LiveBadge(Modifier.align(Alignment.TopStart).padding(8.dp))
                    }
                    Row(Modifier.fillMaxWidth().padding(start = 12.dp, end = 4.dp, top = 6.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(f.label, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(DateFormat.getDateInstance(DateFormat.SHORT).format(Date(f.createdAt)) + if (f.isLive) " · camera" else " · photo",
                                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        IconButton(onClick = { editing = f }) { Icon(Icons.Default.Edit, "Rename", Modifier.size(20.dp)) }
                        IconButton(onClick = { deleting = f }) { Icon(Icons.Default.Delete, "Delete", Modifier.size(20.dp)) }
                    }
                }
            }
        }
    }
    editing?.let { f ->
        var name by remember(f.id) { mutableStateOf(f.label) }
        AlertDialog(
            onDismissRequest = { editing = null },
            title = { Text("Rename face") },
            text = { OutlinedTextField(name, { name = it }, singleLine = true) },
            confirmButton = { TextButton(onClick = { vm.renameFace(f.id, name.trim().ifEmpty { f.label }); editing = null }) { Text("Save") } },
            dismissButton = { TextButton(onClick = { editing = null }) { Text("Cancel") } },
        )
    }
    deleting?.let { f ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete “${f.label}”?") },
            text = { Text("The face and its photo are removed from the library.") },
            confirmButton = { TextButton(onClick = { vm.deleteFace(f.id); deleting = null }) { Text("Delete", color = MaterialTheme.colorScheme.error) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
}
