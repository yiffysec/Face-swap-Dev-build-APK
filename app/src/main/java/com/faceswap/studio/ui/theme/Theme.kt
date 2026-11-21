package com.faceswap.studio.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val Accent = Color(0xFF8C6CFF)
val Accent2 = Color(0xFF1DE9B6)
val LiveRed = Color(0xFFFF5252)

private val scheme = darkColorScheme(
    primary = Accent,
    onPrimary = Color.White,
    secondary = Accent2,
    onSecondary = Color(0xFF00201A),
    background = Color(0xFF0E0F14),
    surface = Color(0xFF16181F),
    surfaceVariant = Color(0xFF20232C),
    surfaceContainer = Color(0xFF1A1C24),
    onSurface = Color(0xFFE8E8F0),
    onSurfaceVariant = Color(0xFFB0B3C0),
    error = Color(0xFFFF6E6E),
)

@Composable
fun FaceSwapTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = scheme, content = content)
