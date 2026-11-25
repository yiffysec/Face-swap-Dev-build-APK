package com.faceswap.studio.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

val Accent = Color(0xFF8B7BFF)
val Accent2 = Color(0xFF3DDBB8)
val LiveRed = Color(0xFFFF5A5F)
val Good = Color(0xFF4ADE80)
val Warn = Color(0xFFF5A524)
val Scrim = Color(0xB3000000)

private val scheme = darkColorScheme(
    primary = Accent,
    onPrimary = Color.White,
    primaryContainer = Color(0xFF2B2550),
    onPrimaryContainer = Color(0xFFE4DFFF),
    secondary = Accent2,
    onSecondary = Color(0xFF00201A),
    secondaryContainer = Color(0xFF113A33),
    onSecondaryContainer = Color(0xFFB8F5E6),
    tertiary = Warn,
    background = Color(0xFF0B0C10),
    onBackground = Color(0xFFE9E9F1),
    surface = Color(0xFF0B0C10),
    onSurface = Color(0xFFE9E9F1),
    surfaceVariant = Color(0xFF1C1E27),
    onSurfaceVariant = Color(0xFFA9ACBB),
    surfaceContainerLowest = Color(0xFF08090C),
    surfaceContainerLow = Color(0xFF111218),
    surfaceContainer = Color(0xFF15171E),
    surfaceContainerHigh = Color(0xFF1B1D26),
    surfaceContainerHighest = Color(0xFF22252F),
    outline = Color(0xFF3A3D4A),
    outlineVariant = Color(0xFF2A2C36),
    error = Color(0xFFFF6E6E),
)

private val base = Typography()
private val typography = Typography(
    headlineSmall = base.headlineSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = (-0.2).sp),
    titleLarge = base.titleLarge.copy(fontWeight = FontWeight.SemiBold),
    titleMedium = base.titleMedium.copy(fontWeight = FontWeight.SemiBold),
    titleSmall = base.titleSmall.copy(fontWeight = FontWeight.SemiBold),
    labelLarge = base.labelLarge.copy(fontWeight = FontWeight.SemiBold),
    labelSmall = TextStyle(fontSize = 11.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.3.sp),
)

private val shapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(22.dp),
    extraLarge = RoundedCornerShape(28.dp),
)

@Composable
fun FaceSwapTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = scheme, typography = typography, shapes = shapes, content = content)
