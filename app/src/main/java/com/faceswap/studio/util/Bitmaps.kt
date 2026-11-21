package com.faceswap.studio.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.faceswap.core.Face
import com.faceswap.core.RgbImage
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object Bitmaps {

    /** Decodes with EXIF rotation applied, downscaled so the longest side ≤ [maxSide]. */
    fun decode(context: Context, uri: Uri, maxSide: Int = 4096): Bitmap {
        val src = ImageDecoder.createSource(context.contentResolver, uri)
        return ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
            val w = info.size.width
            val h = info.size.height
            val longest = max(w, h)
            if (longest > maxSide) {
                val s = maxSide.toFloat() / longest
                decoder.setTargetSize((w * s).roundToInt(), (h * s).roundToInt())
            }
        }.let { if (it.config != Bitmap.Config.ARGB_8888) it.copy(Bitmap.Config.ARGB_8888, false) else it }
    }

    fun decodeFile(file: File): Bitmap? = android.graphics.BitmapFactory.decodeFile(file.absolutePath)

    fun toRgb(bmp: Bitmap): RgbImage {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return RgbImage(bmp.width, bmp.height, px)
    }

    fun toBitmap(img: RgbImage): Bitmap =
        Bitmap.createBitmap(img.pixels, img.width, img.height, Bitmap.Config.ARGB_8888)

    /** Square thumbnail around a face (with margin). */
    fun faceThumb(img: RgbImage, face: Face, size: Int = 256, margin: Float = 0.45f): Bitmap {
        val side = max(face.width, face.height) * (1 + margin * 2)
        val x0 = (face.centerX - side / 2).roundToInt().coerceIn(0, img.width - 1)
        val y0 = (face.centerY - side / 2).roundToInt().coerceIn(0, img.height - 1)
        val s = min(side.roundToInt(), min(img.width - x0, img.height - y0)).coerceAtLeast(1)
        val full = toBitmap(img)
        val crop = Bitmap.createBitmap(full, x0, y0, s, s)
        return Bitmap.createScaledBitmap(crop, size, size, true)
    }

    fun saveToGallery(context: Context, bmp: Bitmap, name: String = "faceswap_${System.currentTimeMillis()}"): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "$name.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= 29) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/FaceSwapStudio")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 96, it) }
        if (Build.VERSION.SDK_INT >= 29) {
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        }
        return uri
    }

    fun saveVideoToGallery(context: Context, file: File): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= 29) {
                put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/FaceSwapStudio")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
        if (Build.VERSION.SDK_INT >= 29) {
            values.clear(); values.put(MediaStore.Video.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        }
        return uri
    }
}

