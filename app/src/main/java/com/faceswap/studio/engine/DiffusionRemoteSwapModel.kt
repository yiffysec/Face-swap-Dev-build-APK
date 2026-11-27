package com.faceswap.studio.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.faceswap.core.DiffusionSourceAware
import com.faceswap.core.FaceSwapModel
import com.faceswap.core.Geometry
import com.faceswap.core.ImageOps
import com.faceswap.core.RgbImage
import com.faceswap.core.SwapOutput
import com.faceswap.core.Templates
import java.io.ByteArrayOutputStream

/**
 * [FaceSwapModel] implementation for a diffusion backend (InstantID / PuLID / REFace) running on the connected
 * server, wired in by [EngineHolder.buildRemoteEngine]. Unlike [com.faceswap.core.HyperSwap]/InSwapper, the actual
 * swap isn't a tensor op at all – it's a whole-image HTTP call (see [RemoteClient.diffusionSwap]) – so this class
 * slots a network round trip into the exact same per-face crop/warp/paste-back pipeline every other swapper uses
 * ([com.faceswap.core.FaceSwapEngine.swapFace]), rather than requiring a separate UI flow or processing path.
 *
 * [size] = 512: a middle ground. The server's SDXL-based backends want ~1024px for best quality, but [size] here
 * controls the aligned crop [com.faceswap.core.FaceSwapEngine] sends (same as every other swapper – `pixelBoost`
 * in [com.faceswap.core.SwapConfig] can push it higher via the usual k² tiling, same as any other model), and a
 * network round trip per *tile* would make pixel-boosted swaps dramatically slower for diffusion ids than for the
 * ONNX ones. 512 keeps one request per face in the common case while still giving the backend enough face detail
 * to work with.
 */
class DiffusionRemoteSwapModel(
    override val id: String,
    private val client: RemoteClient,
) : FaceSwapModel, DiffusionSourceAware {
    override val size = 512
    override val template = Templates.ARCFACE_128

    @Volatile private var activeSource: RgbImage? = null
    override fun setActiveSource(image: RgbImage?) { activeSource = image }

    // Diffusion backends compute their own identity embedding server-side (insightface, for instantid/pulid) from
    // the raw source photo [activeSource] – there's no on-device latent to precompute, so this is a harmless
    // passthrough kept only to satisfy FaceSwapModel's interface (and FaceSwapEngine's latent cache, which is never
    // actually consulted by swap() below).
    override fun sourceLatent(embedding: FloatArray): FloatArray = Geometry.normalize(embedding)

    override fun swap(crop: RgbImage, latent: FloatArray): SwapOutput {
        // Falls back to the target crop itself if no reference photo was set (shouldn't normally happen –
        // FaceSwapEngine.render() always calls setActiveSource first – but swapping "into itself" is a safer
        // degenerate case than crashing mid-pipeline on a null).
        val sourceImg = activeSource ?: crop
        val resultPng = client.diffusionSwap(id, encodePng(sourceImg), encodePng(crop))
        val decoded = decodePng(resultPng)
        val resized = if (decoded.width == crop.width && decoded.height == crop.height) decoded
            else ImageOps.resize(decoded, crop.width, crop.height)
        return SwapOutput(resized, null) // no model mask: the usual BOX/AREA mask types still apply at paste-back
    }

    override fun close() = Unit // no local resources – the network client is owned by RemoteCatalog, not this model
}

/** RgbImage.pixels is packed 0x00RRGGBB (see RgbImage.r()/g()/b()) – OR in an opaque alpha byte for Bitmap's ARGB_8888. */
private fun encodePng(img: RgbImage): ByteArray {
    val argb = IntArray(img.pixels.size) { img.pixels[it] or (0xFF shl 24) }
    val bitmap = Bitmap.createBitmap(argb, img.width, img.height, Bitmap.Config.ARGB_8888)
    return ByteArrayOutputStream().use { out ->
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        out.toByteArray()
    }
}

private fun decodePng(bytes: ByteArray): RgbImage {
    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        ?: throw IllegalStateException("couldn't decode diffusion server's response as an image")
    val argb = IntArray(bitmap.width * bitmap.height)
    bitmap.getPixels(argb, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    val rgb = IntArray(argb.size) { argb[it] and 0x00FFFFFF }
    return RgbImage(bitmap.width, bitmap.height, rgb)
}
