package com.faceswap.studio.engine

import android.content.Context
import android.util.Log
import com.faceswap.core.FaceSwapEngine
import com.faceswap.core.IdentityVerifier
import com.faceswap.core.InferenceModel
import com.faceswap.core.NoIdentityVerifier
import com.faceswap.studio.data.Accelerator
import com.faceswap.studio.data.AppSettings
import com.faceswap.studio.data.ModelManager
import com.faceswap.studio.util.Diagnostics
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** Lazily builds (and rebuilds on backend change) the face swap engine. */
class EngineHolder(private val context: Context, private val models: ModelManager, private val tuning: TuningStore) {
    private val mutex = Mutex()
    private var engine: FaceSwapEngine? = null
    private var builtWith: Triple<Accelerator, Int, Long>? = null

    /** Production identity verification provider plugs in here (Persona / Onfido / Veriff wrapper). */
    var verifier: IdentityVerifier = NoIdentityVerifier
        set(value) { field = value; engine?.policy?.verifier = value }

    @Volatile var internalLiveCaptureMode = false
        set(value) { field = value; engine?.policy?.internalLiveCaptureMode = value }

    /** Backend each loaded model actually ended up on (for the Settings screen). */
    val loadedBackends = java.util.concurrent.ConcurrentHashMap<String, String>()

    suspend fun get(settings: AppSettings): FaceSwapEngine = mutex.withLock {
        withContext(Dispatchers.Default) {
            val key = Triple(settings.accelerator, settings.threads, tuning.version)
            engine?.let { if (builtWith == key) return@withContext it }
            engine?.close(); engine = null; loadedBackends.clear()
            check(models.requiredReady) { "Required models are not downloaded yet" }
            Diagnostics.crumb("engine: building (${settings.accelerator}, ${settings.threads} threads)")
            val tuned = tuning.best()
            val e = FaceSwapEngine(
                resolve = { id -> models.file(id).takeIf { it.exists() } },
                loader = { id, file -> load(id, file, settings, tuned) },
                verifier = verifier,
                emapCache = models.emapCache,
            )
            e.policy.internalLiveCaptureMode = internalLiveCaptureMode
            engine = e; builtWith = key
            e
        }
    }

    private fun load(id: String, file: File, settings: AppSettings, tuned: Map<String, Backend>): InferenceModel {
        val dir = file.parentFile!!
        val backend = when (settings.accelerator) {
            Accelerator.AUTO -> tuned[id] ?: defaultBackend(id)
            Accelerator.CPU -> Backend.CPU
            Accelerator.XNNPACK -> Backend.XNNPACK
            Accelerator.NNAPI -> if (CpuInfo.isSnapdragon) Backend.CPU else Backend.NNAPI_FP16 // no NNAPI driver on Snapdragon
            Accelerator.QNN -> if (CpuInfo.isSnapdragon) Backend.QNN_HTP else Backend.CPU
            Accelerator.GPU -> if (LiteRtModel.available(dir, id)) Backend.LITERT_GPU_ACC32 else Backend.CPU
        }
        // fallback chain: tuned/forced → (INT8 → fp16 NPU) → CPU
        val chain = buildList {
            add(backend)
            if (backend == Backend.QNN_HTP_INT8) add(Backend.QNN_HTP)
            if (backend != Backend.CPU) add(Backend.CPU)
        }
        var last: Throwable? = null
        for ((i, b) in chain.withIndex()) {
            Diagnostics.crumb("model load start: $id ($b, ${settings.threads} threads)")
            try {
                return Backends.load(context, id, file, b, settings.threads).also {
                    loadedBackends[id] = it.backend + if (i > 0) " (fallback)" else ""
                    Diagnostics.crumb("model load ok: $id on ${it.backend}")
                }
            } catch (t: Throwable) {
                last = t
                Log.w("EngineHolder", "$b failed for $id", t)
                Diagnostics.crumb("model load failed with $b (${t.message?.lines()?.firstOrNull()})")
            }
        }
        throw last ?: IllegalStateException("could not load $id")
    }

    /** Before tuning: CPU is the only backend that never crashes natively (XNNPACK did on some builds). */
    private fun defaultBackend(@Suppress("UNUSED_PARAMETER") id: String) = Backend.CPU

    /** Frees model memory but keeps the engine (next use reloads lazily). */
    suspend fun trim() = mutex.withLock { engine?.unloadAll(); loadedBackends.clear() }

    suspend fun release() = mutex.withLock { engine?.close(); engine = null; builtWith = null; loadedBackends.clear() }
}
