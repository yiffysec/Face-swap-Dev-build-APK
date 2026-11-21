package com.faceswap.studio.engine

import android.util.Log
import com.faceswap.core.FaceSwapEngine
import com.faceswap.core.IdentityVerifier
import com.faceswap.core.ModelFiles
import com.faceswap.core.NoIdentityVerifier
import com.faceswap.core.SessionFactory
import com.faceswap.studio.data.Accelerator
import com.faceswap.studio.data.AppSettings
import com.faceswap.studio.data.ModelManager
import com.faceswap.studio.data.ModelRole
import com.faceswap.studio.util.Diagnostics
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Lazily builds (and rebuilds on accelerator change) the face swap engine. */
class EngineHolder(private val models: ModelManager, private val tuning: TuningStore) {
    private val mutex = Mutex()
    private var engine: FaceSwapEngine? = null
    private var builtWith: Triple<Accelerator, Int, Long>? = null

    /** Production identity verification provider plugs in here (Persona / Onfido / Veriff wrapper). */
    var verifier: IdentityVerifier = NoIdentityVerifier
        set(value) { field = value; engine?.policy?.verifier = value }

    @Volatile var internalLiveCaptureMode = false
        set(value) { field = value; engine?.policy?.internalLiveCaptureMode = value }

    suspend fun get(settings: AppSettings): FaceSwapEngine = mutex.withLock {
        withContext(Dispatchers.Default) {
            val key = Triple(settings.accelerator, settings.threads, tuning.version)
            engine?.let { if (builtWith == key) return@withContext it }
            engine?.close(); engine = null
            check(models.requiredReady) { "Required models are not downloaded yet" }
            Diagnostics.crumb("engine: building (${settings.accelerator}, ${settings.threads} threads)")
            val e = FaceSwapEngine(
                ModelFiles(
                    detector = models.file(ModelRole.DETECTOR),
                    recognizer = models.file(ModelRole.RECOGNIZER),
                    swapper = models.file(ModelRole.SWAPPER),
                    contentAnalyser = models.file(ModelRole.ANALYSER),
                    enhancer = models.file(ModelRole.ENHANCER),
                    emapCache = models.emapCache,
                ),
                factory(settings.accelerator, settings.threads),
                verifier,
            )
            e.policy.internalLiveCaptureMode = internalLiveCaptureMode
            engine = e; builtWith = key
            e
        }
    }

    suspend fun release() = mutex.withLock { engine?.close(); engine = null; builtWith = null }

    private fun factory(acc: Accelerator, threads: Int): SessionFactory {
        val tuned = tuning.best()
        return SessionFactory { env, path ->
            val name = java.io.File(path).name
            val id = name.removeSuffix(".onnx")
            val backend = when (acc) {
                Accelerator.AUTO -> tuned[id] ?: Backend.CPU
                Accelerator.CPU -> Backend.CPU
                Accelerator.XNNPACK -> Backend.XNNPACK
                Accelerator.NNAPI -> Backend.NNAPI_FP16
            }
            Diagnostics.crumb("model load start: $name ($backend, $threads threads)")
            try {
                env.createSession(path, Backends.options(backend, threads)).also { Diagnostics.crumb("model load ok: $name") }
            } catch (t: Throwable) {
                Log.w("EngineHolder", "$backend failed for $name, falling back to CPU", t)
                Diagnostics.crumb("model load failed with $backend (${t.message}), retrying on CPU: $name")
                env.createSession(path, Backends.options(Backend.CPU, threads)).also { Diagnostics.crumb("model load ok (CPU fallback): $name") }
            }
        }
    }
}
