package com.faceswap.studio.engine

import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.EnumSet

/** Lazily builds (and rebuilds on accelerator change) the face swap engine. */
class EngineHolder(private val models: ModelManager) {
    private val mutex = Mutex()
    private var engine: FaceSwapEngine? = null
    private var builtWith: Pair<Accelerator, Int>? = null

    /** Production identity verification provider plugs in here (Persona / Onfido / Veriff wrapper). */
    var verifier: IdentityVerifier = NoIdentityVerifier
        set(value) { field = value; engine?.policy?.verifier = value }

    @Volatile var internalLiveCaptureMode = false
        set(value) { field = value; engine?.policy?.internalLiveCaptureMode = value }

    suspend fun get(settings: AppSettings): FaceSwapEngine = mutex.withLock {
        withContext(Dispatchers.Default) {
            val key = settings.accelerator to settings.threads
            engine?.let { if (builtWith == key) return@withContext it }
            engine?.close(); engine = null
            check(models.requiredReady) { "Required models are not downloaded yet" }
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

    private fun factory(acc: Accelerator, threads: Int) = SessionFactory { env, path ->
        fun base() = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(threads)
        }
        try {
            val opts = base()
            when (acc) {
                Accelerator.CPU -> Unit
                Accelerator.XNNPACK -> {
                    opts.setIntraOpNumThreads(1)
                    opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                }
                Accelerator.NNAPI -> opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            env.createSession(path, opts)
        } catch (t: Throwable) {
            Log.w("EngineHolder", "Accelerator $acc failed for $path, falling back to CPU", t)
            env.createSession(path, base())
        }
    }
}
