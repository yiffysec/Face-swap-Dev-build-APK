package com.faceswap.studio

import android.app.Application
import com.faceswap.studio.data.FaceLibrary
import com.faceswap.studio.data.ModelManager
import com.faceswap.studio.data.SettingsStore
import com.faceswap.studio.engine.EngineHolder

class FaceSwapApp : Application() {
    lateinit var models: ModelManager; private set
    lateinit var library: FaceLibrary; private set
    lateinit var settings: SettingsStore; private set
    lateinit var engine: EngineHolder; private set
    lateinit var tuning: com.faceswap.studio.engine.TuningStore; private set
    lateinit var perf: com.faceswap.studio.engine.PerformanceManager; private set

    override fun onCreate() {
        super.onCreate()
        if (getProcessName().endsWith(":tuner")) return // tuner process only needs the Activity
        com.faceswap.studio.util.Diagnostics.install(this, BuildConfig.VERSION_NAME)
        models = ModelManager(this)
        library = FaceLibrary(this)
        settings = SettingsStore(this)
        tuning = com.faceswap.studio.engine.TuningStore(this)
        engine = EngineHolder(this, models, tuning)
        perf = com.faceswap.studio.engine.PerformanceManager(this)
        // Production: engine.verifier = YourKycVerifier(...)  (see README "Identity verification")
    }
}
