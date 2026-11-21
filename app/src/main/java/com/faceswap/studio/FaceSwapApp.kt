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

    override fun onCreate() {
        super.onCreate()
        models = ModelManager(this)
        library = FaceLibrary(this)
        settings = SettingsStore(this)
        engine = EngineHolder(models)
        // Production: engine.verifier = YourKycVerifier(...)  (see README "Identity verification")
    }
}
