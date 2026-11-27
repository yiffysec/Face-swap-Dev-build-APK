package com.faceswap.studio

import android.app.Application
import com.faceswap.studio.data.FaceLibrary
import com.faceswap.studio.data.ModelManager
import com.faceswap.studio.data.ProcessingMode
import com.faceswap.studio.data.SettingsStore
import com.faceswap.studio.engine.EngineHolder
import com.faceswap.studio.engine.RemoteCatalog
import com.faceswap.studio.engine.ServerEndpoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class FaceSwapApp : Application() {
    lateinit var models: ModelManager; private set
    lateinit var library: FaceLibrary; private set
    lateinit var settings: SettingsStore; private set
    lateinit var engine: EngineHolder; private set
    lateinit var tuning: com.faceswap.studio.engine.TuningStore; private set
    lateinit var perf: com.faceswap.studio.engine.PerformanceManager; private set
    lateinit var remote: RemoteCatalog; private set
    /** Lives as long as the process; background connect-on-launch is the only thing that uses it. */
    private val appScope = CoroutineScope(Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        if (getProcessName().endsWith(":tuner")) return // tuner process only needs the Activity
        com.faceswap.studio.util.Diagnostics.install(this, BuildConfig.VERSION_NAME)
        models = ModelManager(this)
        library = FaceLibrary(this)
        settings = SettingsStore(this)
        tuning = com.faceswap.studio.engine.TuningStore(this)
        remote = RemoteCatalog(this)
        engine = EngineHolder(this, models, tuning, remote)
        perf = com.faceswap.studio.engine.PerformanceManager(this)
        // Remote mode was left on from a previous session: reconnect in the background so the app is usable
        // immediately, without blocking startup on a network round trip.
        val s = settings.settings.value
        if (s.processingMode == ProcessingMode.REMOTE && s.serverHost.isNotBlank()) {
            appScope.launch { remote.connect(ServerEndpoint(s.serverHost, s.serverPort, s.serverUseTls, s.serverApiKey)) }
        }
        // Production: engine.verifier = YourKycVerifier(...)  (see README "Identity verification")
    }
}
