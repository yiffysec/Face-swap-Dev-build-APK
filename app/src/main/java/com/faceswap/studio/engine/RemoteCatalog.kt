package com.faceswap.studio.engine

import android.content.Context
import com.faceswap.core.ModelIds
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File

sealed class ServerStatus {
    object Disconnected : ServerStatus()
    object Connecting : ServerStatus()
    data class Connected(val info: ServerInfo) : ServerStatus()
    data class Failed(val message: String) : ServerStatus()
}

/**
 * Model availability backed by a remote server instead of local files. Exposes the same `has(id)` shape as
 * [com.faceswap.studio.data.ModelManager] so the rest of the app (engine, settings UI) doesn't need two code paths
 * for "is this model usable right now".
 *
 * One wrinkle: InSwapper's `sourceLatent` is computed on-device from the raw identity-projection matrix read
 * straight out of the .onnx file (see `InSwapper.loadEmap` in Models.kt) – a local-file optimisation with no remote
 * equivalent. [has] therefore never reports an "inswapper*" id as available remotely; HyperSwap (the default
 * swapper, which carries its own mask and needs no such trick) and everything else works identically in both modes.
 */
class RemoteCatalog(context: Context) {
    // Marker files exist only so FaceSwapEngine's `resolve(id)?.exists()` gate passes; their contents are never read.
    private val markerDir = File(context.filesDir, "remote-markers").apply { mkdirs() }

    private val _status = MutableStateFlow<ServerStatus>(ServerStatus.Disconnected)
    val status: StateFlow<ServerStatus> = _status.asStateFlow()

    @Volatile private var client: RemoteClient? = null
    @Volatile private var connectedTo: ServerEndpoint? = null
    @Volatile private var manifests: Map<String, ServerModelInfo> = emptyMap()
    @Volatile private var diffusionModels: Map<String, DiffusionModelInfo> = emptyMap()

    fun currentClient(): RemoteClient? = client
    fun currentEndpoint(): ServerEndpoint? = connectedTo

    fun has(id: String): Boolean = !id.startsWith("inswapper") && (manifests.containsKey(id) || diffusionModels.containsKey(id))
    fun manifest(id: String): ServerModelInfo? = manifests[id]
    fun diffusionModel(id: String): DiffusionModelInfo? = diffusionModels[id]

    /** The minimum a photo/video swap needs: a detector, recognizer, a non-InSwapper swapper, and the content policy model. */
    val requiredReady: Boolean
        get() = status.value is ServerStatus.Connected && has(ModelIds.ARCFACE) && has(ModelIds.OPEN_NSFW) &&
            (has(ModelIds.YUNET) || has(ModelIds.SCRFD)) && ModelIds.SWAPPERS.any { has(it) }

    fun markerFile(id: String): File? = File(markerDir, "$id.marker").takeIf { has(id) }

    /** (Re)connects to [ep] and refreshes the model manifest. Safe to call repeatedly, e.g. from a "Connect" button. */
    suspend fun connect(ep: ServerEndpoint): ServerStatus = withContext(Dispatchers.IO) {
        if (!ep.valid) return@withContext setStatus(ServerStatus.Failed("Enter a server address and port"))
        setStatus(ServerStatus.Connecting)
        try {
            val newClient = RemoteClient(ep)
            val info = newClient.fetchServerInfo()
            client?.close()
            client = newClient
            connectedTo = ep
            manifests = info.models.associateBy { it.id }
            // Best-effort: fetchDiffusionInfo() never throws (see its own doc), so a box with no diffusion tier
            // installed (setup_diffusion.sh never run) just contributes an empty map here, same as if those model
            // ids simply weren't in diffusion_manifest.yaml — never fails the overall connect().
            diffusionModels = newClient.fetchDiffusionInfo().associateBy { it.id }
            resyncMarkers()
            setStatus(ServerStatus.Connected(info))
        } catch (t: Throwable) {
            manifests = emptyMap()
            diffusionModels = emptyMap()
            resyncMarkers()
            setStatus(ServerStatus.Failed(describeFailure(t)))
        }
    }

    fun disconnect() {
        client?.close(); client = null; connectedTo = null
        manifests = emptyMap()
        diffusionModels = emptyMap()
        resyncMarkers()
        setStatus(ServerStatus.Disconnected)
    }

    private fun setStatus(s: ServerStatus): ServerStatus { _status.value = s; return s }

    private fun resyncMarkers() {
        markerDir.listFiles()?.forEach { it.delete() }
        manifests.keys.forEach { File(markerDir, "$it.marker").createNewFile() }
        diffusionModels.keys.forEach { File(markerDir, "$it.marker").createNewFile() }
    }

    private fun describeFailure(t: Throwable): String = when {
        t is java.net.UnknownHostException -> "Can't find that address – check the hostname/IP"
        t is java.net.ConnectException || t is java.net.SocketTimeoutException -> "Couldn't reach the server – check the address, port, and that it's running"
        t is javax.net.ssl.SSLException -> "TLS handshake failed – check the server's certificate or turn TLS off for a plain-HTTP test server"
        else -> t.message ?: t.javaClass.simpleName
    }
}
