package com.faceswap.core

/**
 * Identity bound to a face: produced by a verification provider (third-party KYC in production).
 * [faceEmbedding] is the normalised ArcFace embedding of the verified person's live selfie.
 */
data class VerifiedIdentity(
    val faceEmbedding: FloatArray,
    val provider: String,
    val verifiedAtMillis: Long,
)

/** Plug point for identity verification. Production: wrap Persona/Onfido/Veriff etc. */
interface IdentityVerifier {
    /** Currently verified identity, or null. */
    fun current(): VerifiedIdentity?
    val providerName: String
}

/** Default verifier: nobody is verified. Explicit targets stay locked. */
object NoIdentityVerifier : IdentityVerifier {
    override fun current(): VerifiedIdentity? = null
    override val providerName = "none"
}

/** How a source face entered the app. */
enum class SourceProvenance { GALLERY, LIVE_CAMERA }

sealed class PolicyDecision {
    object Allowed : PolicyDecision()
    /** Target is explicit and no verified identity / live capture exists. */
    object ExplicitRequiresVerification : PolicyDecision()
    /** Target is explicit and at least one source face is not the verified person. */
    data class ExplicitIdentityMismatch(val bestSimilarity: Float) : PolicyDecision()
    /** Internal build: target is explicit and at least one source was not captured live in-app. */
    object ExplicitRequiresLiveCapture : PolicyDecision()

    val allowed get() = this is Allowed
}

/**
 * Explicit targets may only receive:
 *  - production: the verified user's own face (third-party ID + selfie, matched by ArcFace), or
 *  - internal builds with dev mode unlocked: faces captured live with the in-app camera.
 * Everything else is unrestricted.
 */
class ContentPolicy(
    analyserFactory: () -> ContentAnalyser,
    var verifier: IdentityVerifier = NoIdentityVerifier,
    private val explicitThreshold: Float = 0.80f,
    /** ArcFace cosine; same person typically 0.45–0.8, different people < 0.3. Strict on purpose. */
    private val identityThreshold: Float = 0.50f,
) : java.io.Closeable {
    private val analyserLazy = lazy(analyserFactory)
    private val analyser by analyserLazy

    /** Set ONLY by internal/debug builds after the developer passphrase is entered. */
    @Volatile var internalLiveCaptureMode: Boolean = false

    fun explicitProbability(target: RgbImage) = analyser.explicitProbability(target)
    fun isExplicit(target: RgbImage): Boolean = explicitProbability(target) >= explicitThreshold

    fun evaluate(targetIsExplicit: Boolean, sources: Collection<SourceFace>): PolicyDecision {
        if (!targetIsExplicit) return PolicyDecision.Allowed
        if (sources.isEmpty()) return PolicyDecision.ExplicitRequiresVerification

        // Production path: every source must be the verified person.
        val identity = verifier.current()
        if (identity != null) {
            var worst = 1f
            for (src in sources) for (e in src.embeddings) worst = minOf(worst, Geometry.cosine(e, identity.faceEmbedding))
            if (worst >= identityThreshold) return PolicyDecision.Allowed
            if (!internalLiveCaptureMode) return PolicyDecision.ExplicitIdentityMismatch(worst)
        }

        // Internal build path: every source must be a live in-app capture that passed liveness.
        if (internalLiveCaptureMode) {
            return if (sources.all { it.provenance == SourceProvenance.LIVE_CAMERA }) PolicyDecision.Allowed
            else PolicyDecision.ExplicitRequiresLiveCapture
        }
        return PolicyDecision.ExplicitRequiresVerification
    }

    fun sameIdentity(a: FloatArray, b: FloatArray) = Geometry.cosine(a, b) >= identityThreshold

    override fun close() { if (analyserLazy.isInitialized()) analyser.close() }
}

