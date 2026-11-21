package com.faceswap.studio.internal

import com.faceswap.studio.BuildConfig
import java.security.MessageDigest

/** Internal (debug) build: passphrase unlocks live-capture mode for explicit targets. */
object InternalTesting {
    const val AVAILABLE = true

    fun checkPassphrase(input: String): Boolean {
        val expected = BuildConfig.DEV_PASSPHRASE_SHA256
        if (expected.isEmpty()) return false
        val actual = MessageDigest.getInstance("SHA-256").digest(input.trim().toByteArray()).joinToString("") { "%02x".format(it) }
        return MessageDigest.isEqual(actual.toByteArray(), expected.toByteArray())
    }
}
