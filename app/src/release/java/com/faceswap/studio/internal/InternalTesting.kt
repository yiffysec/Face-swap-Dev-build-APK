package com.faceswap.studio.internal

/** Release build: internal testing mode does not exist. */
object InternalTesting {
    const val AVAILABLE = false
    @Suppress("UNUSED_PARAMETER")
    fun checkPassphrase(input: String): Boolean = false
}
