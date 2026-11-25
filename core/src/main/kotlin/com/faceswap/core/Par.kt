package com.faceswap.core

import java.util.concurrent.ForkJoinTask
import java.util.concurrent.RecursiveAction

/**
 * Row-parallel helper for the per-pixel stages (warp, paste-back, tensor conversion, mask blur, YUV).
 *
 * Once inference runs on the NPU/GPU these CPU loops become a large share of each frame, so they are split across
 * the big cores. Small images run inline (forking costs more than it saves). Every caller writes disjoint rows,
 * so no synchronisation is needed. Uses the common ForkJoin pool, so nested calls from video workers are fine.
 */
object Par {
    /** Master switch (tests compare serial vs parallel output). */
    @Volatile var enabled = true

    /** Work below this many pixels runs on the calling thread. */
    private const val MIN_PIXELS = 48 * 1024

    private val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)

    /** Runs [body] for row ranges [from, to) covering 0 until [rows]. [pixelsPerRow] decides whether to fork. */
    fun rows(rows: Int, pixelsPerRow: Int, body: (from: Int, to: Int) -> Unit) {
        if (rows <= 0) return
        if (!enabled || threads == 1 || rows < 4 || rows.toLong() * pixelsPerRow < MIN_PIXELS) { body(0, rows); return }
        val chunks = (threads * 2).coerceAtMost(rows)
        val step = (rows + chunks - 1) / chunks
        val tasks = ArrayList<RecursiveAction>(chunks)
        var a = 0
        while (a < rows) {
            val from = a; val to = minOf(rows, a + step)
            tasks += object : RecursiveAction() { override fun compute() = body(from, to) }
            a = to
        }
        ForkJoinTask.invokeAll(tasks) // fork() from a non-pool thread submits to the common pool
    }
}
