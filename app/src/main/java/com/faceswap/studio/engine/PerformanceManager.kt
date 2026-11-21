package com.faceswap.studio.engine

import android.app.Activity
import android.content.Context
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process
import com.faceswap.studio.util.Diagnostics
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Peak-performance controls for long jobs (video, batch, live preview):
 *
 *  - Non-root (both phones): Window sustained-performance mode, ADPF performance-hint session for the worker threads
 *    (tells the scheduler our per-frame target so it boosts big cores instead of guessing), urgent-display priority.
 *  - Root (Pixel 8 Pro): CPU governors → "performance" on every policy and min freq = max freq for the big clusters
 *    while a job runs, then everything is restored exactly. Thermal protection is never touched – the SoC still
 *    throttles if it gets hot, which protects the hardware and battery.
 */
class PerformanceManager(private val context: Context) {
    private var hint: PerformanceHintManager.Session? = null
    private val saved = LinkedHashMap<String, String>()
    @Volatile private var rootActive = false

    val rootAvailable: Boolean by lazy {
        listOf("/system/bin/su", "/system/xbin/su", "/sbin/su", "/debug_ramdisk/su", "/data/adb/magisk").any { File(it).exists() } ||
            runCatching { Runtime.getRuntime().exec(arrayOf("which", "su")).inputStream.bufferedReader().readText().isNotBlank() }.getOrDefault(false)
    }

    /** Call when a heavy job starts. [targetFrameMs] = desired per-frame time (for video ~ the measured average). */
    fun begin(activity: Activity?, threads: List<Int>, targetFrameMs: Long, useRoot: Boolean) {
        activity?.runOnUiThread { runCatching { activity.window.setSustainedPerformanceMode(true) } }
        runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY) }
        if (Build.VERSION.SDK_INT >= 31 && threads.isNotEmpty()) runCatching {
            val phm = context.getSystemService(PerformanceHintManager::class.java)
            hint = phm?.createHintSession(threads.toIntArray(), TimeUnit.MILLISECONDS.toNanos(targetFrameMs.coerceAtLeast(8)))
            Diagnostics.crumb("perf: ADPF hint session ${if (hint != null) "on" else "unavailable"} (${threads.size} threads, ${targetFrameMs} ms)")
        }
        if (useRoot && rootAvailable) boostRoot()
    }

    /** Report how long the last unit of work took (keeps ADPF's boost calibrated). */
    fun report(actualMs: Long) {
        runCatching { hint?.reportActualWorkDuration(TimeUnit.MILLISECONDS.toNanos(actualMs)) }
    }

    fun end(activity: Activity?) {
        activity?.runOnUiThread { runCatching { activity.window.setSustainedPerformanceMode(false) } }
        runCatching { hint?.close() }; hint = null
        if (rootActive) restoreRoot()
    }

    // ------------------------------------------------------------------------------------------------- root

    private fun su(script: String): String = runCatching {
        val p = Runtime.getRuntime().exec(arrayOf("su", "-c", script))
        val out = p.inputStream.bufferedReader().readText()
        p.waitFor(10, TimeUnit.SECONDS)
        out
    }.getOrDefault("")

    private fun boostRoot() {
        val policies = su("ls -d /sys/devices/system/cpu/cpufreq/policy*").lines().filter { it.isNotBlank() }
        if (policies.isEmpty()) { Diagnostics.crumb("perf: root not granted / no cpufreq policies"); return }
        val script = StringBuilder()
        for (p in policies) {
            val gov = su("cat $p/scaling_governor").trim()
            val minF = su("cat $p/scaling_min_freq").trim()
            val maxF = su("cat $p/cpuinfo_max_freq").trim()
            if (gov.isNotEmpty()) saved["$p/scaling_governor"] = gov
            if (minF.isNotEmpty()) saved["$p/scaling_min_freq"] = minF
            script.append("echo performance > $p/scaling_governor 2>/dev/null; ")
            if (maxF.isNotEmpty()) script.append("echo $maxF > $p/scaling_min_freq 2>/dev/null; ")
        }
        // Tensor (zuma) memory-latency governors: keep DDR clocks up so model weights stream faster.
        val memlat = su("ls -d /sys/class/devfreq/*mif* /sys/class/devfreq/*bci* 2>/dev/null").lines().filter { it.isNotBlank() }
        for (d in memlat) {
            val g = su("cat $d/governor").trim()
            if (g.isNotEmpty()) { saved["$d/governor"] = g; script.append("echo performance > $d/governor 2>/dev/null; ") }
        }
        su(script.toString())
        rootActive = true
        Diagnostics.crumb("perf: root boost on (${policies.size} cpu policies, ${memlat.size} devfreq nodes)")
    }

    private fun restoreRoot() {
        val script = saved.entries.joinToString("; ") { (path, v) -> "echo $v > $path 2>/dev/null" }
        if (script.isNotEmpty()) su(script)
        saved.clear(); rootActive = false
        Diagnostics.crumb("perf: root boost restored")
    }
}
