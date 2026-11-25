package com.faceswap.studio.engine

import android.content.Context
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process
import com.faceswap.core.PerfHints
import com.faceswap.studio.util.Diagnostics
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Peak-performance controls for long jobs (video, live preview, batch):
 *
 *  - Both phones: NPU clock mode switches to "sustained high performance" ([PerfHints.sustained]) so the Hexagon NPU
 *    doesn't burst and throttle; an ADPF performance-hint session over the pipeline threads reports the real
 *    per-frame time so the scheduler places them on big cores at the right clocks; urgent-display priority.
 *  - Root (Pixel 8 Pro): CPU governors → "performance" with min = max frequency, plus the memory-latency devfreq
 *    governors, while the job runs; everything is restored exactly afterwards. Thermal protection is never touched –
 *    the SoC still throttles when hot, which protects the hardware and battery.
 *
 * Jobs nest safely: [beginJob]/[endJob] are reference counted, and a second hint session replaces the first.
 */
class PerformanceManager(private val context: Context) {
    private var hint: PerformanceHintManager.Session? = null
    private val saved = LinkedHashMap<String, String>()
    @Volatile private var rootActive = false
    private var jobs = 0
    private var lastReportNs = 0L

    val rootAvailable: Boolean by lazy {
        listOf("/system/bin/su", "/system/xbin/su", "/sbin/su", "/debug_ramdisk/su", "/data/adb/magisk").any { File(it).exists() } ||
            runCatching { Runtime.getRuntime().exec(arrayOf("which", "su")).inputStream.bufferedReader().readText().isNotBlank() }.getOrDefault(false)
    }

    val adpfSupported: Boolean get() = Build.VERSION.SDK_INT >= 31 && runCatching {
        context.getSystemService(PerformanceHintManager::class.java)?.preferredUpdateRateNanos?.let { it > 0 } ?: false
    }.getOrDefault(false)

    /** A sustained job (video / live) starts. [useRoot] also pins CPU + memory clocks when root is available. */
    @Synchronized
    fun beginJob(useRoot: Boolean) {
        jobs++
        if (jobs == 1) {
            PerfHints.sustained = true
            if (useRoot && rootAvailable) boostRoot()
        }
    }

    @Synchronized
    fun endJob() {
        if (jobs == 0) return
        jobs--
        if (jobs == 0) {
            PerfHints.sustained = false
            closeHint()
            if (rootActive) restoreRoot()
        }
    }

    /** (Re)creates the ADPF hint session for [threads] with a per-frame target of [targetFrameMs]. */
    @Synchronized
    fun hint(threads: List<Int>, targetFrameMs: Long) {
        closeHint()
        runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY) }
        if (Build.VERSION.SDK_INT >= 31 && threads.isNotEmpty()) runCatching {
            val phm = context.getSystemService(PerformanceHintManager::class.java)
            hint = phm?.createHintSession(threads.distinct().toIntArray(), TimeUnit.MILLISECONDS.toNanos(targetFrameMs.coerceAtLeast(8)))
            lastReportNs = 0L
            Diagnostics.crumb("perf: ADPF hint session ${if (hint != null) "on" else "unavailable"} (${threads.size} threads, $targetFrameMs ms)")
        }
    }

    /** Call once per finished frame: reports the wall time since the previous frame. */
    fun frameDone() {
        val now = System.nanoTime()
        val prev = lastReportNs
        lastReportNs = now
        if (prev == 0L) return
        if (Build.VERSION.SDK_INT >= 31) runCatching { hint?.reportActualWorkDuration(now - prev) }
    }

    @Synchronized
    private fun closeHint() { if (Build.VERSION.SDK_INT >= 31) runCatching { hint?.close() }; hint = null }

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
        // min freq first (it must not exceed the restored governor's range), then governors
        val entries = saved.entries.sortedBy { if (it.key.endsWith("scaling_min_freq")) 0 else 1 }
        val script = entries.joinToString("; ") { (path, v) -> "echo $v > $path 2>/dev/null" }
        if (script.isNotEmpty()) su(script)
        saved.clear(); rootActive = false
        Diagnostics.crumb("perf: root boost restored")
    }
}
