package com.faceswap.studio.util

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Offline crash diagnostics (test devices have no internet, so nothing is uploaded):
 *  - Java/Kotlin crashes: full stack trace written by an uncaught-exception handler.
 *  - Native crashes / low-memory kills (Android 11+): reason taken from ApplicationExitInfo on next launch.
 *  - Breadcrumbs: the last ~200 pipeline steps, written synchronously, so a native crash still shows the
 *    step it died in.
 * The report is shown in-app on the next launch (copy / share) and kept under Settings → Diagnostics.
 */
object Diagnostics {
    private const val TAG = "FaceSwapDiag"
    private lateinit var dir: File
    private lateinit var appInfo: String
    private val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val crumbs = ArrayDeque<String>()

    private val crumbFile get() = File(dir, "breadcrumbs.txt")
    private val reportFile get() = File(dir, "last_report.txt")
    private val pendingFlag get() = File(dir, "report_pending")
    private val seenExitFile get() = File(dir, "last_exit_ts")

    fun install(context: Context, versionName: String) {
        dir = File(context.filesDir, "diagnostics").apply { mkdirs() }
        appInfo = "FaceSwap Studio $versionName · ${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) · " +
            "RAM ${(context.getSystemService(ActivityManager::class.java)?.let { am -> ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem } ?: 0L) / 1_000_000} MB"

        // previous session's breadcrumbs (needed if the previous run died natively)
        val previousCrumbs = crumbFile.takeIf { it.exists() }?.readText().orEmpty()
        crumbFile.delete()

        checkExitReasons(context, previousCrumbs)

        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            runCatching {
                val sw = StringWriter(); error.printStackTrace(PrintWriter(sw))
                writeReport("Crash (exception) on thread ${thread.name}", sw.toString(), crumbFile.takeIf { it.exists() }?.readText().orEmpty())
            }
            previous?.uncaughtException(thread, error)
        }
        crumb("app start")
    }

    /** Records a pipeline step. Cheap enough to call around every heavy operation. */
    @Synchronized
    fun crumb(msg: String) {
        val line = "${fmt.format(Date())}  $msg"
        Log.i(TAG, msg)
        crumbs.addLast(line)
        if (crumbs.size > 200) crumbs.removeFirst()
        if (::dir.isInitialized) runCatching { crumbFile.writeText(crumbs.joinToString("\n")) }
    }

    private fun checkExitReasons(context: Context, previousCrumbs: String) {
        if (Build.VERSION.SDK_INT < 30) return
        val am = context.getSystemService(ActivityManager::class.java) ?: return
        val last = runCatching { am.getHistoricalProcessExitReasons(context.packageName, 0, 1).firstOrNull() }.getOrNull() ?: return
        val seen = seenExitFile.takeIf { it.exists() }?.readText()?.toLongOrNull() ?: 0L
        if (last.timestamp <= seen) return
        seenExitFile.writeText(last.timestamp.toString())
        val reason = when (last.reason) {
            ApplicationExitInfo.REASON_CRASH_NATIVE -> "Native crash (ONNX Runtime / codec / system library)"
            ApplicationExitInfo.REASON_LOW_MEMORY -> "Killed by Android for low memory"
            ApplicationExitInfo.REASON_ANR -> "App not responding (ANR)"
            ApplicationExitInfo.REASON_CRASH -> "Crash (exception)"
            ApplicationExitInfo.REASON_SIGNALED -> "Killed by signal ${last.status}"
            ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "Killed for excessive resource usage"
            else -> null
        } ?: return
        // Java crashes already wrote a full report via the handler; only add exit info.
        val existing = if (pendingFlag.exists()) reportFile.readText() else null
        val exitInfo = buildString {
            appendLine("Exit reason: $reason")
            appendLine("Description: ${last.description ?: "-"}")
            appendLine("Process memory at exit: PSS ${last.pss / 1024} MB, RSS ${last.rss / 1024} MB")
            appendLine("Time: ${fmt.format(Date(last.timestamp))}")
            if (last.reason == ApplicationExitInfo.REASON_ANR) runCatching {
                last.traceInputStream?.bufferedReader()?.use { appendLine(it.readText().take(6000)) }
            }
        }
        if (existing != null) {
            reportFile.writeText(existing + "\n\n" + exitInfo)
        } else {
            writeReport(reason, exitInfo, previousCrumbs)
        }
    }

    private fun writeReport(title: String, body: String, crumbsText: String) {
        val text = buildString {
            appendLine("=== $title ===")
            appendLine(appInfo)
            appendLine()
            appendLine(body.trim())
            appendLine()
            appendLine("--- Last steps before the crash ---")
            appendLine(crumbsText.lines().takeLast(40).joinToString("\n"))
        }
        reportFile.writeText(text)
        pendingFlag.writeText("1")
    }

    /** Report not yet shown to the user, or null. */
    fun pendingReport(): String? = if (::dir.isInitialized && pendingFlag.exists() && reportFile.exists()) reportFile.readText() else null
    fun markSeen() { if (::dir.isInitialized) pendingFlag.delete() }
    fun lastReport(): String? = if (::dir.isInitialized && reportFile.exists()) reportFile.readText() else null
    fun currentCrumbs(): String = synchronized(this) { crumbs.joinToString("\n") }
    fun reportFile(): File = reportFile
}
