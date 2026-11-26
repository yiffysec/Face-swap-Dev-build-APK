plugins {
    id("org.jetbrains.kotlin.jvm")
}

val ortVersion = "1.22.0"

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    // API only: Android app supplies onnxruntime-android, JVM tests use the desktop runtime (same ai.onnxruntime API).
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.13.2")
}

tasks.test {
    useJUnit()
    systemProperty("models.dir", System.getProperty("models.dir") ?: (System.getenv("FACESWAP_MODELS") ?: ""))
    // calibration capture for tools/quantize_qnn.py (CI): -Dcalib.dir=… [-Dcalib.samples=48 -Dcalib.mb=96]
    for (k in listOf("calib.dir", "calib.samples", "calib.mb", "calib.skip")) System.getProperty(k)?.let { systemProperty(k, it) }
    maxHeapSize = System.getProperty("test.heap") ?: "2g"
    testLogging { showStandardStreams = true; events("passed", "failed", "skipped") }
}
