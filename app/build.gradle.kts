import java.util.Properties
import java.security.MessageDigest
import java.net.URI
import java.util.zip.CRC32

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "com.faceswap.studio"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.faceswap.studio"
        minSdk = 28
        targetSdk = 36
        versionCode = 3
        versionName = "2.0.1"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".internal"
            versionNameSuffix = "-internal"
            // Internal-testing passphrase: set DEV_PASSPHRASE=... in local.properties (not committed). Only its SHA-256 is compiled in.
            val pass = localProps.getProperty("DEV_PASSPHRASE", "faceswap-internal")
            val sha = MessageDigest.getInstance("SHA-256").digest(pass.toByteArray()).joinToString("") { "%02x".format(it) }
            buildConfigField("String", "DEV_PASSPHRASE_SHA256", "\"$sha\"")
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            buildConfigField("String", "DEV_PASSPHRASE_SHA256", "\"\"")
            signingConfig = signingConfigs.getByName("debug") // replace with your release keystore
        }
    }
    // universal: ONNX Runtime (CPU/XNNPACK/NNAPI) + LiteRT GPU – Pixel 8 Pro and everything else.
    // snapdragon: + Qualcomm's QNN plugin EP (Hexagon NPU, int8 QDQ models) – Galaxy S24 Ultra.
    flavorDimensions += "soc"
    productFlavors {
        create("universal") { dimension = "soc"; isDefault = true }
        create("snapdragon") { dimension = "soc"; versionNameSuffix = "-snapdragon" }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        // models stay uncompressed: first-launch install is a straight copy and twins can be size-checked (openFd)
        noCompress += listOf("onnx", "tflite", "bin", "json")
    }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("bundled-models"))
    // calibrated int8 (w8a16 QDQ) twins only help the Hexagon NPU → Snapdragon APK only
    sourceSets["snapdragon"].assets.srcDir(layout.buildDirectory.dir("bundled-models-npu"))
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
        // QNN's HTP skeleton libraries are loaded by the DSP from a real file path → extract native libs.
        jniLibs.useLegacyPackaging = true
    }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(project(":core"))
    // One ONNX Runtime for both flavors; Snapdragon adds Qualcomm's QNN plugin EP (v2.8: multi-core NPU, fp16 overflow
    // clamp, faster context loading) + the matching QNN SDK runtime libraries (QAIRT 2.52).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")
    "snapdragonImplementation"("com.qualcomm.qti:onnxruntime-android-qnn:2.8.0")
    "snapdragonImplementation"("com.qualcomm.qti:qnn-runtime:2.52.0")

    implementation("com.google.ai.edge.litert:litert:2.4.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.4.0")

    val camerax = "1.5.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")

    val bom = platform("androidx.compose:compose-bom:2026.01.01")
    implementation(bom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.activity:activity-compose:1.11.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.4")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.4")
    implementation("androidx.exifinterface:exifinterface:1.4.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    // Remote processing mode: talks to server/ (see RemoteClient.kt) over plain HTTP(S) with a custom binary body.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}

// ---------------------------------------------------------------------------------------------
// Model bundling: downloads models once into ~/.cache (or MODELS_CACHE) and packs them into the APK.
//   -PbundleModels=false   skip bundling (app downloads / imports models instead)
//   -PmodelSet=full        bundle every model (very large APK)
// ---------------------------------------------------------------------------------------------
val bundleModels = (project.findProperty("bundleModels") as String?)?.toBoolean() ?: true
// "core" (default, ~0.85 GB) or "full" (everything; >3 GB, only for sideloading on devices with lots of storage)
val modelSet = (project.findProperty("modelSet") as String?) ?: "core"
fun ff(release: String, id: String) = "https://huggingface.co/facefusion/models-$release/resolve/main/$id"
val coreModels = linkedMapOf(
    "yunet_2023_mar" to ff("3.4.0", "yunet_2023_mar.onnx"),
    "scrfd_2.5g" to ff("3.0.0", "scrfd_2.5g.onnx"),
    "arcface_w600k_r50" to ff("3.0.0", "arcface_w600k_r50.onnx"),
    "hyperswap_1a_256" to ff("3.3.0", "hyperswap_1a_256.onnx"),
    "open_nsfw" to ff("3.0.0", "open_nsfw.onnx"),
    "hrffa" to ff("3.9.0", "hrffa.onnx"),
    "bisenet_resnet_18" to ff("3.1.0", "bisenet_resnet_18.onnx"),
    "gpen_bfr_256" to ff("3.0.0", "gpen_bfr_256.onnx"),
    "modnet" to ff("3.5.0", "modnet.onnx"),
    "span_kendata_x4" to ff("3.0.0", "span_kendata_x4.onnx"),
    "real_esrgan_x2_fp16" to ff("3.0.0", "real_esrgan_x2_fp16.onnx"),
    // Qualcomm AI Hub zip (ONNX + external data) – merged into one file by tools/fetch_qualcomm.py
    "real_esrgan_general_x4v3" to "qualcomm:",
)
val extraModels = linkedMapOf(
    "inswapper_128_fp16" to ff("3.0.0", "inswapper_128_fp16.onnx"),
    "hyperswap_1b_256" to ff("3.3.0", "hyperswap_1b_256.onnx"),
    "hyperswap_1c_256" to ff("3.3.0", "hyperswap_1c_256.onnx"),
    "gfpgan_1.4" to ff("3.0.0", "gfpgan_1.4.onnx"),
    "gpen_bfr_512" to ff("3.0.0", "gpen_bfr_512.onnx"),
    "restoreformer_plus_plus" to ff("3.0.0", "restoreformer_plus_plus.onnx"),
    "codeformer" to ff("3.0.0", "codeformer.onnx"),
    "fran" to ff("3.6.0", "fran.onnx"),
    "u2netp" to ff("3.5.0", "u2netp.onnx"),
    "real_esrgan_x4_fp16" to ff("3.0.0", "real_esrgan_x4_fp16.onnx"),
    "live_portrait_feature_extractor" to ff("3.0.0", "live_portrait_feature_extractor.onnx"),
    "live_portrait_motion_extractor" to ff("3.0.0", "live_portrait_motion_extractor.onnx"),
    "live_portrait_generator" to ff("3.0.0", "live_portrait_generator.onnx"),
    "live_portrait_stitcher" to ff("3.0.0", "live_portrait_stitcher.onnx"),
    "live_portrait_eye_retargeter" to ff("3.0.0", "live_portrait_eye_retargeter.onnx"),
    "live_portrait_lip_retargeter" to ff("3.0.0", "live_portrait_lip_retargeter.onnx"),
    "MobileFaceSwapUNetRuntime" to "https://huggingface.co/ZZZHHHHW/mobilefaceswap/resolve/main/MobileFaceSwapUNetRuntime.onnx",
    "BuildFaceSwap" to "https://huggingface.co/ZZZHHHHW/mobilefaceswap/resolve/main/BuildFaceSwap.onnx",
    "mobilefaceswap_arcface" to "https://huggingface.co/ZZZHHHHW/mobilefaceswap/resolve/main/arcface.onnx",
)
val bundled = if (modelSet == "full") coreModels + extraModels else coreModels
val modelsCache = file(System.getenv("MODELS_CACHE") ?: "${System.getProperty("user.home")}/.cache/faceswap-models")

fun download(id: String, url: String, cache: File) {
    val cached = File(cache, "$id.onnx")
    if (cached.exists() && cached.length() > 0L) return
    if (url.startsWith("qualcomm:")) {
        logger.lifecycle("Fetching $id (Qualcomm AI Hub) …")
        val p = ProcessBuilder("python3", rootProject.file("tools/fetch_qualcomm.py").absolutePath, cache.absolutePath).inheritIO().start()
        check(p.waitFor() == 0 && cached.exists()) { "tools/fetch_qualcomm.py failed for $id (needs python3 + onnx)" }
        return
    }
    logger.lifecycle("Downloading $id.onnx …")
    val tmp = File(cache, "$id.onnx.part")
    URI(url).toURL().openStream().use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
    if (url.contains("/facefusion/")) {
        val hash = runCatching { URI(url.removeSuffix(".onnx") + ".hash").toURL().readText().trim().lowercase() }.getOrNull()
        if (hash != null && hash.length == 8) {
            val crc = CRC32()
            tmp.inputStream().buffered(1 shl 20).use { s -> val b = ByteArray(1 shl 20); while (true) { val n = s.read(b); if (n < 0) break; crc.update(b, 0, n) } }
            check("%08x".format(crc.value) == hash) { "Checksum mismatch for $id" }
        }
    }
    tmp.renameTo(cached)
}

val fetchModels by tasks.registering {
    description = "Downloads ONNX models (cached) and stages the bundled set (+ TFLite twins) as APK assets"
    val outDir = layout.buildDirectory.dir("bundled-models/models")
    outputs.dir(outDir)
    // always re-stage from the cache: tools/prepare_models.py / convert_tflite.py may have rewritten cached models
    outputs.upToDateWhen { false }
    val npuDir = layout.buildDirectory.dir("bundled-models-npu/models")
    outputs.dir(npuDir)
    doLast {
        val out = outDir.get().asFile
        val npu = npuDir.get().asFile
        out.deleteRecursively(); out.mkdirs()
        npu.deleteRecursively(); npu.mkdirs()
        if (!bundleModels) return@doLast
        modelsCache.mkdirs()
        for ((id, url) in bundled) {
            download(id, url, modelsCache)
            File(modelsCache, "$id.onnx").copyTo(File(out, "$id.onnx"), overwrite = true)
            // GPU twins + real sample inputs (tuner) for every flavor
            for (ext in listOf("tflite", "tflite.json", "sample.bin")) File(modelsCache, "$id.$ext").takeIf { it.exists() }?.copyTo(File(out, "$id.$ext"), overwrite = true)
            File(modelsCache, "$id.qdq.onnx").takeIf { it.exists() }?.copyTo(File(npu, "$id.qdq.onnx"), overwrite = true)
        }
    }
}

/** Downloads every non-bundled model into the cache (CI zips them into offline model packs). */
val fetchExtraModels by tasks.registering {
    doLast { modelsCache.mkdirs(); for ((id, url) in extraModels) download(id, url, modelsCache) }
}
tasks.named("preBuild") { dependsOn(fetchModels) }
