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
        versionCode = 1
        versionName = "1.0.0"
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
    buildFeatures {
        compose = true
        buildConfig = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        // keep .onnx uncompressed so first-launch install is a fast straight copy
        noCompress += "onnx"
    }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("bundled-models"))
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
        jniLibs.useLegacyPackaging = false
    }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

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
}

// ---------------------------------------------------------------------------------------------
// Model bundling: downloads models once into ~/.cache (or MODELS_CACHE) and packs them into the APK.
//   -PbundleModels=false   skip bundling (app downloads models at runtime instead)
//   -PbundleEnhancer=false leave out the optional 340 MB GFPGAN model
// ---------------------------------------------------------------------------------------------
val bundleModels = (project.findProperty("bundleModels") as String?)?.toBoolean() ?: true
val bundleEnhancer = (project.findProperty("bundleEnhancer") as String?)?.toBoolean() ?: true
val modelBase = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"
val modelIds = buildList {
    addAll(listOf("scrfd_2.5g", "arcface_w600k_r50", "inswapper_128_fp16", "open_nsfw"))
    if (bundleEnhancer) add("gfpgan_1.4")
}
val modelsCache = file(System.getenv("MODELS_CACHE") ?: "${System.getProperty("user.home")}/.cache/faceswap-models")

val fetchModels by tasks.registering {
    description = "Downloads ONNX models (cached) and stages them as APK assets"
    val outDir = layout.buildDirectory.dir("bundled-models/models")
    outputs.dir(outDir)
    inputs.property("ids", modelIds)
    inputs.property("bundle", bundleModels)
    doLast {
        val out = outDir.get().asFile
        out.deleteRecursively(); out.mkdirs()
        if (!bundleModels) return@doLast
        modelsCache.mkdirs()
        for (id in modelIds) {
            val cached = File(modelsCache, "$id.onnx")
            if (!cached.exists() || cached.length() == 0L) {
                logger.lifecycle("Downloading $id.onnx …")
                val tmp = File(modelsCache, "$id.onnx.part")
                URI("$modelBase/$id.onnx").toURL().openStream().use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
                val hash = runCatching { URI("$modelBase/$id.hash").toURL().readText().trim().lowercase() }.getOrNull()
                if (hash != null && hash.length == 8) {
                    val crc = CRC32()
                    tmp.inputStream().buffered(1 shl 20).use { s -> val b = ByteArray(1 shl 20); while (true) { val n = s.read(b); if (n < 0) break; crc.update(b, 0, n) } }
                    check("%08x".format(crc.value) == hash) { "Checksum mismatch for $id" }
                }
                tmp.renameTo(cached)
            }
            cached.copyTo(File(out, "$id.onnx"), overwrite = true)
        }
    }
}
tasks.named("preBuild") { dependsOn(fetchModels) }
