# FaceSwap Studio (internal dev build)

On-device photo and video face swapping for Android. All inference runs locally through ONNX Runtime. No network is needed at runtime when models are bundled.

## Features

- **Photo swap** – multi-face, before/after compare, save to `Pictures/FaceSwapStudio`, share.
- **Video swap** – hardware decode/encode (MediaCodec H.264), original audio passthrough, rotation-aware, landmark smoothing tracker to cut flicker, live preview + ETA, cancel.
- **Face library** – reusable source faces. A camera capture is stored as a PNG with `FS:Live=true` tEXt metadata in app-private storage.
- **Per-face mapping** – assign a different library face to each person in the target (matched by identity, so it follows people through a video).
- **Multi-photo sources** – select several library entries of the same person and they are averaged into one identity for a better likeness.
- **Quality presets** – Fast (128px), Balanced (2× pixel boost), High (boost + GFPGAN), Ultra (4× boost + GFPGAN).
- **Blending controls** – box / oval / combined mask, feather, forehead/chin trim, swap strength, colour match, sharpen, enhancer blend.
- **Acceleration** – CPU, XNNPACK, NNAPI (automatic CPU fallback), thread count.
- **Offline** – models bundled into the APK at build time, or imported from storage over USB.

## Content policy (explicit targets)

The bundled open_nsfw classifier only gates **explicit targets**. Everything else is unrestricted, with no watermark and no labels.

| Build | Explicit target allowed when… |
|---|---|
| Release | every source face matches the user's **verified identity** (`IdentityVerifier`; plug in the KYC provider) |
| Internal (debug) | internal testing is unlocked with the passphrase **and** every source face was captured with the camera (LIVE in the library) |

Video frames flagged explicit that don't meet the rule pass through unchanged.

### Internal testing passphrase
Set `DEV_PASSPHRASE=...` in `local.properties` locally, or as the `DEV_PASSPHRASE` repository secret for Actions. Only its SHA-256 is compiled into debug builds. Default if unset: `faceswap-internal`.

### Identity verification (production)
Implement `com.faceswap.core.IdentityVerifier` around your KYC provider. Its `current()` returns the ArcFace embedding of the verified live selfie (use `FaceSwapEngine.embed`). Then set `engine.verifier = ...` in `FaceSwapApp.onCreate`.

## Building

```bash
# models are downloaded once into ~/.cache/faceswap-models and bundled into the APK
./gradlew :app:assembleDebug                         # ~935 MB, fully offline
./gradlew :app:assembleDebug -PbundleEnhancer=false  # ~600 MB, no GFPGAN
./gradlew :app:assembleDebug -PbundleModels=false    # ~116 MB, models fetched in-app
./gradlew :core:test -Dmodels.dir=$HOME/.cache/faceswap-models   # JVM tests with real models
```

GitHub Actions (`.github/workflows/build.yml`) builds on every push, caches the models, runs the core tests and uploads the APK as an artifact. Use **Run workflow** to toggle model bundling.

Requirements: JDK 17+, Android SDK platform 36. minSdk 28 (Android 9).

## Models

Downloaded from the FaceFusion mirror at `huggingface.co/facefusion/models-3.0.0`, CRC32-verified.

| Model | Purpose | Size | License |
|---|---|---|---|
| scrfd_2.5g | face + 5-point landmark detection | 3 MB | InsightFace (non-commercial) |
| arcface_w600k_r50 | identity embeddings | 174 MB | InsightFace (non-commercial) |
| inswapper_128_fp16 | face swap | 278 MB | InsightFace (non-commercial) |
| open_nsfw | content classification | 24 MB | BSD-2 |
| gfpgan_1.4 | face restoration (optional) | 340 MB | Apache-2.0 |

⚠️ The InsightFace models are licensed for **non-commercial research only**. A commercial release needs a commercial licence from InsightFace or a swap/recognition model with commercial-friendly terms.

## Layout

```
core/   pure Kotlin/JVM pipeline (testable without a device)
  Image.kt        warping, masks, blending, colour transfer
  Geometry.kt     landmark alignment (Umeyama), NMS, templates
  Models.kt       SCRFD, ArcFace, InSwapper, GFPGAN, content analyser
  OnnxInitializerReader.kt  streams the inswapper emap matrix out of the .onnx
  Engine.kt       FaceSwapEngine, pixel boost, tracker, mappings
  Policy.kt       explicit-target identity / live-capture policy
app/    Android (Compose)
  data/   ModelManager (download/bundle/import), FaceLibrary, PngText, Settings
  engine/ EngineHolder (session options, accelerators)
  video/  VideoSwapper (MediaCodec pipeline, YUV conversion)
  ui/     screens
  src/debug|release/.../internal/InternalTesting.kt  passphrase gate (absent from release)
```
