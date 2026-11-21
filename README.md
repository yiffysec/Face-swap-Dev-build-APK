# FaceSwap Studio (internal dev build)

On-device photo and video face swapping for Android. All inference runs locally through ONNX Runtime. No network is needed at runtime when models are bundled.

## Features

**Swapping**
- **HyperSwap 1a/1b/1c** (default, 256 px, built-in face mask) and **InSwapper 128**. Pixel boost works for any swapper (2×/4× sub-swaps).
- **YuNet** (230 KB) or **SCRFD** detection, plus **HRFFA 68-point landmarks** for more accurate alignment.
- **Masks** combined with min(), like FaceFusion: box, swapper mask, **BiSeNet face parser** (keeps hair, hands and glasses; regions are selectable), landmark **area** mask, oval.
- **Enhancers:** GPEN-256 (default), GFPGAN 1.4, GPEN-512, RestoreFormer++, CodeFormer (with fidelity control).
- **Expression restorer** (LivePortrait) re-applies the target's original expression after the swap.
- Per-face mapping, multi-photo identities, debug overlay.

**Studio tools** (FaceFusion processors, applied to the current photo with undo)
- Face editor (LivePortrait): smile, eyebrows, eyes open, lips open, gaze, pout, grim, head pitch/yaw/roll.
- Age modifier (FRAN), background blur / remove (PNG) / colour / replace (MODNet, U²-Netp fallback), tiled upscaling (SPAN ×4, Real-ESRGAN ×2/×4), stand-alone face enhancement.

**Video:** hardware decode/encode, audio passthrough, rotation-aware, landmark smoothing, and **parallel render workers**. Detection and tracking run in order while swapping and enhancing run on 2–4 threads.

**Live preview:** real-time camera swap with **MobileFaceSwap** (Baidu, AAAI-22), whose per-frame network is 364 KB.

**Performance**
- Backends: ONNX Runtime CPU / XNNPACK / NNAPI (fp16+fp32), **QNN HTP + GPU** (Snapdragon flavor), and **LiteRT GPU** (TFLite versions of models, converted and verified in CI).
- *Tune for this device* benchmarks every installed model on every backend in an isolated process and checks each backend's output against CPU. Auto mode then uses the fastest correct backend per model.
- QNN EP context caching (the HTP graph compiles once), big-core thread count, ADPF performance hints and sustained-performance mode.
- Optional **root boost** (Pixel): CPU governors and memory-latency devfreq set to performance during video, restored afterwards. Thermal limits are never touched.
- Diagnostics: offline crash reports with exit reasons and pipeline step logs.

## APK flavors

| Flavor | For | Runtime |
|---|---|---|
| `universal` | Pixel 8 Pro and other devices | ORT (CPU/XNNPACK/NNAPI) + LiteRT GPU |
| `snapdragon` | Galaxy S24 Ultra | ORT with QNN EP (Hexagon HTP V75 / Adreno) + LiteRT GPU |

Core models (~0.85 GB) are bundled in both. Extras ship as offline **model packs** (`models-*.zip` on the release). Import them with *Models → Import* over USB.

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

`tools/prepare_models.py` fixes dynamic input shapes (SCRFD 640², ArcFace and open_nsfw batch 1) so NNAPI can offload whole graphs. CI runs it automatically; run it on `~/.cache/faceswap-models` for local builds.

GitHub Actions (`.github/workflows/build.yml`) runs on every push. It downloads and caches all models, fixes their static shapes, converts them to TFLite for LiteRT GPU (best effort), runs the core tests, builds both flavors, packs the extra models, and publishes everything to a rolling pre-release named `internal-<branch>`.

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
