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

**Performance** (see `research/performance-game-plan.md` in the project for the research behind this)
- Backends: ONNX Runtime 1.32 CPU / XNNPACK / NNAPI (Tensor TPU), Qualcomm's **QNN plugin EP 2.8** (QAIRT 2.52) on the Hexagon NPU in fp16 **and calibrated w8a16 INT8**, QNN GPU, and **LiteRT GPU** in fp16, fp16 + fp32 accumulation, and fp32.
- *Settings → Performance → Tune* benchmarks every installed model on every backend the chip offers, in an isolated process, on a **real face input** captured in CI, and checks *all* outputs against fp32 CPU. NPU candidates must prove full offload first (CPU fallback disabled); partial offloads are measured and labelled. Auto then uses the fastest correct backend per model. NNAPI is never tried on Snapdragon (it has no NNAPI driver there).
- NPU clocks: burst for single photos, sustained high-performance for video and live. The compiled NPU graph is cached and reused, and rebuilt automatically when the model or app changes.
- Pipeline: row-parallel pixel stages (warp, paste-back, tensor conversion, mask blur, YUV), one face-parser pass per face, **instant draft preview** while High/Ultra and the enhancer render, **smart enhance** for video (skips faces already as sharp as the original), non-blocking video progress, ADPF hint sessions fed with real frame times.
- Optional **root boost** (Pixel): CPU governors and memory-latency devfreq set to performance during video, restored afterwards. Thermal limits are never touched.
- Diagnostics: offline crash reports with exit reasons and pipeline step logs.

## APK flavors

| Flavor | For | Runtime |
|---|---|---|
| `universal` | Pixel 8 Pro and other devices | ORT (CPU/XNNPACK/NNAPI) + LiteRT GPU |
| `snapdragon` | Galaxy S24 Ultra | + Qualcomm QNN plugin EP (Hexagon HTP V75 / Adreno) and the INT8 NPU twins of the core models |

Core models (~0.85 GB) are bundled in both. Extras ship as offline **model packs** (`models-*.zip` on the release; `models-npu-int8.zip` holds the INT8 twins of the extras, Snapdragon only). Import them with *Settings → Models → Import* over USB. After installing a new build, run *Settings → Performance → Tune*.

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

GitHub Actions (`.github/workflows/build.yml`) runs on every push. It downloads and caches all models, fixes their static shapes, converts them to TFLite for LiteRT GPU, captures calibration inputs from the real pipeline (`CalibrationDump` test) and builds **quality-gated w8a16 INT8 twins** for the NPU (`tools/quantize_qnn.py`: kept only if worst cosine ≥ 0.998 and PSNR ≥ 40 dB against fp32; `qnn-report.txt` lists the result per model), runs the core tests, builds both flavors, packs the extra models, and publishes everything to a rolling pre-release named `internal-<branch>`. GPU twins and INT8 twins are cached, so only the first run pays for conversion.

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
  Engine.kt       FaceSwapEngine: analyze/render stages, draft render, pixel boost, tracker, mappings
  Processors.kt   FRAN age, background, frame upscaling, LivePortrait expression + face editor
  Par.kt          row-parallel pixel loops · TensorFile.kt real-input tensor files (calibration, tuner)
  Policy.kt       explicit-target identity / live-capture policy
app/    Android (Compose)
  data/   ModelManager (download/bundle/import, twins), FaceLibrary, PngText, Settings
  engine/ Backends + tuner (QNN plugin, LiteRT, NNAPI), EngineHolder, PerformanceManager (ADPF, root)
  video/  VideoSwapper (MediaCodec pipeline, parallel render workers, YUV conversion)
  ui/     Photo · Video · Live · Faces · Settings (Swapping, Masks, Performance, Models, Diagnostics)
tools/  prepare_models.py (static shapes) · convert_tflite.py (GPU twins) · quantize_qnn.py (INT8 NPU twins)
  src/debug|release/.../internal/InternalTesting.kt  passphrase gate (absent from release)
```
