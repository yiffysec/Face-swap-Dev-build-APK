# FaceSwap Studio — inference server

A generic, multi-model ONNX Runtime server for the Android app's **Remote** processing mode. The app's whole
swap/mask/enhance pipeline is unchanged whether it's running on-device or against this server — the server just
answers "run model X on these tensors" over HTTP. See `app/inference.py`'s docstring and
`core/src/main/kotlin/com/faceswap/core/Remote.kt` on the Android side for why that split is possible.

## Quick start

On a Linux box with (optionally) an NVIDIA GPU and driver already installed:

```bash
cd server
sudo ./setup_server.sh
```

This detects your hardware, writes a performance plan (`server_config.yaml`), downloads the default model set,
generates a self-signed TLS cert, and installs + starts a `systemd` service (`faceswapd`). At the end it prints
the address and API key to put into the app under **Settings → Processing → Remote server**.

Useful flags:

```bash
sudo ./setup_server.sh --models full              # also fetch extra swappers/enhancers/studio models
sudo ./setup_server.sh --port 9443 --api-key xyz   # pick your own port/key instead of a random one
sudo ./setup_server.sh --domain infer.example.com  # real cert via Let's Encrypt instead of self-signed
sudo ./setup_server.sh --no-service                # set everything up, print the run command, skip systemd
```

Re-run it any time — every step is idempotent. If you add or remove a GPU, re-run `setup/plan.py` (see below) to
regenerate `server_config.yaml` without re-downloading models.

## How hardware detection works

`setup/detect_hardware.py` reads every GPU's **CUDA compute capability and VRAM** (via `pynvml`, or `nvidia-smi`
as a fallback) and classifies it into a tier — `hopper`, `ampere_datacenter`, `ada_datacenter`,
`ampere_workstation`, `ada_consumer`, `turing_datacenter`, `volta_datacenter`, or `cpu`. Compute capability is the
primary signal because it's what actually determines what a GPU can do (Tensor Core generation, fp16/int8/fp8
support); a card's marketing name is only used as a tiebreaker where one silicon generation spans both consumer
and datacenter products (e.g. Ada Lovelace covers both the RTX 40 series and the L40/L40S). That's why an unusual
or custom-named card — like a hypothetical "A80" — still classifies correctly: it reports `compute_capability
8.0`, the same as an A100, and gets treated as one.

`setup/plan.py` turns that into `server_config.yaml`: for each GPU it decides whether to enable TensorRT (on for
every datacenter/workstation tier), fp16 (on by default — broadly safe speedup), a TensorRT workspace size and
`gpu_mem_limit` scaled to VRAM, and a **concurrency** figure — how many simultaneous requests that GPU accepts —
estimated from `(VRAM - fixed overhead) / per-replica memory`, rather than one fixed number for every card.

**What this does *not* do automatically**: int8 quantization or fp8 graph conversion. TensorRT int8 needs a
calibration run with representative data per model, which this script can't responsibly fabricate — set
`trt_int8: true` and `trt_int8_calibration_cache` by hand in `server_config.yaml` if you've built one. Hopper's
fp8 Transformer Engine support isn't something ONNX Runtime's execution providers expose for arbitrary graphs
either; Hopper gets the same fp16 TensorRT path as other datacenter tiers, just with more workspace. Nothing in
this server claims speedups it doesn't deliver.

## Architecture

One FastAPI process, one `InferenceSession` per (model, device) pair, created lazily on first use (except the
"core" model set, preloaded at startup). Dispatch picks whichever device is least busy relative to its configured
concurrency. This is deliberately **not** a multi-process or `nginx`-fronted cluster — ONNX Runtime sessions are
thread-safe for concurrent `Run()` calls, and IPC would just add overhead moving image-sized tensors between
processes for no benefit, given the client only ever needs one IP:port to talk to.

```
server/
  app/
    main.py        FastAPI app: GET /v1/health, POST /v1/infer/{model_id}
    inference.py    ModelSession (one ONNX model, one device), DeviceWorker, Dispatcher
    tensor_wire.py  Binary tensor protocol — byte-for-byte mirror of core/.../TensorWire.kt
    catalog.py      Model id -> metadata, loaded from models_manifest.yaml
    config.py       server_config.yaml <-> ServerConfig/DeviceConfig dataclasses
  setup/
    detect_hardware.py   GPU/CPU inventory -> JSON
    plan.py               Hardware JSON -> server_config.yaml
  models_manifest.yaml    Same model ids/URLs as the Android app's ModelCatalog
  download_models.py      Fetches weights into models/
  setup_server.sh          Orchestrates all of the above + systemd
  systemd/faceswapd.service.template
```

## API

```
GET /v1/health
  -> { "name", "version", "device", "models": [ { "id", "role", "inputs", "outputs" }, ... ] }

POST /v1/infer/{model_id}
  body: application/octet-stream, TensorWire-encoded named input tensors
  -> 200: application/octet-stream, TensorWire-encoded named output tensors
  -> 4xx/5xx: { "error": "..." }
```

Both routes require `Authorization: Bearer <api_key>` when `server_config.yaml` sets one (`setup_server.sh`
always generates one unless you pass `--api-key`). Leave `api_key` unset only on a network you already trust.

## Manual operation

```bash
source .venv/bin/activate
python3 setup/detect_hardware.py | python3 setup/plan.py --api-key <key> --out server_config.yaml --print
python3 download_models.py --set core
uvicorn app.main:app --host 0.0.0.0 --port 8443 --ssl-certfile tls/server.crt --ssl-keyfile tls/server.key
```

`FSW_SERVER_CONFIG` picks which config file `app/main.py` loads (default `server_config.yaml` in the current
directory) — handy for running two instances with different configs.

## Troubleshooting

- **`onnxruntime` isn't using the GPU**: check `python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"`
  lists `CUDAExecutionProvider`. If it doesn't, the `onnxruntime-gpu` wheel didn't find a compatible CUDA/cuDNN —
  usually means the NVIDIA driver is older than the wheel expects. Check `nvidia-smi`'s reported CUDA Version and
  install a matching `onnxruntime-gpu` release (see the [ONNX Runtime compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)).
- **First request after startup is slow**: with TensorRT enabled, the first call for each model builds and caches
  a TensorRT engine (`trt_cache/`); subsequent restarts reuse the cache. Pre-warm by hitting `/v1/health` right
  after startup, or `POST /v1/infer/<id>` once per model you plan to use.
- **`journalctl -u faceswapd -n 100`** for service logs; **`systemctl status faceswapd`** for current state.
- **Re-planning after a hardware change** (added/removed a GPU): `python3 setup/detect_hardware.py | python3
  setup/plan.py --api-key <existing key from server_config.yaml> --out server_config.yaml && systemctl restart
  faceswapd`.
