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
generates a self-signed TLS cert, installs + starts a `systemd` service (`faceswapd`), and sets up a **Cloudflare
Tunnel** for access from outside the LAN. At the end it prints both addresses and the API key to put into the app
under **Settings → Processing → Remote server**.

Useful flags:

```bash
sudo ./setup_server.sh --models full              # also fetch extra swappers/enhancers/studio models
sudo ./setup_server.sh --port 9443 --api-key xyz   # pick your own port/key instead of a random one
sudo ./setup_server.sh --domain infer.example.com  # real cert via Let's Encrypt instead of self-signed
sudo ./setup_server.sh --no-service                # set everything up, print the run command, skip systemd
sudo ./setup_server.sh --no-cloudflared             # LAN/IP access only, skip the tunnel entirely
sudo ./setup_server.sh --cloudflared-token <TOKEN>  # stable named tunnel instead of a quick one (see below)
```

Re-run it any time — every step is idempotent. If you add or remove a GPU, re-run `setup/plan.py` (see below) to
regenerate `server_config.yaml` without re-downloading models.

## Remote access via Cloudflare Tunnel

The LAN address (`<ip>:<port>`) only works when the phone is on the same network as the server. For everywhere
else — mobile data, a different Wi-Fi — `setup_server.sh` also sets up a [Cloudflare
Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) by default, giving the
server a second, public address with no port-forwarding and no static IP needed.

Two modes:

- **Quick Tunnel** (default): zero-config, no Cloudflare account needed. `cloudflared` is downloaded automatically
  and run as a `cloudflared-quicktunnel` systemd service, which prints a `https://<random-words>.trycloudflare.com`
  address — that's what setup_server.sh prints at the end, and what re-appears each time you run
  `./get_cloudflare_url.sh`. **The hostname changes every time this service restarts** (reboot, crash, manual
  restart) — fine for personal use/testing, but re-check it with `./get_cloudflare_url.sh` after any restart, and
  update the app if it changed.
- **Named Tunnel** (`--cloudflared-token <token>`): a stable hostname that doesn't change. Create the tunnel once
  in the [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com/) (Networks → Tunnels → Create a
  tunnel), configure its public hostname and point its ingress rule at `https://localhost:<port>` (same port
  you're passing to `setup_server.sh`), then copy the tunnel's token and pass it here. Use this if you want the
  address in the app to stay valid long-term, e.g. for always-on remote access.

Either way, the app still needs the API key, and it still trusts whatever TLS certificate it finds (real,
Cloudflare-terminated TLS over the tunnel; self-signed over the LAN — see `RemoteClient.kt`'s `TrustAnyCert`).

To remove the tunnel later: `systemctl disable --now cloudflared-quicktunnel` (quick tunnel) or `cloudflared
service uninstall` (named tunnel).

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

## Acceleration

Three layers, all automatic except the last:

- **TensorRT engine caching** (datacenter/workstation/Blackwell-consumer tiers — see `setup/plan.py`'s
  `TIER_DEFAULTS`): the TensorRT execution provider compiles each model's graph into a fused, fp16 engine on first
  load and caches it under `trt_cache/dev<N>/`, so only the very first request after a model or driver upgrade pays
  the compile cost. Consumer-but-capable tiers (`blackwell_consumer` — RTX 50-series) get this too, unlike older
  `ada_consumer`/`ampere_consumer` cards, since a dedicated always-on server box is exactly the case where the
  longer first-load cost is worth it.
- **IO Binding** (always on, every call, GPU or CPU): `ModelSession` binds every input/output as a device-resident
  `OrtValue` and calls `run_with_iobinding()` instead of plain `session.run()`. This skips an implicit host↔device
  staging copy that `session.run()` does on every call — a small, stable, broadly-supported ORT API with no shape
  constraints, so it's never gated behind a config flag.
- **CUDA Graph replay** (experimental, opt-in per device via `cuda_graph: true` in `server_config.yaml`, default
  `false`): captures the full kernel-launch sequence once and replays it on every later call, which matters for
  small/fast graphs — most of the face-crop models — where kernel-launch overhead is a real fraction of total
  latency. Only applies to a model whose ONNX graph has fully static input shapes (a detector running on a
  full-resolution image doesn't qualify; a 128×128 face-crop swapper does); a one-time trial capture at load time
  either succeeds or the model silently falls back to the plain IO-bound path with a log line — never a crash,
  never stale output. This has **not been validated on real GPU hardware** as part of this change (built and
  logic-tested in an environment with no CUDA device) — treat it as experimental until you've confirmed it actually
  helps on your own box, and flip it back to `false` per-device if a model misbehaves.

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
