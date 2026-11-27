#!/usr/bin/env python3
"""
Turns a hardware report from detect_hardware.py into server_config.yaml: per-device execution-provider choices,
TensorRT settings, memory limits, and a concurrency figure for each GPU. Pure logic, no side effects beyond
writing the YAML — easy to re-run or dry-run with --print.

    python3 detect_hardware.py | python3 plan.py > server_config.yaml
    python3 plan.py --from hardware.json --out server_config.yaml --api-key <key>

Honesty note (see also inference.py's build_providers docstring): this plan picks execution providers and
memory/concurrency settings per tier. It does NOT attempt automatic model requantization or fp8 graph conversion —
ONNX Runtime's CUDA/TensorRT providers run the graph at the precision it was exported in, plus TensorRT's own
fp16/int8 kernel selection on top. fp16 TensorRT is enabled by default on datacenter/workstation-class GPUs
because it's safe and broadly faster; int8 is only ever enabled when the operator explicitly opts in with a
calibration cache, because a real calibration requires representative data per model that this script can't
responsibly fabricate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DeviceConfig, ServerConfig, save_config  # noqa: E402

# Per-tier defaults. "datacenter_grade" tiers get TensorRT + engine caching; everything else sticks to the CUDA EP
# alone, since TensorRT's cold engine-build cost usually isn't worth it on a card that's memory- or
# thermal-constrained anyway (shared workstations, laptops, older cards).
TIER_DEFAULTS: Dict[str, dict] = {
    "hopper":             dict(datacenter_grade=True,  trt_fp16=True,  workspace_frac=0.25, mem_frac=0.85, arena="kNextPowerOfTwo", per_replica_gb=1.5),
    "ampere_datacenter":  dict(datacenter_grade=True,  trt_fp16=True,  workspace_frac=0.20, mem_frac=0.85, arena="kNextPowerOfTwo", per_replica_gb=1.5),
    "ada_datacenter":     dict(datacenter_grade=True,  trt_fp16=True,  workspace_frac=0.20, mem_frac=0.85, arena="kNextPowerOfTwo", per_replica_gb=1.4),
    "ampere_workstation": dict(datacenter_grade=True,  trt_fp16=True,  workspace_frac=0.15, mem_frac=0.80, arena="kNextPowerOfTwo", per_replica_gb=1.4),
    "volta_datacenter":   dict(datacenter_grade=True,  trt_fp16=False, workspace_frac=0.15, mem_frac=0.80, arena="kNextPowerOfTwo", per_replica_gb=1.4),  # V100: fp16 TRT support is spotty pre-Turing
    "turing_datacenter":  dict(datacenter_grade=True,  trt_fp16=True,  workspace_frac=0.15, mem_frac=0.80, arena="kNextPowerOfTwo", per_replica_gb=1.3),
    "ada_consumer":       dict(datacenter_grade=False, trt_fp16=True,  workspace_frac=0.10, mem_frac=0.70, arena="kSameAsRequested", per_replica_gb=1.3),
    "ampere_consumer":    dict(datacenter_grade=False, trt_fp16=True,  workspace_frac=0.10, mem_frac=0.70, arena="kSameAsRequested", per_replica_gb=1.3),
    "turing_consumer":    dict(datacenter_grade=False, trt_fp16=False, workspace_frac=0.08, mem_frac=0.65, arena="kSameAsRequested", per_replica_gb=1.2),
    "older":              dict(datacenter_grade=False, trt_fp16=False, workspace_frac=0.0,  mem_frac=0.60, arena="kSameAsRequested", per_replica_gb=1.2),
}

# Weights resident for the "core" model set (required + recommended, see models_manifest.yaml) plus CUDA
# context/cuDNN workspace overhead — the fixed cost every GPU running this server pays once, before any
# per-request activation memory. ~1.0GB of weights (fp32 ONNX, core set) + ~1.5GB context/cuDNN/framework
# overhead, rounded up for headroom. This is an estimate, not a measurement: override per-device in the generated
# YAML if a profiling run shows otherwise, or lower per_replica_gb above if actual usage comes in lower.
RESERVED_GB = 2.5


def plan_device(index: int, name: str, tier: str, vram_mb: int) -> DeviceConfig:
    d = TIER_DEFAULTS.get(tier, TIER_DEFAULTS["older"])
    vram_gb = vram_mb / 1024
    headroom_gb = max(0.0, vram_gb - RESERVED_GB)
    concurrency = max(1, min(8, int(headroom_gb / d["per_replica_gb"])))
    workspace_mb = int(vram_mb * d["workspace_frac"]) if d["datacenter_grade"] else 0
    gpu_mem_limit_mb = int(vram_mb * d["mem_frac"])
    return DeviceConfig(
        index=index, name=name, tier=tier, vram_mb=vram_mb,
        use_tensorrt=d["datacenter_grade"],
        trt_fp16=d["trt_fp16"],
        trt_int8=False,
        trt_int8_calibration_cache=None,
        trt_workspace_mb=max(workspace_mb, 512) if d["datacenter_grade"] else 0,
        gpu_mem_limit_mb=gpu_mem_limit_mb,
        arena_extend_strategy=d["arena"],
        concurrency=concurrency,
    )


def plan_cpu_device(cores_logical: int) -> DeviceConfig:
    return DeviceConfig(
        index=-1, name="CPU", tier="cpu", vram_mb=0, use_tensorrt=False, trt_fp16=False, trt_int8=False,
        trt_int8_calibration_cache=None, trt_workspace_mb=0, gpu_mem_limit_mb=0,
        arena_extend_strategy="kSameAsRequested", concurrency=max(1, cores_logical // 2),
    )


def build_config(hardware: dict, api_key: str | None, port: int, tls_cert: str | None, tls_key: str | None) -> ServerConfig:
    gpus = hardware.get("gpus") or []
    devices: List[DeviceConfig] = [plan_device(g["index"], g["name"], g["tier"], g["vram_mb"]) for g in gpus]
    if not devices:
        devices = [plan_cpu_device(hardware.get("cpu_cores_logical", 4))]

    return ServerConfig(
        host="0.0.0.0", port=port, tls_cert=tls_cert, tls_key=tls_key, api_key=api_key,
        models_dir="models", manifest_path="models_manifest.yaml", model_set="core",
        devices=devices, trt_cache_dir="trt_cache", log_level="info",
    )


def summarize(cfg: ServerConfig) -> str:
    lines = [f"Plan: {len(cfg.devices)} device(s)"]
    for d in cfg.devices:
        if d.index < 0:
            lines.append(f"  CPU fallback — concurrency {d.concurrency}")
            continue
        trt = f"TensorRT(fp16={d.trt_fp16}, workspace={d.trt_workspace_mb}MB, cache={('trt_cache/dev'+str(d.index))})" if d.use_tensorrt else "CUDA EP only (no TensorRT)"
        lines.append(
            f"  [{d.index}] {d.name} — tier={d.tier}, {d.vram_mb}MB VRAM, mem_limit={d.gpu_mem_limit_mb}MB, "
            f"arena={d.arena_extend_strategy}, concurrency={d.concurrency}, {trt}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_file", help="hardware JSON file (default: stdin)")
    ap.add_argument("--out", default="server_config.yaml")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--tls-cert", default=None)
    ap.add_argument("--tls-key", default=None)
    ap.add_argument("--print", action="store_true", help="print the plan summary to stderr")
    args = ap.parse_args()

    raw = Path(args.from_file).read_text() if args.from_file else sys.stdin.read()
    hardware = json.loads(raw)

    cfg = build_config(hardware, args.api_key, args.port, args.tls_cert, args.tls_key)
    save_config(cfg, Path(args.out))
    if args.print:
        print(summarize(cfg), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
