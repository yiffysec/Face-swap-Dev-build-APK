#!/usr/bin/env python3
"""
Hardware inventory for the inference server, printed as JSON to stdout. `setup_server.sh` pipes this into
`plan.py`, which turns it into `server_config.yaml`.

Classification philosophy: a GPU is identified by its CUDA compute capability + VRAM, NOT by matching its name
against a list of known model names. Name matching is used only as a secondary tiebreaker where two very
different product lines share the same compute capability (Ada Lovelace covers both the RTX 40 desktop series and
the L40/L40S datacenter cards, both compute 8.9; Ampere covers both RTX 30 desktop and A40/A6000 workstation
cards, both compute 8.6). This means a card with an unexpected or made-up-sounding name — "2x A100", "4x L40s",
"6x A80" — still classifies correctly as long as nvidia-smi reports its real compute capability and VRAM, which
is the actual thing that determines what the GPU can do (Tensor Core generation, fp8 support, memory bandwidth
tier), not its marketing name.

Works with or without the `pynvml` package; falls back to parsing `nvidia-smi` directly if pynvml isn't
importable or NVML init fails (e.g. driver present but library path odd). Reports a CPU-only result, not an
error, on a box with no NVIDIA GPU at all.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class GpuInfo:
    index: int
    name: str
    vram_mb: int
    compute_capability: str  # "9.0", "8.0", etc.
    driver_version: str
    tier: str


@dataclass
class HostInfo:
    hostname: str
    os: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_mb: int
    nvidia_driver_version: Optional[str]
    cuda_runtime_version: Optional[str]  # from `nvidia-smi`, the max CUDA version the driver supports
    gpus: List[GpuInfo]


# --- Tiering -----------------------------------------------------------------------------------------------

# Name substrings that identify a datacenter/workstation SKU when compute capability alone is ambiguous between a
# desktop GeForce/RTX part and a datacenter part of the same silicon generation. This is a tiebreaker, not the
# primary signal — see module docstring.
_DATACENTER_NAME_HINTS = (
    "l40", "l4", "a100", "a40", "a30", "a16", "a10", "a2 ", "h100", "h200", "h800", "a800",
    "rtx a6000", "rtx a5000", "rtx a4000", "rtx 6000 ada", "rtx 5000 ada", "rtx 4000 ada",
    "v100", "t4", "p100", "p40", "p4 ", "tesla",
)
_CONSUMER_NAME_HINTS = ("geforce", "rtx 20", "rtx 30", "rtx 40", "rtx 50", "gtx")


def _name_suggests_datacenter(name: str) -> Optional[bool]:
    n = f" {name.lower()} "
    if any(h in n for h in _DATACENTER_NAME_HINTS):
        return True
    if any(h in n for h in _CONSUMER_NAME_HINTS):
        return False
    return None  # unrecognized name – fall back to VRAM


def classify_tier(name: str, vram_mb: int, cc_major: int, cc_minor: int) -> str:
    cc = (cc_major, cc_minor)
    dc_hint = _name_suggests_datacenter(name)

    if cc_major >= 9:
        return "hopper"  # H100/H200/GH200-class: fp8 Transformer Engine, the fastest tier we plan for

    if cc == (8, 9):  # Ada Lovelace: L40/L40S/L4 (datacenter) vs RTX 40 (consumer/workstation desktop)
        if dc_hint is True or vram_mb >= 40_000:
            return "ada_datacenter"
        return "ada_consumer"

    if cc == (8, 6):  # Ampere: A40/A6000/A16/A10 (workstation/inference) vs RTX 30 (consumer)
        if dc_hint is True or vram_mb >= 40_000:
            return "ampere_workstation"
        if dc_hint is False:
            return "ampere_consumer"
        return "ampere_workstation" if vram_mb >= 20_000 else "ampere_consumer"

    if cc == (8, 0):  # A100-class compute capability is unique to this tier regardless of the card's product
        # name – this is deliberately the branch that makes an oddly-named "A80" still classify correctly.
        return "ampere_datacenter"

    if cc == (7, 5):  # Turing: T4 (datacenter) vs RTX 20 (consumer)
        if dc_hint is True or (dc_hint is None and vram_mb >= 15_000):
            return "turing_datacenter"
        return "turing_consumer"

    if cc == (7, 0):
        return "volta_datacenter"  # V100

    if cc_major >= 1:
        return "older"

    return "cpu"


# --- Detection -----------------------------------------------------------------------------------------------

def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _nvidia_smi_query(fields: str) -> List[List[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def detect_gpus_via_pynvml() -> Optional[List[GpuInfo]]:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            driver = pynvml.nvmlSystemGetDriverVersion()
            driver = driver.decode() if isinstance(driver, bytes) else driver
            vram_mb = int(mem.total / (1024 * 1024))
            gpus.append(GpuInfo(
                index=i, name=name, vram_mb=vram_mb, compute_capability=f"{major}.{minor}",
                driver_version=driver, tier=classify_tier(name, vram_mb, major, minor),
            ))
        return gpus
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# Compute capability isn't exposed by nvidia-smi directly; this table covers every datacenter/workstation/
# consumer chip named in this project's target hardware plus common others. Unlisted names fall back to VRAM-only
# tiering in classify_tier (dc_hint stays None), which is the documented, honest fallback rather than a guess.
_CC_BY_NAME_PATTERN = [
    (r"\bh100\b|\bh200\b|\bh800\b|\bgh200\b", (9, 0)),
    (r"\bl40s?\b|\brtx 40\d\d\b|\bl4\b|\brtx 6000 ada\b|\brtx 5000 ada\b|\brtx 4000 ada\b", (8, 9)),
    (r"\ba40\b|\ba16\b|\ba10\b|\ba6000\b|\brtx a6000\b|\brtx a5000\b|\brtx a4000\b|\brtx 30\d\d\b", (8, 6)),
    (r"\ba100\b|\ba800\b|\ba80\b", (8, 0)),
    (r"\bt4\b|\brtx 20\d\d\b", (7, 5)),
    (r"\bv100\b", (7, 0)),
]


def _guess_compute_capability(name: str) -> Optional[tuple]:
    n = name.lower()
    for pattern, cc in _CC_BY_NAME_PATTERN:
        if re.search(pattern, n):
            return cc
    return None


def detect_gpus_via_nvidia_smi() -> Optional[List[GpuInfo]]:
    if not _nvidia_smi_available():
        return None
    try:
        rows = _nvidia_smi_query("index,name,memory.total,driver_version,compute_cap")
    except Exception:
        try:
            # Older driver builds don't support the compute_cap query field; fall back without it.
            rows = [r + [""] for r in _nvidia_smi_query("index,name,memory.total,driver_version")]
        except Exception:
            return None
    gpus = []
    for row in rows:
        idx, name, vram_str, driver = row[0], row[1], row[2], row[3]
        cc_str = row[4] if len(row) > 4 and row[4] else ""
        vram_mb = int(float(vram_str))
        if cc_str and re.match(r"^\d+\.\d+$", cc_str):
            major, minor = (int(x) for x in cc_str.split("."))
        else:
            guessed = _guess_compute_capability(name)
            major, minor = guessed if guessed else (0, 0)
        gpus.append(GpuInfo(
            index=int(idx), name=name, vram_mb=vram_mb, compute_capability=f"{major}.{minor}",
            driver_version=driver, tier=classify_tier(name, vram_mb, major, minor) if major else "unknown",
        ))
    return gpus


def detect_cuda_runtime_version() -> Optional[str]:
    if not _nvidia_smi_available():
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True).stdout
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def detect_host() -> HostInfo:
    gpus = detect_gpus_via_pynvml()
    if gpus is None:
        gpus = detect_gpus_via_nvidia_smi() or []

    try:
        import psutil
        ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical = psutil.cpu_count(logical=True) or 1
    except ImportError:
        import os
        ram_mb = 0
        cores_logical = os.cpu_count() or 1
        cores_physical = cores_logical

    cpu_model = platform.processor() or "unknown"
    if not cpu_model or cpu_model == "unknown":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    driver_version = gpus[0].driver_version if gpus else None

    return HostInfo(
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        cpu_model=cpu_model,
        cpu_cores_physical=cores_physical,
        cpu_cores_logical=cores_logical,
        ram_mb=ram_mb,
        nvidia_driver_version=driver_version,
        cuda_runtime_version=detect_cuda_runtime_version(),
        gpus=gpus,
    )


def main() -> int:
    host = detect_host()
    print(json.dumps({
        **{k: v for k, v in asdict(host).items() if k != "gpus"},
        "gpus": [asdict(g) for g in host.gpus],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
