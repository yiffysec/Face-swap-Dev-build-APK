"""
Runtime configuration for the diffusion server, loaded from diffusion_config.yaml. Deliberately a separate file
from the ONNX server's server_config.yaml (app/config.py) even though both live on the same box: the two
processes have different dependency stacks (onnxruntime-gpu vs torch/diffusers) and are started/stopped
independently (see server/setup_diffusion.sh and the supervisor/systemd templates alongside the ONNX ones), so
giving them separate config files keeps a restart of one from needing to touch the other's settings.

The api_key is still expected to match the ONNX server's, since app/main.py's /v1/diffusion/{path} route forwards
the client's bearer token through verbatim rather than re-deriving one — see that route's docstring.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("FSW_DIFFUSION_CONFIG", "diffusion_config.yaml"))


@dataclasses.dataclass
class DiffusionConfig:
    host: str = "127.0.0.1"  # loopback only: reached via the ONNX server's proxy route, never exposed directly
    port: int = 8444
    api_key: Optional[str] = None
    manifest_path: str = "diffusion_manifest.yaml"
    weights_dir: str = "diffusion/weights"
    # Backend ids to load eagerly at startup; empty means "load lazily on first request for each id" (the default
    # — these pipelines are multi-GB each, so eagerly loading all of them on a box that only has a couple of
    # unshared GPUs to spare is usually the wrong call; see diffusion_app/main.py's lifespan handler).
    preload: List[str] = dataclasses.field(default_factory=list)
    # CUDA device ordinal for diffusion inference. A single device (not a per-device worker pool like the ONNX
    # server's) because these pipelines are already large enough per-copy that running more than one concurrent
    # replica per GPU is rarely worthwhile; see diffusion_app/backends/base.py's docstring for the concurrency
    # model this implies.
    device: int = 0
    log_level: str = "info"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> DiffusionConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return DiffusionConfig(
        host=raw.get("host", "127.0.0.1"),
        port=int(raw.get("port", 8444)),
        api_key=raw.get("api_key"),
        manifest_path=raw.get("manifest_path", "diffusion_manifest.yaml"),
        weights_dir=raw.get("weights_dir", "diffusion/weights"),
        preload=list(raw.get("preload", [])),
        device=int(raw.get("device", 0)),
        log_level=raw.get("log_level", "info"),
    )


def save_config(cfg: DiffusionConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    raw = {
        "host": cfg.host, "port": cfg.port, "api_key": cfg.api_key, "manifest_path": cfg.manifest_path,
        "weights_dir": cfg.weights_dir, "preload": cfg.preload, "device": cfg.device, "log_level": cfg.log_level,
    }
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
