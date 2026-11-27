"""
The diffusion backend catalog: which backend ids exist, what Python class implements them, and where their weights
come from. Mirrors app/catalog.py's role in the ONNX server — same idea (never hardcode ids into the serving
path), different file, because diffusion backends are a different wire protocol and process from the ONNX models
(see diffusion_app/main.py's module docstring).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "diffusion_manifest.yaml"


@dataclasses.dataclass(frozen=True)
class WeightSpec:
    repo: Optional[str] = None       # Hugging Face repo id, downloaded via huggingface_hub
    files: Optional[List[str]] = None  # specific files within that repo (None = whole repo)
    url: Optional[str] = None        # or a plain URL instead of a HF repo (e.g. the dlib landmark file)
    filename: Optional[str] = None   # local filename to save a plain `url` download as


@dataclasses.dataclass(frozen=True)
class BackendEntry:
    id: str
    display_name: str
    backend: str  # name of the diffusion_app/backends/<name>.py module implementing this id
    role: str
    base_model: str
    weights: List[WeightSpec]
    license: str = ""
    approx_vram_gb: float = 0.0
    approx_disk_gb: float = 0.0
    notes: str = ""
    shares_base_with: Optional[str] = None  # another id whose loaded base pipeline this one hangs an adapter off


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> List[BackendEntry]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    entries = []
    for item in raw.get("backends", []):
        weights = [WeightSpec(**w) for w in item.get("weights", [])]
        entries.append(BackendEntry(
            id=item["id"],
            display_name=item.get("display_name", item["id"]),
            backend=item["backend"],
            role=item.get("role", "swapper"),
            base_model=item.get("base_model", ""),
            weights=weights,
            license=item.get("license", ""),
            approx_vram_gb=float(item.get("approx_vram_gb", 0)),
            approx_disk_gb=float(item.get("approx_disk_gb", 0)),
            notes=item.get("notes", ""),
            shares_base_with=item.get("shares_base_with"),
        ))
    return entries


def by_id(entries: List[BackendEntry]) -> Dict[str, BackendEntry]:
    return {e.id: e for e in entries}
