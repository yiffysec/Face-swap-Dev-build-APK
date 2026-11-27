"""
The model catalog: which model ids exist, what role they play, and where to download their weights from. This is
the server-side mirror of the Android app's `ModelCatalog` (app/src/main/java/com/faceswap/studio/data/ModelManager.kt)
— same ids, same Hugging Face mirror, same versions, so a phone and a server set up from this repo always agree on
what a model id means. Ids are never hardcoded into the inference path (see inference.py): this file is the only
place that lists them.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "models_manifest.yaml"


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    id: str
    role: str  # detector | recognizer | swapper | enhancer | analyser | parser | landmarker | background | studio | live
    group: str  # required | recommended | swappers | enhancers | studio | live  (mirrors ModelGroup in Settings.kt)
    url: str
    filename: str
    hash_url: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: int = 0
    license: str = ""
    notes: str = ""


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> List[ModelEntry]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    entries = []
    for item in raw.get("models", []):
        entries.append(ModelEntry(
            id=item["id"],
            role=item.get("role", "other"),
            group=item.get("group", "recommended"),
            url=item["url"],
            filename=item.get("filename", f"{item['id']}.onnx"),
            hash_url=item.get("hash_url"),
            sha256=item.get("sha256"),
            size_bytes=int(item.get("size_bytes", 0)),
            license=item.get("license", ""),
            notes=item.get("notes", ""),
        ))
    return entries


def by_id(entries: List[ModelEntry]) -> Dict[str, ModelEntry]:
    return {e.id: e for e in entries}


# Model ids whose Kotlin-side wrapper needs raw .onnx bytes for a trick that has no remote equivalent (InSwapper's
# identity-projection matrix is read directly out of the file instead of run through the network – see
# core/src/main/kotlin/com/faceswap/core/Models.kt:InSwapper.loadEmap). The Android client already excludes these
# from the remote catalog (RemoteCatalog.has); the server leaves them out of /v1/health for the same reason, so a
# client never advertises or offers a swapper it can't actually use remotely.
NO_REMOTE_EQUIVALENT_PREFIXES = ("inswapper",)


def is_remotely_servable(model_id: str) -> bool:
    return not model_id.startswith(NO_REMOTE_EQUIVALENT_PREFIXES)
