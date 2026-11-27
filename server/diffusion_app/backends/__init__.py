"""
Registry mapping a diffusion_manifest.yaml `backend:` name to the Python class that implements it, plus the
BackendManager that the FastAPI app (diffusion_app/main.py) actually calls: lazy per-id loading, and sharing one
base pipeline across backend ids that declare `shares_base_with` (see instantid_sdxl/pulid_sdxl in the manifest) so
a 7GB SDXL checkpoint is paid for once, not once per adapter that happens to sit on top of it.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict

from PIL import Image

from .. import catalog
from .base import DiffusionSwapBackend, SwapParams

log = logging.getLogger("faceswapd.diffusion")


def _backend_class(name: str):
    # Imported lazily, one at a time, so a box that only has instantid_sdxl's weights downloaded doesn't need
    # reface's extra dependencies (dlib, the REFace repo checkout) importable just to serve health/instantid
    # requests — an ImportError here only breaks the one backend that actually needs the missing package.
    if name == "instantid":
        from .instantid import InstantIDBackend
        return InstantIDBackend
    if name == "pulid":
        from .pulid import PuLIDBackend
        return PuLIDBackend
    if name == "reface":
        from .reface import REFaceBackend
        return REFaceBackend
    raise ValueError(f"no diffusion backend implementation registered for {name!r}")


class BackendManager:
    """
    Owns every loaded backend instance plus the base-pipeline cache they share. One instance lives in
    diffusion_app/main.py's app state, same pattern as app/inference.py's Dispatcher on the ONNX side.

    Thread-safety note: a lock guards loading (first request for a cold id blocks concurrent requests for the
    *same* id rather than double-loading a multi-GB pipeline), but swap() calls themselves are not serialized here
    — unlike the ONNX server's per-device worker queue, a diffusion pipeline is a single big GPU job already, so
    FastAPI's own request concurrency plus CUDA's own stream serialization is what actually rate-limits this; a
    production deployment wanting bounded concurrency per GPU should front this with a semaphore sized to how much
    headroom device VRAM leaves after one pipeline copy, which is a device/model-specific number this file doesn't
    try to guess.
    """

    def __init__(self, entries_by_id: Dict[str, catalog.BackendEntry], weights_dir: Path, device: int):
        self._entries = entries_by_id
        self._weights_dir = weights_dir
        self._device = device
        self._instances: Dict[str, DiffusionSwapBackend] = {}
        self._shared_pipelines: Dict[str, object] = {}  # keyed by base_model string
        self._lock = threading.Lock()

    def loaded_ids(self) -> list[str]:
        return list(self._instances.keys())

    def _get_or_load(self, model_id: str) -> DiffusionSwapBackend:
        if model_id in self._instances:
            return self._instances[model_id]
        with self._lock:
            if model_id in self._instances:  # re-check: another request may have loaded it while we waited
                return self._instances[model_id]
            entry = self._entries.get(model_id)
            if entry is None:
                raise KeyError(model_id)
            cls = _backend_class(entry.backend)
            instance = cls(entry, self._weights_dir, self._device)
            t0 = __import__("time").monotonic()
            log.info("loading diffusion backend %s (%s)...", model_id, entry.backend)
            instance.load(self._shared_pipelines)
            log.info("%s ready in %.1fs", model_id, __import__("time").monotonic() - t0)
            self._instances[model_id] = instance
            return instance

    def preload(self, model_ids: list[str]) -> None:
        for mid in model_ids:
            self._get_or_load(mid)

    def swap(self, model_id: str, source: Image.Image, target: Image.Image, params: SwapParams) -> Image.Image:
        backend = self._get_or_load(model_id)
        return backend.swap(source, target, params)
