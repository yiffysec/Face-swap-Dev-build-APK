"""
The interface every diffusion_manifest.yaml `backend:` implementation must satisfy, plus the request-parameter
shape shared across all of them. Deliberately much smaller than the ONNX server's ModelSession (app/inference.py):
there's no tensor-shape negotiation here because the wire protocol is just "two images in, one image out" (see
diffusion_app/main.py) — whatever happens between those two points (detection, cropping, denoising steps,
compositing back onto the target) is entirely the backend's own business.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from ..catalog import BackendEntry


@dataclasses.dataclass
class SwapParams:
    """
    Optional knobs a client may pass on a /v1/diffusion/swap/{model_id} request (as multipart form fields
    alongside the two image files). Every field has a sensible default so a client can omit all of them; a
    backend that doesn't use a given knob (e.g. reface has no cfg "guidance_scale" concept the way the SDXL-based
    ones do) just ignores it rather than erroring.
    """
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    identity_strength: Optional[float] = None  # 0..1, how hard to push toward the source face vs. preserving pose
    seed: Optional[int] = None


def get_or_create_sdxl_base(shared: Dict[str, object], base_model: str, device: int):
    """
    Shared helper for any two backends whose manifest entries point at the same `base_model` and want to avoid
    loading its ~7GB of UNet/VAE/text-encoder weights twice (see diffusion_manifest.yaml's `shares_base_with`).
    Returns a plain `diffusers.StableDiffusionXLPipeline` cached in `shared[base_model]`.

    A backend that needs a *different* pipeline class on top of the same weights (InstantID's custom
    controlnet-carrying pipeline, for instance) should build its own class via that class's `.from_pipe(base_pipe,
    ...)` — diffusers' official "reuse already-loaded components across pipeline types" API — rather than calling
    `from_pretrained` a second time; see instantid.py for the pattern. A backend that just needs the plain SDXL
    pipeline (PuLID, which patches attention processors onto the UNet directly rather than wrapping the pipeline
    class) can use the return value of this function as-is.
    """
    import torch
    import diffusers

    if base_model in shared:
        return shared[base_model]
    pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe.to(f"cuda:{device}")
    shared[base_model] = pipe
    return pipe


class DiffusionSwapBackend(ABC):
    def __init__(self, entry: BackendEntry, weights_dir: Path, device: int):
        self.entry = entry
        self.weights_dir = weights_dir / entry.id
        self.device = device

    @abstractmethod
    def load(self, shared: Dict[str, object]) -> None:
        """
        Load weights onto `self.device`. `shared` is the BackendManager's cross-backend cache, keyed by
        `entry.base_model` — a backend whose manifest entry sets `shares_base_with` should check `shared` for an
        already-loaded base pipeline before building its own, and should register whatever it builds back into
        `shared` under `self.entry.base_model` so a backend loaded after it can reuse the same checkpoint.
        """

    @abstractmethod
    def swap(self, source: Image.Image, target: Image.Image, params: SwapParams) -> Image.Image:
        """
        `source` is the face identity to apply; `target` is the image whose face(s) get replaced. Returns a new
        PIL Image the same size as `target`. Must not mutate either input image in place.
        """
