"""
PuLID (SDXL) backend — https://github.com/ToTheBeginning/PuLID. Unlike InstantID this doesn't wrap the pipeline
class in a ControlNet; it patches PuLID's own attention processors directly onto the shared SDXL UNet and feeds an
ID-embedding computed by PuLID's own EVA-CLIP-based `IDEncoder` through them at each step. That's the whole reason
it can sit on top of instantid_sdxl's already-loaded base pipeline for free (see diffusion_manifest.yaml's
`shares_base_with`) — it never needs its own copy of the UNet/VAE/text-encoders, only its own small (~1GB)
`IDEncoder` + attention-processor weights layered on top.

PuLID's reference implementation lives in their GitHub repo rather than as pip-installable Python; `_pulid_module`
below does a one-time, no-op-if-present checkout and adds it to `sys.path` so the rest of this file can `import
pulid` the way their own `app.py`/`infer.py` scripts do. This keeps the backend's own code a thin adapter rather
than a reimplementation of their attention-processor patching, which is the part genuinely worth not duplicating.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from .base import DiffusionSwapBackend, SwapParams, get_or_create_sdxl_base

log = logging.getLogger("faceswapd.diffusion.pulid")

PULID_REPO_URL = "https://github.com/ToTheBeginning/PuLID.git"
PULID_WEIGHTS_REPO = "guozinan/PuLID"
PULID_WEIGHTS_FILENAME = "pulid_sdxl_v1.1.safetensors"


def _ensure_pulid_repo(checkout_dir: Path) -> Path:
    if not (checkout_dir / "pulid").exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning PuLID reference implementation into %s", checkout_dir)
        subprocess.run(["git", "clone", "--depth", "1", PULID_REPO_URL, str(checkout_dir)], check=True)
    if str(checkout_dir) not in sys.path:
        sys.path.insert(0, str(checkout_dir))
    return checkout_dir


class PuLIDBackend(DiffusionSwapBackend):
    def load(self, shared: Dict[str, object]) -> None:
        repo_dir = _ensure_pulid_repo(self.weights_dir.parent / "_pulid_repo")
        from pulid.pipeline import PuLIDPipeline  # noqa: E402  (import depends on the sys.path insert above)
        from pulid.utils import resize_numpy_image_long  # noqa: E402

        self._resize_numpy_image_long = resize_numpy_image_long

        base_pipe = get_or_create_sdxl_base(shared, self.entry.base_model, self.device)

        from huggingface_hub import hf_hub_download
        weights_path = self.weights_dir / PULID_WEIGHTS_FILENAME
        if not weights_path.exists():
            weights_path = Path(hf_hub_download(PULID_WEIGHTS_REPO, PULID_WEIGHTS_FILENAME))

        # PuLIDPipeline wraps attention-processor patching around an existing SDXL pipe rather than constructing
        # its own — exactly the shared-base reuse this backend exists to get for free. See repo's pulid/pipeline.py
        # for what this actually does to base_pipe.unet; kept as an opaque call here rather than duplicated.
        self._pulid = PuLIDPipeline(sdxl_pipe=base_pipe, device=f"cuda:{self.device}", sdxl_weight_dtype=torch.float16)
        self._pulid.load_pretrain(str(weights_path))

    def swap(self, source: Image.Image, target: Image.Image, params: SwapParams) -> Image.Image:
        source_np = self._resize_numpy_image_long(np.array(source.convert("RGB")), 1024)
        id_embeddings, uncond_id_embeddings = self._pulid.get_id_embedding(source_np)

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=f"cuda:{self.device}").manual_seed(params.seed)

        images = self._pulid.inference(
            prompt="",
            size=(1, target.height, target.width),
            prompt_n="lowres, blurry, distorted face, extra limbs",
            id_embeddings=id_embeddings,
            uncond_id_embeddings=uncond_id_embeddings,
            id_scale=params.identity_strength if params.identity_strength is not None else 0.8,
            guidance_scale=params.guidance_scale if params.guidance_scale is not None else 5.0,
            steps=params.steps or 30,
            seed=params.seed if params.seed is not None else -1,
        )
        return Image.fromarray(images[0]) if isinstance(images[0], np.ndarray) else images[0]
