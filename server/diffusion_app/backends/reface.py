"""
REFace (SD 1.4 inpainting) backend — https://github.com/Sanoojan/REFace. A genuinely different approach from
instantid/pulid: it's a diffusion-inpainting swap (a 9-channel UNet fine-tuned from SD-1.4 to take a masked target
region + source face conditioning and inpaint the swapped face directly), not an IP-Adapter identity injection on
top of a general text-to-image model. Worth offering as a third option for that reason, but it does NOT share a
base checkpoint with instantid_sdxl/pulid_sdxl (different base model entirely — SD-1.4 vs SDXL — see
diffusion_manifest.yaml), and its trained checkpoint's license is more restrictive (non-commercial) than the other
two; main.py's /v1/diffusion/health response surfaces each backend's `license` field so a client UI can show that
distinction rather than presenting all three as interchangeable.

Like pulid.py, this is a thin adapter around REFace's own reference implementation (cloned on first load) rather
than a reimplementation of their inpainting pipeline construction, face-parsing mask generation, or dlib-based
face alignment — see their `inference_test_bench.py` for the equivalent single-process flow this mirrors.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image

from .base import DiffusionSwapBackend, SwapParams

log = logging.getLogger("faceswapd.diffusion.reface")

REFACE_REPO_URL = "https://github.com/Sanoojan/REFace.git"
REFACE_WEIGHTS_REPO = "Sanoojan/REFace"
REFACE_CKPT_FILENAME = "last.ckpt"
DLIB_LANDMARKS_FILENAME = "shape_predictor_68_face_landmarks.dat"


def _ensure_reface_repo(checkout_dir: Path) -> Path:
    if not checkout_dir.exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning REFace reference implementation into %s", checkout_dir)
        subprocess.run(["git", "clone", "--depth", "1", REFACE_REPO_URL, str(checkout_dir)], check=True)
    if str(checkout_dir) not in sys.path:
        sys.path.insert(0, str(checkout_dir))
    return checkout_dir


class REFaceBackend(DiffusionSwapBackend):
    def load(self, shared: Dict[str, object]) -> None:
        # REFace's base_model (SD-1.4) is never shared with instantid/pulid's SDXL base, so unlike those two this
        # backend ignores `shared` entirely — it always loads its own copy. See this module's docstring.
        repo_dir = _ensure_reface_repo(self.weights_dir.parent / "_reface_repo")

        import dlib  # noqa: E402  (only required by this backend; see backends/__init__.py's lazy-import note)
        from omegaconf import OmegaConf  # noqa: E402
        from ldm.util import instantiate_from_config  # noqa: E402  (vendored under repo_dir; see sys.path insert)

        from huggingface_hub import hf_hub_download
        ckpt_path = self.weights_dir / REFACE_CKPT_FILENAME
        if not ckpt_path.exists():
            ckpt_path = Path(hf_hub_download(REFACE_WEIGHTS_REPO, REFACE_CKPT_FILENAME))
        landmarks_path = self.weights_dir / DLIB_LANDMARKS_FILENAME

        config_path = repo_dir / "configs" / "project.yaml"  # see repo's Demo.sh for the config this checkpoint pairs with
        config = OmegaConf.load(str(config_path))
        model = instantiate_from_config(config.model)
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
        model = model.to(f"cuda:{self.device}").eval()
        self._model = model

        from ldm.models.diffusion.ddim import DDIMSampler  # noqa: E402
        self._sampler = DDIMSampler(model)

        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(str(landmarks_path))

        # Face parsing (CelebAMask-HQ 19-class BiSeNet) for the swap mask — REFace's own repo bundles a loader for
        # this under `src/face_parsing/`; imported lazily here for the same reason as the dlib/ldm imports above.
        from src.face_parsing.model import BiSeNet  # noqa: E402
        parsing_ckpt = self.weights_dir / "79999_iter.pth"
        parser = BiSeNet(n_classes=19)
        parser.load_state_dict(torch.load(str(parsing_ckpt), map_location="cpu"))
        self._face_parser = parser.to(f"cuda:{self.device}").eval()

    def swap(self, source: Image.Image, target: Image.Image, params: SwapParams) -> Image.Image:
        from src.demo.utils import run_face_swap  # noqa: E402  (REFace's own end-to-end helper; see Demo.sh)

        source_np = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        target_np = cv2.cvtColor(np.array(target.convert("RGB")), cv2.COLOR_RGB2BGR)

        result_np = run_face_swap(
            source_image=source_np,
            target_image=target_np,
            model=self._model,
            sampler=self._sampler,
            face_detector=self._face_detector,
            landmark_predictor=self._landmark_predictor,
            face_parser=self._face_parser,
            ddim_steps=params.steps or 50,
            device=f"cuda:{self.device}",
        )
        return Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))
