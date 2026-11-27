"""
InstantID (SDXL) backend — IP-Adapter + ControlNet identity injection on top of an SDXL base checkpoint. See
https://github.com/InstantID/InstantID. This is the "default" diffusion swapper: single forward pass per
denoising step (no custom inpainting mask pipeline the way reface needs), good identity fidelity, and the
fastest of the three backends to load and run.

InstantID ships its SDXL pipeline as a *custom* diffusers pipeline file (not merged into core `diffusers` as of
this writing), so we fetch `pipeline_stable_diffusion_xl_instantid.py` from their repo the same way their own
README's quickstart does, via `custom_pipeline=...,`. If a future `diffusers` release folds this in natively,
swap the `from_pretrained(..., custom_pipeline=...)` call below for the plain class import — everything else here
(face embedding, keypoint conditioning image, generation call) stays the same either way.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image

from .base import DiffusionSwapBackend, SwapParams, get_or_create_sdxl_base

log = logging.getLogger("faceswapd.diffusion.instantid")

CUSTOM_PIPELINE_REPO = "InstantX/InstantID"
CONTROLNET_SUBFOLDER = "ControlNetModel"
IP_ADAPTER_FILENAME = "ip-adapter.bin"


def _draw_kps(image_size: tuple[int, int], kps: np.ndarray) -> Image.Image:
    """
    Renders the 5-point face keypoints as the colored-dot/line control image InstantID's ControlNet was trained
    on. Lifted from InstantID's own `pipeline_stable_diffusion_xl_instantid.py::draw_kps` reference implementation
    — kept here rather than imported from the custom pipeline module so this works even before that module has
    been fetched (face detection can be sanity-checked independently of the generation pipeline).
    """
    stickwidth = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    for i in range(len(limb_seq)):
        idx0, idx1 = limb_seq[i]
        x = kps[[idx0, idx1], 0]
        y = kps[[idx0, idx1], 1]
        length = np.hypot(x[0] - x[1], y[0] - y[1])
        angle = np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1,
        )
        cv2.fillConvexPoly(canvas, polygon, colors[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for idx, (x, y) in enumerate(kps):
        cv2.circle(canvas, (int(x), int(y)), 10, colors[idx], -1)
    return Image.fromarray(canvas)


class InstantIDBackend(DiffusionSwapBackend):
    def load(self, shared: Dict[str, object]) -> None:
        import diffusers
        from huggingface_hub import hf_hub_download
        from insightface.app import FaceAnalysis

        device = f"cuda:{self.device}"

        self._face_app = FaceAnalysis(
            name="antelopev2", root=str(self.weights_dir / "insightface"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._face_app.prepare(ctx_id=self.device, det_size=(640, 640))

        # Reuses the plain SDXL pipeline (UNet/VAE/text-encoders) that another shares_base_with backend may have
        # already loaded for this base_model, via diffusers' from_pipe() component-reuse API, rather than loading
        # those multi-GB weights from disk a second time. See base.get_or_create_sdxl_base's docstring.
        base_pipe = get_or_create_sdxl_base(shared, self.entry.base_model, self.device)

        controlnet_path = self.weights_dir / CONTROLNET_SUBFOLDER
        controlnet = diffusers.ControlNetModel.from_pretrained(
            str(controlnet_path) if controlnet_path.exists() else CUSTOM_PIPELINE_REPO,
            subfolder=None if controlnet_path.exists() else CONTROLNET_SUBFOLDER,
            torch_dtype=torch.float16,
        ).to(device)

        # Fetches just the custom pipeline *class* (the small .py file InstantID publishes alongside their repo,
        # same mechanism from_pretrained(custom_pipeline=...) uses internally) without pulling its weights in a
        # second time — from_pipe() below hands it base_pipe's already-loaded unet/vae/text_encoders instead.
        from diffusers.utils import get_class_from_dynamic_module
        instantid_cls = get_class_from_dynamic_module(
            CUSTOM_PIPELINE_REPO, module_file="pipeline_stable_diffusion_xl_instantid.py",
        )
        self._pipe = instantid_cls.from_pipe(base_pipe, controlnet=controlnet)
        self._pipe.enable_vae_tiling()

        ip_adapter_path = self.weights_dir / IP_ADAPTER_FILENAME
        if not ip_adapter_path.exists():
            ip_adapter_path = Path(hf_hub_download(CUSTOM_PIPELINE_REPO, IP_ADAPTER_FILENAME))
        self._pipe.load_ip_adapter_instantid(str(ip_adapter_path))

    def swap(self, source: Image.Image, target: Image.Image, params: SwapParams) -> Image.Image:
        source_cv = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = self._face_app.get(source_cv)
        if not faces:
            raise ValueError("no face detected in source image")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        face_emb = face.normed_embedding

        target_cv = cv2.cvtColor(np.array(target.convert("RGB")), cv2.COLOR_RGB2BGR)
        target_faces = self._face_app.get(target_cv)
        kps = target_faces[0].kps if target_faces else face.kps
        kps_image = _draw_kps(target.size, kps)

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=f"cuda:{self.device}").manual_seed(params.seed)

        self._pipe.set_ip_adapter_scale(params.identity_strength if params.identity_strength is not None else 0.8)
        result = self._pipe(
            prompt="",
            negative_prompt="lowres, blurry, distorted face, extra limbs",
            image_embeds=face_emb,
            image=kps_image,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=params.steps or 30,
            guidance_scale=params.guidance_scale if params.guidance_scale is not None else 5.0,
            height=target.height,
            width=target.width,
            generator=generator,
        )
        return result.images[0]
