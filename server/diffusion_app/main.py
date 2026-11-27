"""
FastAPI entrypoint for the diffusion swap tier — a separate process from app/main.py's ONNX server (different
dependency stack: torch/diffusers vs onnxruntime-gpu) that the ONNX server proxies into transparently, so the
Android client and the Cloudflare Tunnel only ever see one host:port. See app/main.py's `/v1/diffusion/{path}`
route docstring for that side of the wiring.

Wire protocol (deliberately NOT TensorWire — see tensor_wire.py's docstring on the ONNX side for why that protocol
exists; it doesn't apply here because these backends don't take named tensors, they take two whole images):

    GET  /v1/diffusion/health                 -> {"name","version","device","models":[{"id","display_name","role",
                                                   "license","approx_vram_gb","notes"}]}
    POST /v1/diffusion/swap/{model_id}        <- multipart/form-data: `source` (image file, the identity to
                                                  apply), `target` (image file, whose face gets replaced), and
                                                  optional form fields `steps`, `guidance_scale`,
                                                  `identity_strength`, `seed` (see backends/base.py:SwapParams)
                                                -> image/png bytes, or a non-2xx response with {"error": "..."}

Both routes require the same bearer token as the ONNX server (server_config.yaml's api_key, copied into
diffusion_config.yaml by setup_diffusion.sh) — see config.py's module docstring for why the two config files are
separate despite sharing that one value.

Run with: uvicorn diffusion_app.main:app --host 127.0.0.1 --port 8444
(setup_diffusion.sh wires this up as a supervisor/systemd unit alongside the ONNX server's.)
"""
from __future__ import annotations

import hmac
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from . import catalog
from .backends import BackendManager
from .backends.base import SwapParams
from .config import DEFAULT_CONFIG_PATH, load_config

VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("faceswapd.diffusion")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    entries = catalog.load_manifest(Path(cfg.manifest_path))
    entries_by_id = catalog.by_id(entries)
    manager = BackendManager(entries_by_id, Path(cfg.weights_dir), cfg.device)

    if cfg.preload:
        t0 = time.monotonic()
        log.info("preloading %d diffusion backend(s): %s", len(cfg.preload), cfg.preload)
        manager.preload(cfg.preload)
        log.info("preload done in %.1fs", time.monotonic() - t0)
    else:
        log.info("no backends preloaded — each loads lazily on its first request (see diffusion_config.yaml)")

    state["cfg"] = cfg
    state["entries_by_id"] = entries_by_id
    state["manager"] = manager
    yield
    state.clear()


app = FastAPI(title="faceswap-diffusion-server", version=VERSION, lifespan=lifespan)


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    expected = state["cfg"].api_key
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[len("Bearer "):]
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/v1/diffusion/health", dependencies=[Depends(require_auth)])
async def health():
    manager: BackendManager = state["manager"]
    entries_by_id: dict = state["entries_by_id"]
    models = []
    for entry in entries_by_id.values():
        models.append({
            "id": entry.id,
            "display_name": entry.display_name,
            "role": entry.role,
            "license": entry.license,
            "approx_vram_gb": entry.approx_vram_gb,
            "notes": entry.notes,
            "loaded": entry.id in manager.loaded_ids(),
        })
    return {
        "name": "faceswap-diffusion-server",
        "version": VERSION,
        "device": state["cfg"].device,
        "models": models,
    }


@app.post("/v1/diffusion/swap/{model_id}", dependencies=[Depends(require_auth)])
async def swap(
    model_id: str,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    steps: Optional[int] = Form(default=None),
    guidance_scale: Optional[float] = Form(default=None),
    identity_strength: Optional[float] = Form(default=None),
    seed: Optional[int] = Form(default=None),
):
    manager: BackendManager = state["manager"]
    entries_by_id: dict = state["entries_by_id"]
    if model_id not in entries_by_id:
        raise HTTPException(status_code=404, detail=f"unknown diffusion backend: {model_id}")

    try:
        source_img = Image.open(io.BytesIO(await source.read())).convert("RGB")
        target_img = Image.open(io.BytesIO(await target.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"couldn't decode an uploaded image: {exc}") from exc

    params = SwapParams(steps=steps, guidance_scale=guidance_scale, identity_strength=identity_strength, seed=seed)
    t0 = time.monotonic()
    try:
        result = manager.swap(model_id, source_img, target_img, params)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown diffusion backend: {model_id}")
    except ValueError as exc:  # e.g. "no face detected in source image" — a client-fixable input problem
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("diffusion swap failed for %s", model_id)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    log.info("%s swap done in %.1fs", model_id, time.monotonic() - t0)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
