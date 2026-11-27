"""
FastAPI entrypoint. Matches the Android client's RemoteClient.kt contract exactly:

    GET  /v1/health            -> {"name", "version", "device", "models": [{"id","role","inputs","outputs"}]}
    POST /v1/infer/{model_id}  <- application/octet-stream TensorWire body of named input tensors
                                -> application/octet-stream TensorWire body of named output tensors, or
                                   a non-2xx response with {"error": "..."} on failure

Both routes require the same bearer token when server_config.yaml sets api_key, so a key also protects the model
manifest from anonymous probing, not just inference.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8443 [--ssl-certfile ... --ssl-keyfile ...]
(setup_server.sh wires this up as a systemd unit with flags chosen for the detected hardware.)
"""
from __future__ import annotations

import hmac
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from . import catalog, tensor_wire
from .config import DEFAULT_CONFIG_PATH, load_config
from .inference import Dispatcher, DeviceWorker, ModelNotFoundError

VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("faceswapd")

# Populated by the lifespan handler below; module-level so route functions (which FastAPI instantiates fresh per
# request) and the auth dependency can both reach it without threading it through every signature.
state: dict = {}


def _model_set_ids(model_set: str, entries) -> list[str]:
    remote_entries = [e for e in entries if catalog.is_remotely_servable(e.id)]
    if model_set == "full":
        return [e.id for e in remote_entries]
    if model_set == "core":
        # Mirrors the Android app's own "core" bundle (ModelSpec.core == true in ModelManager.kt): required +
        # recommended only. Extra swappers/enhancers/studio/live models load lazily on first request instead.
        return [e.id for e in remote_entries if e.group in ("required", "recommended")]
    # Anything else is treated as a literal comma-separated id list.
    wanted = {x.strip() for x in model_set.split(",") if x.strip()}
    return [e.id for e in remote_entries if e.id in wanted]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    entries = catalog.load_manifest(Path(cfg.manifest_path))
    models_dir = Path(cfg.models_dir)
    entries_by_id = catalog.by_id(entries)

    workers = [DeviceWorker(d, models_dir, cfg.trt_cache_dir, entries_by_id) for d in cfg.devices]
    dispatcher = Dispatcher(workers, entries)

    wanted_ids = _model_set_ids(cfg.model_set, entries)
    t0 = time.monotonic()
    log.info("loading %d model(s) across %d device(s): %s", len(wanted_ids), len(workers), dispatcher.device_summary())
    # Spread the preload set round-robin across devices so a multi-GPU box shares VRAM instead of piling every
    # model onto device 0; infer() still routes to whichever device(s) actually hold a given model.
    for i, mid in enumerate(wanted_ids):
        workers[i % len(workers)].preload([mid])
    log.info("ready in %.1fs — %d model(s) resident", time.monotonic() - t0, len(dispatcher.loaded_model_ids()))

    state["cfg"] = cfg
    state["entries_by_id"] = entries_by_id
    state["dispatcher"] = dispatcher
    yield
    state.clear()


app = FastAPI(title="faceswap-inference-server", version=VERSION, lifespan=lifespan)


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    expected = state["cfg"].api_key
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[len("Bearer "):]
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/v1/health", dependencies=[Depends(require_auth)])
async def health():
    dispatcher: Dispatcher = state["dispatcher"]
    entries_by_id = state["entries_by_id"]
    models = []
    for mid in dispatcher.loaded_model_ids():
        entry = entries_by_id[mid]
        sess = dispatcher.session_if_loaded(mid)
        models.append({
            "id": mid,
            "role": entry.role,
            "inputs": sess.input_names if sess else [],
            "outputs": sess.output_names if sess else [],
        })
    return {
        "name": "faceswap-inference-server",
        "version": VERSION,
        "device": dispatcher.device_summary(),
        "models": models,
    }


@app.post("/v1/infer/{model_id}", dependencies=[Depends(require_auth)])
async def infer(model_id: str, request: Request):
    dispatcher: Dispatcher = state["dispatcher"]
    body = await request.body()
    try:
        inputs = tensor_wire.read_map(io.BytesIO(body))
    except tensor_wire.WireError as e:
        return JSONResponse(status_code=400, content={"error": f"malformed request body: {e}"})

    try:
        outputs = dispatcher.infer(model_id, inputs)
    except ModelNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": f"unknown or unavailable model: {e}"})
    except Exception as e:  # noqa: BLE001 - any ORT/runtime failure becomes a reported error, not a crash
        log.exception("inference failed for %s", model_id)
        return JSONResponse(status_code=500, content={"error": str(e)})

    out = io.BytesIO()
    tensor_wire.write_map(out, outputs)
    return Response(content=out.getvalue(), media_type="application/octet-stream")
