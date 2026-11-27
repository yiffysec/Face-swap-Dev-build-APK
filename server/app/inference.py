"""
Generic multi-model ONNX Runtime serving. Deliberately dumb: given a model id, it loads whatever .onnx file has that
name and runs whatever tensors it's handed. All face-swap-specific logic (HyperSwap's own mask, InSwapper's
identity-projection matmul, pixel-boost tiling, masking, blending, the whole pipeline in short) lives in the Android
client's core module and runs unchanged whether a model is local or remote — see
core/src/main/kotlin/com/faceswap/core/Engine.kt + Models.kt. This file never special-cases a model id.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

from .catalog import ModelEntry
from .config import DeviceConfig

log = logging.getLogger("faceswapd.inference")

_ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


class ModelNotFoundError(Exception):
    pass


class ModelSession:
    """One ONNX model loaded once on one device. Thread-safe: ORT's `Run` may be called concurrently."""

    def __init__(self, entry: ModelEntry, model_path: Path, device: DeviceConfig, providers: List[tuple]):
        self.id = entry.id
        self.role = entry.role
        t0 = __import__("time").monotonic()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self._session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        self.input_names = [i.name for i in self._session.get_inputs()]
        self.output_names = [o.name for o in self._session.get_outputs()]
        self._input_types = {i.name: _ORT_TO_NUMPY.get(i.type, np.float32) for i in self._session.get_inputs()}
        log.info("loaded %s on device %s (%s) in %.1fs [%s]", entry.id, device.index, device.name,
                  __import__("time").monotonic() - t0, self._session.get_providers()[0])

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feeds = {}
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"{self.id}: missing input '{name}' (expects {self.input_names})")
            feeds[name] = np.ascontiguousarray(inputs[name], dtype=self._input_types.get(name, np.float32))
        outputs = self._session.run(self.output_names, feeds)
        return {name: np.asarray(val, dtype=np.float32) for name, val in zip(self.output_names, outputs)}


def build_providers(device: DeviceConfig, trt_cache_dir: str) -> List[tuple]:
    """Execution-provider stack for one device, decided by setup/plan.py's hardware classification (see that
    module for the reasoning). Order matters: ORT tries providers left to right and falls back down the list."""
    if device.index < 0:
        return [("CPUExecutionProvider", {})]

    providers: List[tuple] = []
    if device.use_tensorrt:
        trt_opts = {
            "device_id": device.index,
            "trt_fp16_enable": device.trt_fp16,
            "trt_max_workspace_size": device.trt_workspace_mb * 1024 * 1024,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": f"{trt_cache_dir}/dev{device.index}",
            "trt_timing_cache_enable": True,
        }
        if device.trt_int8:
            trt_opts["trt_int8_enable"] = True
            if device.trt_int8_calibration_cache:
                trt_opts["trt_int8_calibration_table_name"] = device.trt_int8_calibration_cache
        providers.append(("TensorrtExecutionProvider", trt_opts))

    providers.append(("CUDAExecutionProvider", {
        "device_id": device.index,
        "gpu_mem_limit": device.gpu_mem_limit_mb * 1024 * 1024,
        "arena_extend_strategy": device.arena_extend_strategy,
        "cudnn_conv_algo_search": "EXHAUSTIVE" if device.tier in ("hopper", "ampere_datacenter", "ada_datacenter") else "HEURISTIC",
        "do_copy_in_default_stream": True,
    }))
    providers.append(("CPUExecutionProvider", {}))
    return providers


class DeviceWorker:
    """Every model session resident on one device, plus a concurrency gate sized to that device's VRAM headroom."""

    def __init__(self, device: DeviceConfig, models_dir: Path, trt_cache_dir: str, entries_by_id: Dict[str, ModelEntry]):
        self.device = device
        self._models_dir = models_dir
        self._trt_cache_dir = trt_cache_dir
        self._entries = entries_by_id
        self._sessions: Dict[str, ModelSession] = {}
        self._lock = threading.Lock()
        self.semaphore = threading.Semaphore(device.concurrency)
        self.active = 0
        self.total_served = 0

    def preload(self, model_ids: List[str]) -> None:
        for mid in model_ids:
            self._get_or_load(mid)

    def available_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def session_if_loaded(self, model_id: str) -> "ModelSession | None":
        return self._sessions.get(model_id)

    def _get_or_load(self, model_id: str) -> ModelSession:
        with self._lock:
            sess = self._sessions.get(model_id)
            if sess is not None:
                return sess
            entry = self._entries.get(model_id)
            if entry is None:
                raise ModelNotFoundError(model_id)
            path = self._models_dir / entry.filename
            if not path.exists():
                raise ModelNotFoundError(f"{model_id} (weights not downloaded: {path} missing – run download_models.py)")
            providers = build_providers(self.device, self._trt_cache_dir)
            sess = ModelSession(entry, path, self.device, providers)
            self._sessions[model_id] = sess
            return sess

    def infer(self, model_id: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        session = self._get_or_load(model_id)  # lazy: a model not in the preload set loads on first use
        with self.semaphore:
            self.active += 1
            try:
                return session.run(inputs)
            finally:
                self.active -= 1
                self.total_served += 1


class Dispatcher:
    """Picks the least-busy device for each request. One process, N devices – see server/README.md for why this
    beats a process-per-GPU design here (ORT sessions are thread-safe; IPC would just add overhead moving MBs of
    image tensors between processes)."""

    def __init__(self, workers: List[DeviceWorker], entries: List[ModelEntry]):
        self.workers = workers
        self.entries_by_id = {e.id: e for e in entries}

    def device_summary(self) -> str:
        return ", ".join(f"{w.device.name} ({w.device.tier})" for w in self.workers)

    def loaded_model_ids(self) -> List[str]:
        ids = set()
        for w in self.workers:
            ids.update(w.available_ids())
        return sorted(ids)

    def preload(self, model_ids: List[str]) -> None:
        for w in self.workers:
            w.preload(model_ids)

    def pick_worker(self) -> DeviceWorker:
        return min(self.workers, key=lambda w: w.active / max(1, w.device.concurrency))

    def session_if_loaded(self, model_id: str) -> "ModelSession | None":
        for w in self.workers:
            sess = w.session_if_loaded(model_id)
            if sess is not None:
                return sess
        return None

    def infer(self, model_id: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if model_id not in self.entries_by_id:
            raise ModelNotFoundError(model_id)
        return self.pick_worker().infer(model_id, inputs)
