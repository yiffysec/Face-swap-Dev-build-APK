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
    """
    One ONNX model loaded once on one device. Thread-safe: ORT's `Run` (and `run_with_iobinding`) may be called
    concurrently on the same session.

    Two execution paths:
      - IO Binding (always, when the device is a GPU): binds inputs/outputs as device-resident OrtValues instead
        of letting `session.run()` do its own implicit host<->device staging, which saves a copy on every call.
        This is the one safe to always turn on — it's a stable, broadly-supported API with no shape constraints.
      - CUDA Graph replay (opt-in via `device.cuda_graph` in server_config.yaml, default off): captures the whole
        kernel-launch sequence once and replays it on every subsequent call, which matters a lot for small/fast
        graphs like these face models where kernel-launch overhead is a real fraction of total latency. This is
        genuinely fragile: it requires every input shape to be static (no dynamic axes) and the same device
        memory addresses reused call to call, which conflicts with this server's "any model, any shape" design
        for anything whose input size varies per request (a detector running on a full-resolution image, an
        upscaler at an arbitrary resolution). So it's attempted only when the model's own ONNX graph declares
        fully static input shapes, and a one-time trial capture at load time must succeed, or it's silently
        disabled for that session with a log line — never a crash, and never silently wrong output. This path
        has NOT been validated on real GPU hardware as part of this change (built in an environment with no
        CUDA device); treat it as experimental until you've confirmed it helps on your own box, and turn it off
        per-device in server_config.yaml (`cuda_graph: false`) if a model misbehaves.
    """

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

        self._is_gpu = device.index >= 0
        self._ort_device_type = "cuda" if self._is_gpu else "cpu"
        self._ort_device_id = device.index if self._is_gpu else 0

        self._graph_lock = threading.Lock()
        self._graph_ready = False  # becomes True only after a successful trial capture
        self._graph_input_shapes: Dict[str, tuple] = {}
        self._bound_inputs: Dict[str, "ort.OrtValue"] = {}
        self._bound_outputs: Dict[str, "ort.OrtValue"] = {}
        self._io_binding: "ort.IOBinding | None" = None

        want_graph = self._is_gpu and device.cuda_graph
        static_shapes = self._static_input_shapes() if want_graph else None
        if want_graph and static_shapes is not None:
            try:
                self._try_capture_graph(static_shapes)
                self._graph_ready = True
            except Exception as e:  # noqa: BLE001 - any failure here just means "no graph replay", not an error
                log.warning("%s: CUDA graph capture skipped (%s) - falling back to normal IO-bound run", entry.id, e)
        elif want_graph:
            log.info("%s: has dynamic input shapes - CUDA graph replay doesn't apply, using IO-bound run", entry.id)

        log.info("loaded %s on device %s (%s) in %.1fs [%s]%s", entry.id, device.index, device.name,
                  __import__("time").monotonic() - t0, self._session.get_providers()[0],
                  " [cuda-graph]" if self._graph_ready else "")

    def _static_input_shapes(self) -> "Dict[str, tuple] | None":
        """None if any input has a dynamic (symbolic or negative) dimension; a shape map otherwise."""
        shapes = {}
        for i in self._session.get_inputs():
            if any(not isinstance(d, int) or d <= 0 for d in i.shape):
                return None
            shapes[i.name] = tuple(i.shape)
        return shapes

    def _try_capture_graph(self, shapes: Dict[str, tuple]) -> None:
        """
        Prepares and performs the one capture call CUDA graph replay depends on. Two steps, both required:

        1. An ordinary (non-bound) run with zero-filled dummy inputs, just to learn each output's *concrete*
           shape. Static input shapes don't guarantee the exported graph also declares static output shapes
           (shape inference can leave an output symbolic even when it's deterministic in practice) - running it
           once is the only reliable way to know the real sizes.
        2. The actual capture: every input and output bound as a persistent, fixed-address OrtValue sized from
           step 1, then one `run_with_iobinding` call, which is what ORT's CUDA EP treats as the capture call
           when `enable_cuda_graph` is set. Every later call reuses these same buffers (see `update_inplace` in
           `_run_captured`) so the replay stays valid.
        """
        dummy_feeds = {n: np.zeros(shapes[n], dtype=self._input_types.get(n, np.float32)) for n in self.input_names}
        dummy_outputs = self._session.run(self.output_names, dummy_feeds)

        io_binding = self._session.io_binding()
        for name in self.input_names:
            ov = ort.OrtValue.ortvalue_from_numpy(dummy_feeds[name], self._ort_device_type, self._ort_device_id)
            self._bound_inputs[name] = ov
            io_binding.bind_ortvalue_input(name, ov)
        for name, sample in zip(self.output_names, dummy_outputs):
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                list(sample.shape), np.float32, self._ort_device_type, self._ort_device_id
            )
            self._bound_outputs[name] = ov
            io_binding.bind_ortvalue_output(name, ov)
        self._session.run_with_iobinding(io_binding)  # captures the graph on this call
        self._graph_input_shapes = shapes
        self._io_binding = io_binding

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"{self.id}: missing input '{name}' (expects {self.input_names})")

        if self._graph_ready:
            shapes_match = all(
                np.ascontiguousarray(inputs[n]).shape == self._graph_input_shapes[n] for n in self.input_names
            )
            if shapes_match:
                return self._run_captured(inputs)
            # A request with a different shape than what was captured can't replay the graph - fall through to
            # the plain IO-bound path for this one call rather than erroring or silently using stale data.

        return self._run_io_bound(inputs)

    def _run_captured(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        with self._graph_lock:
            for name in self.input_names:
                arr = np.ascontiguousarray(inputs[name], dtype=self._input_types.get(name, np.float32))
                self._bound_inputs[name].update_inplace(arr)  # same device buffer -> graph replay stays valid
            self._session.run_with_iobinding(self._io_binding)
            return {name: self._bound_outputs[name].numpy().astype(np.float32) for name in self.output_names}

    def _run_io_bound(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        io_binding = self._session.io_binding()
        for name in self.input_names:
            arr = np.ascontiguousarray(inputs[name], dtype=self._input_types.get(name, np.float32))
            io_binding.bind_cpu_input(name, arr) if not self._is_gpu else io_binding.bind_ortvalue_input(
                name, ort.OrtValue.ortvalue_from_numpy(arr, self._ort_device_type, self._ort_device_id)
            )
        for name in self.output_names:
            io_binding.bind_output(name, self._ort_device_type, device_id=self._ort_device_id)
        self._session.run_with_iobinding(io_binding)
        return {name: val.astype(np.float32) for name, val in zip(self.output_names, io_binding.copy_outputs_to_cpu())}


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
        "cudnn_conv_algo_search": "EXHAUSTIVE" if device.tier in (
            "hopper", "ampere_datacenter", "ada_datacenter",
            "blackwell_datacenter", "blackwell_workstation", "blackwell_consumer",
        ) else "HEURISTIC",
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
