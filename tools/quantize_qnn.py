#!/usr/bin/env python3
"""Build calibrated w8a16 QDQ twins (<id>.qdq.onnx) for the Snapdragon Hexagon NPU (QNN HTP).

Why: the HTP's fast path is integer. Qualcomm's own Snapdragon 8 Gen 3 numbers show ~3x for w8a8 over float on
GAN-class image models (Real-ESRGAN x4plus 49.9 ms -> 15.4 ms). We use 16-bit activations + 8-bit weights
(w8a16): almost all of the speed, far less quality loss than w8a8.

Input: calibration tensors captured from the real pipeline (core CalibrationDump test -> <calib>/<id>/NNN.bin).
For every model:
  1. fp16 -> fp32, input shapes fixed to the captured shapes (QNN needs static shapes)
  2. qnn_preprocess_model (fuses / decomposes ops the HTP can't run -> fewer CPU fallbacks)
  3. static QDQ quantization (activations uint16, weights int8 per-channel symmetric), strided min/max calibration
  4. quality gate on held-out samples vs fp32 (ORT CPU): worst cosine >= 0.998 and, for image outputs,
     PSNR >= 40 dB. Models that fail are not shipped (the app keeps fp16 on the NPU for them).
Also writes <id>.sample.bin (one held-out real input) for the on-device tuner.

Usage: python3 tools/quantize_qnn.py <models_dir> <calib_dir> [id ...]
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

ort.set_default_logger_severity(3)

# YuNet is excluded: 230 KB, already sub-millisecond, and its sigmoid score heads don't survive int8.
DEFAULT_IDS = [
    "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper_128_fp16",
    "gpen_bfr_256", "gpen_bfr_512", "gfpgan_1.4", "restoreformer_plus_plus", "codeformer",
    "arcface_w600k_r50", "bisenet_resnet_18", "hrffa", "scrfd_2.5g",
    "modnet", "u2netp", "real_esrgan_x2_fp16", "real_esrgan_x4_fp16", "span_kendata_x4",
    "fran", "live_portrait_motion_extractor", "live_portrait_feature_extractor", "live_portrait_generator",
]
VERSION = 2
MIN_COS = 0.998
MIN_PSNR = 40.0
HELD_OUT = 3


def read_tensors(path):
    out = {}
    with open(path, "rb") as f:
        magic, _ver, n = struct.unpack(">iii", f.read(12))
        assert magic == 0x4653494E, path
        for _ in range(n):
            (ln,) = struct.unpack(">H", f.read(2))
            name = f.read(ln).decode("utf-8")
            (rank,) = struct.unpack(">i", f.read(4))
            shape = struct.unpack(">" + "q" * rank, f.read(8 * rank)) if rank else ()
            (cnt,) = struct.unpack(">i", f.read(4))
            data = np.frombuffer(f.read(4 * cnt), dtype=">f4").astype(np.float32)
            out[name] = data.reshape(shape) if rank else data.reshape(())
    return out


def to_fp32(src, dst):
    m = onnx.load(src)
    changed = False
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name)); changed = True
    for node in m.graph.node:
        for a in node.attribute:
            if node.op_type == "Cast" and a.name == "to" and a.i == onnx.TensorProto.FLOAT16:
                a.i = onnx.TensorProto.FLOAT; changed = True
            if node.op_type == "Constant" and a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32))); changed = True
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT; changed = True
    if changed:
        del m.graph.value_info[:]
    onnx.save(m, dst)


def fold_constants(src, dst):
    """With static shapes, shape arithmetic (e.g. Resize scales computed from Shape/Div) folds into constants.
    Unfolded, the quantizer would put Q/DQ on those scales and break Resize (outer scales no longer exactly 1)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


def dynamic_resize_producers(path):
    """Nodes computing non-constant Resize roi/scales/sizes inputs (excluded from quantization)."""
    m = onnx.load(path, load_external_data=False)
    consts = {i.name for i in m.graph.initializer} | {o for n in m.graph.node if n.op_type == "Constant" for o in n.output}
    producer = {o: n for n in m.graph.node for o in n.output}
    todo = [t for n in m.graph.node if n.op_type == "Resize" for t in list(n.input)[1:] if t and t not in consts]
    out, seen = [], set()
    while todo:
        t = todo.pop()
        n = producer.get(t)
        if n is None or n.name in seen:
            continue
        seen.add(n.name); out.append(n.name)
        todo += [i for i in n.input if i and i not in consts]
        if len(out) > 64:
            break
    return out


def edge_nodes(path, depth=2):
    """The first and last [depth] compute nodes (conv/matmul) – kept in float on the retry: image nets lose most
    precision at the input stem and the output head."""
    m = onnx.load(path, load_external_data=False)
    heavy = [n for n in m.graph.node if n.op_type in ("Conv", "ConvTranspose", "Gemm", "MatMul")]
    if len(heavy) <= 2 * depth:
        return []
    picks = heavy[:depth] + heavy[-depth:]
    outs = {o.name for o in m.graph.output}
    tail = [n for n in m.graph.node if any(o in outs for o in n.output)]
    return [n.name for n in picks + tail if n.name]


def fix_shapes(path, shapes):
    m = onnx.load(path)
    init_names = {i.name for i in m.graph.initializer}
    for inp in m.graph.input:
        if inp.name in shapes and inp.name not in init_names:
            dims = inp.type.tensor_type.shape.dim
            if len(dims) == len(shapes[inp.name]):
                for d, v in zip(dims, shapes[inp.name]):
                    d.ClearField("dim_param"); d.dim_value = int(v)
    onnx.save(m, path)


class Reader(CalibrationDataReader):
    """Feeds captured samples; supports ORT's strided calibration (set_range) to bound memory on big models."""

    def __init__(self, samples, names):
        self.samples = samples; self.names = set(names); self.start = 0; self.end = len(samples); self.i = 0

    def get_next(self):
        if self.i >= self.end:
            return None
        s = self.samples[self.i]; self.i += 1
        return {k: v for k, v in s.items() if k in self.names}

    def set_range(self, start_index, end_index):
        self.start = start_index; self.end = min(end_index, len(self.samples)); self.i = start_index

    def rewind(self):
        self.i = self.start

    def __len__(self):
        return len(self.samples)


def psnr(ref, out):
    rng = float(ref.max() - ref.min())
    mse = float(np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2))
    if mse == 0 or rng == 0:
        return 99.0
    return 20 * np.log10(rng / np.sqrt(mse))


def cosine(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    return float(a @ b / (na * nb + 1e-12))


def first_line(e):
    t = (str(e) or repr(e)).strip().splitlines()
    return (t[0] if t else type(e).__name__)[:160]


def is_image(arr):
    return arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[2] >= 64


def quantize_one(models_dir, calib_dir, mid, report):
    src = os.path.join(models_dir, mid + ".onnx")
    cdir = os.path.join(calib_dir, mid)
    out = os.path.join(models_dir, mid + ".qdq.onnx")
    meta = out + ".json"
    if not os.path.exists(src):
        return report(mid, "no-model")
    files = sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []
    if len(files) < HELD_OUT + 4:
        return report(mid, f"too-few-samples ({len(files)})")
    stamp = {"version": VERSION, "src": os.path.getsize(src), "samples": len(files)}
    if os.path.exists(out) and os.path.exists(meta):
        try:
            old = json.load(open(meta))
            if {k: old.get(k) for k in stamp} == stamp:
                return report(mid, f"cached ({old.get('result', '?')})")
        except Exception:
            pass
    samples = [read_tensors(os.path.join(cdir, f)) for f in files]
    calib, held = samples[:-HELD_OUT], samples[-HELD_OUT:]
    stride = min(4, len(calib))
    calib = calib[: len(calib) // stride * stride]  # strided min/max needs a multiple of the stride
    # tuner sample: one real held-out input
    shutil.copy(os.path.join(cdir, files[-1]), os.path.join(models_dir, mid + ".sample.bin"))

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = os.path.join(tmp, "fp32.onnx")
        to_fp32(src, fp32)
        fix_shapes(fp32, {k: v.shape for k, v in samples[0].items()})
        folded = os.path.join(tmp, "folded.onnx")
        try:
            fold_constants(fp32, folded)
        except Exception as e:
            print(f"  {mid}: constant folding skipped ({first_line(e)})")
            shutil.copy(fp32, folded)
        pre = os.path.join(tmp, "pre.onnx")
        try:
            if not qnn_preprocess_model(folded, pre, exclude_initializer_from_input=True, fuse_layernorm=True):
                shutil.copy(folded, pre)
        except Exception as e:  # preprocessing is an optimisation, not a requirement
            print(f"  {mid}: preprocess skipped ({first_line(e)})")
            shutil.copy(folded, pre)
        names = [i.name for i in onnx.load(pre, load_external_data=False).graph.input]
        exclude = dynamic_resize_producers(pre)
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ref = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
        ref_inputs = {i.name for i in ref.get_inputs()}
        ref_out = [ref.run(None, {k: v for k, v in s_.items() if k in ref_inputs}) for s_ in held]

        def attempt(extra_exclude, tag):
            qdq = os.path.join(tmp, f"qdq_{tag}.onnx")
            err = None
            for wt, per_channel in ((QuantType.QInt8, True), (QuantType.QUInt8, False)):
                try:
                    cfg = get_qnn_qdq_config(pre, Reader(calib, names), activation_type=QuantType.QUInt16, weight_type=wt,
                                             per_channel=per_channel, weight_symmetric=per_channel, stride=stride,
                                             nodes_to_exclude=(exclude + extra_exclude) or None)
                    quantize(pre, qdq, cfg)
                    err = None
                    break
                except Exception as e:
                    err = e
            if err is not None:
                return None, f"quantize-failed: {first_line(err)}"
            q = ort.InferenceSession(qdq, so, providers=["CPUExecutionProvider"])
            wc, wp = 1.0, 99.0
            for s_, ro in zip(held, ref_out):
                qo = q.run(None, {k: v for k, v in s_.items() if k in ref_inputs})
                for a, b_ in zip(ro, qo):
                    a = np.asarray(a, dtype=np.float32); b_ = np.asarray(b_, dtype=np.float32)
                    if a.size < 2:
                        continue
                    wc = min(wc, cosine(a, b_))
                    if is_image(a):
                        wp = min(wp, psnr(a, b_))
            return (qdq if wc >= MIN_COS and wp >= MIN_PSNR else None), f"{tag} cos={wc:.5f} psnr={wp:.1f}dB"

        good, result = attempt([], "w8a16")
        if good is None and not result.startswith("quantize-failed"):
            edges = edge_nodes(pre)
            if edges:
                good, r2 = attempt(edges, "w8a16+fp-edges")
                result += " | " + r2
        ok = good is not None
        stamp["result"] = ("ok " if ok else "rejected ") + result
        if ok:
            shutil.copy(good, out)
        elif os.path.exists(out):
            os.remove(out)
        json.dump(stamp, open(meta, "w"))
        size = os.path.getsize(out) / 1e6 if ok else 0
        return report(mid, f"{'ok' if ok else 'rejected'} {result} ({size:.0f} MB, {time.time() - t0:.0f}s)")


def main():
    models_dir, calib_dir = sys.argv[1], sys.argv[2]
    ids = sys.argv[3:] or DEFAULT_IDS
    lines = []

    def report(mid, msg):
        line = f"{mid:<34} {msg}"
        print(line, flush=True); lines.append(line)

    # real sample input for the on-device tuner, for every captured model (quantized or not)
    for mid in sorted(os.listdir(calib_dir)) if os.path.isdir(calib_dir) else []:
        d = os.path.join(calib_dir, mid)
        if os.path.isdir(d) and os.path.exists(os.path.join(models_dir, mid + ".onnx")):
            files = sorted(f for f in os.listdir(d) if f.endswith(".bin"))
            if files:
                shutil.copy(os.path.join(d, files[-1]), os.path.join(models_dir, mid + ".sample.bin"))
    for mid in ids:
        try:
            quantize_one(models_dir, calib_dir, mid, report)
        except Exception as e:
            report(mid, f"error: {first_line(e)}")
    with open(os.path.join(models_dir, "qnn-report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
