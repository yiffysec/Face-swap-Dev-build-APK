#!/usr/bin/env python3
"""Convert ONNX models to TFLite twins for the LiteRT GPU backend (Pixel / Mali, Adreno).

For every model id given, writes <id>.tflite (fp16 weights) and <id>.tflite.json (tensor mapping) next to the
ONNX file, but only if the converted model matches ONNX Runtime numerically. Models that fail to convert or
verify are skipped; the app then simply doesn't offer LiteRT GPU for them.

Usage: python3 tools/convert_tflite.py <models_dir> [id ...]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

DEFAULT_IDS = [
    "yunet_2023_mar", "scrfd_2.5g", "arcface_w600k_r50", "hrffa", "hyperswap_1a_256", "hyperswap_1b_256",
    "hyperswap_1c_256", "bisenet_resnet_18", "gpen_bfr_256", "gpen_bfr_512", "gfpgan_1.4", "modnet",
    "span_kendata_x4", "real_esrgan_x2_fp16", "real_esrgan_x4_fp16", "open_nsfw",
]
# dynamic dims are fixed for conversion (GPU delegates want static shapes)
FIXED = {
    "span_kendata_x4": {"input": [1, 3, 128, 128]},
    "real_esrgan_x2_fp16": {"input": [1, 3, 256, 256]},
    "real_esrgan_x4_fp16": {"input": [1, 3, 256, 256]},
    "modnet": {"input": [1, 3, 512, 512]},
    "bisenet_resnet_18": {"input": [1, 3, 512, 512]},
    "arcface_w600k_r50": {"input": [1, 3, 112, 112]},
    "open_nsfw": {"input": [1, 224, 224, 3]},
    "scrfd_2.5g": {"input": [1, 3, 640, 640]},
}


def to_fp32(src: str, dst: str) -> None:
    """onnx2tf handles fp32 graphs best: upcast fp16 initializers and drop Cast-to-fp16 nodes' effect."""
    m = onnx.load(src)
    changed = False
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))
            changed = True
    for node in m.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == onnx.TensorProto.FLOAT16:
                    a.i = onnx.TensorProto.FLOAT
                    changed = True
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == onnx.TensorProto.FLOAT16:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32)))
                    changed = True
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            changed = True
    if changed:
        del m.graph.value_info[:]
    onnx.save(m, dst)


def fix_shapes(path: str, shapes: dict) -> None:
    m = onnx.load(path)
    for inp in m.graph.input:
        if inp.name in shapes:
            for d, v in zip(inp.type.tensor_type.shape.dim, shapes[inp.name]):
                d.ClearField("dim_param")
                d.dim_value = v
    onnx.save(m, path)


def feeds_for(sess, rng):
    feeds = {}
    for i in sess.get_inputs():
        shp = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        x = rng.random(shp).astype(np.float32)
        if "source" in i.name:
            x /= np.linalg.norm(x)
        feeds[i.name] = x
    return feeds


def convert(models_dir: str, mid: str) -> str:
    src = os.path.join(models_dir, f"{mid}.onnx")
    if not os.path.exists(src):
        return "missing"
    out_tfl = os.path.join(models_dir, f"{mid}.tflite")
    if os.path.exists(out_tfl) and os.path.exists(out_tfl + ".json"):
        return "cached"
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, f"{mid}.onnx")
        to_fp32(os.path.realpath(src), work)
        if mid in FIXED:
            fix_shapes(work, FIXED[mid])
        sess = ort.InferenceSession(work, providers=["CPUExecutionProvider"])
        img_inputs = [i.name for i in sess.get_inputs() if len(i.shape) == 4 and i.shape[1] == 3]
        cmd = [sys.executable, "-m", "onnx2tf", "-i", work, "-o", os.path.join(tmp, "out"), "-n", "-osd", "-tb", "tf_converter"]
        if img_inputs:
            cmd += ["-k"] + img_inputs
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        fp16 = os.path.join(tmp, "out", f"{mid}_float16.tflite")
        if r.returncode != 0 or not os.path.exists(fp16):
            return "convert-failed: " + (r.stderr or r.stdout)[-300:].replace("\n", " ")
        # verify against ONNX Runtime
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=fp16)
        it.allocate_tensors()
        rng = np.random.default_rng(0)
        feeds = feeds_for(sess, rng)
        ref = sess.run(None, feeds)
        ins, outs = it.get_input_details(), it.get_output_details()
        mapping = {"inputs": [], "outputs": []}
        for d in ins:
            name = next((k for k in feeds if d["name"] == k or d["name"].startswith(k) or k in d["name"]), None)
            if name is None:
                return f"unmatched input {d['name']}"
            x = feeds[name]
            nhwc = list(x.shape) != [int(v) for v in d["shape"]]
            it.set_tensor(d["index"], x.transpose(0, 2, 3, 1) if nhwc else x)
            mapping["inputs"].append({"onnx": name, "tflite": d["name"], "index": len(mapping["inputs"]), "nhwc": nhwc,
                                      "shape": [int(v) for v in x.shape]})
        it.invoke()
        worst = 0.0
        used = set()
        for o, rv in zip(sess.get_outputs(), ref):
            match = None
            for j, d in enumerate(outs):
                if j in used:
                    continue
                y = it.get_tensor(d["index"])
                for nhwc, cand in ((False, y), (True, y.transpose(0, 3, 1, 2) if y.ndim == 4 else None)):
                    if cand is not None and cand.shape == rv.shape:
                        err = float(np.abs(cand - rv).max() / (np.abs(rv).max() + 1e-6))
                        if match is None or err < match[0]:
                            match = (err, j, nhwc, d["name"])
            if match is None:
                return f"unmatched output {o.name}"
            used.add(match[1])
            worst = max(worst, match[0])
            mapping["outputs"].append({"onnx": o.name, "tflite": match[3], "index": match[1], "nhwc": match[2],
                                       "shape": list(rv.shape)})
        if worst > 0.02:
            return f"verify-failed rel err {worst:.4f}"
        mapping["maxRelError"] = worst
        shutil.copy(fp16, out_tfl)
        with open(out_tfl + ".json", "w") as f:
            json.dump(mapping, f, indent=1)
        return f"ok rel err {worst:.5f} size {os.path.getsize(out_tfl) / 1e6:.1f} MB"


def main():
    models_dir = sys.argv[1]
    ids = sys.argv[2:] or DEFAULT_IDS
    for mid in ids:
        try:
            status = convert(models_dir, mid)
        except Exception as e:  # keep going: a failed model just won't have a GPU twin
            status = f"error: {e}"[:300]
        print(f"{mid:28s} {status}", flush=True)


if __name__ == "__main__":
    main()
