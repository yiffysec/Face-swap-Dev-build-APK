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

# small → large: the biggest GANs go last, so a memory/time skip never costs the cheap wins
DEFAULT_IDS = [
    "yunet_2023_mar", "scrfd_2.5g", "span_kendata_x4", "real_esrgan_general_x4v3", "open_nsfw", "hrffa", "modnet",
    "bisenet_resnet_18", "real_esrgan_x2_fp16", "real_esrgan_x4_fp16", "gpen_bfr_256", "arcface_w600k_r50",
    "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "gpen_bfr_512", "gfpgan_1.4",
]
MAPPING_VERSION = 2  # 2 = tensors addressed by signature name (LiteRT CompiledModel runs by signature)

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


MAX_RSS = int(float(os.environ.get("CONVERT_MAX_GB", "9")) * (1 << 30))
MAX_SECONDS = int(os.environ.get("CONVERT_MAX_MIN", "20")) * 60


class Guarded:
    def __init__(self, code, out):
        self.returncode = code; self.stdout = out; self.stderr = ""


def run_guarded(cmd):
    """onnx2tf on a big GAN can eat the runner's RAM and take the whole CI job down. Watch the process tree and
    kill it above MAX_RSS or MAX_SECONDS – the model is then just skipped (no GPU twin)."""
    import psutil, time
    log = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    t0 = time.time(); reason = None
    while p.poll() is None:
        time.sleep(2)
        try:
            proc = psutil.Process(p.pid)
            rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
        except psutil.Error:
            continue
        if rss > MAX_RSS: reason = f"memory limit ({rss / 2**30:.1f} GB)"
        elif time.time() - t0 > MAX_SECONDS: reason = "time limit"
        if reason:
            for c in psutil.Process(p.pid).children(recursive=True):
                c.kill()
            p.kill(); p.wait()
            return Guarded(1, "skipped: " + reason)
    log.seek(0)
    return Guarded(p.returncode, log.read())


def convert(models_dir: str, mid: str) -> str:
    src = os.path.join(models_dir, f"{mid}.onnx")
    if not os.path.exists(src):
        return "missing"
    out_tfl = os.path.join(models_dir, f"{mid}.tflite")
    fail_mark = out_tfl + ".failed"
    stamp = f"{MAPPING_VERSION}:{os.path.getsize(src)}"
    if os.path.exists(fail_mark) and open(fail_mark).read().split("\n")[0] == stamp:
        return "skipped (failed before: " + open(fail_mark).read().split("\n", 1)[-1][:120] + ")"
    if os.path.exists(out_tfl) and os.path.exists(out_tfl + ".json"):
        try:
            if json.load(open(out_tfl + ".json")).get("version") == MAPPING_VERSION:
                return "cached"
        except Exception:
            pass
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
        r = run_guarded(cmd)
        fp16 = os.path.join(tmp, "out", f"{mid}_float16.tflite")
        if r.returncode != 0 or not os.path.exists(fp16):
            return "convert-failed: " + (r.stderr or r.stdout)[-300:].replace("\n", " ")
        # verify against ONNX Runtime through the SIGNATURE runner (same addressing the app uses)
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=fp16)
        sigs = it.get_signature_list()
        if not sigs:
            return "no signature"
        sig_key = next(iter(sigs))
        runner = it.get_signature_runner(sig_key)
        in_details = runner.get_input_details()
        rng = np.random.default_rng(0)
        feeds = feeds_for(sess, rng)
        ref = sess.run(None, feeds)
        mapping = {"version": MAPPING_VERSION, "signature": sig_key, "inputs": [], "outputs": []}
        call = {}
        for sname, d in in_details.items():
            name = next((k for k in feeds if sname == k or sname.startswith(k) or k in sname or d["name"] == k), None)
            if name is None and len(feeds) == 1:
                name = next(iter(feeds))
            if name is None:
                return f"unmatched input {sname}"
            x = feeds[name]
            nhwc = list(x.shape) != [int(v) for v in d["shape"]]
            call[sname] = x.transpose(0, 2, 3, 1) if nhwc else x
            mapping["inputs"].append({"onnx": name, "tflite": sname, "nhwc": nhwc, "shape": [int(v) for v in x.shape]})
        outs = runner(**call)
        worst = 0.0
        used = set()
        for o, rv in zip(sess.get_outputs(), ref):
            match = None
            for sname, y in outs.items():
                if sname in used:
                    continue
                for nhwc, cand in ((False, y), (True, y.transpose(0, 3, 1, 2) if y.ndim == 4 else None)):
                    if cand is not None and cand.shape == rv.shape:
                        err = float(np.abs(cand - rv).max() / (np.abs(rv).max() + 1e-6))
                        if match is None or err < match[0]:
                            match = (err, sname, nhwc)
            if match is None:
                return f"unmatched output {o.name}"
            used.add(match[1])
            worst = max(worst, match[0])
            mapping["outputs"].append({"onnx": o.name, "tflite": match[1], "nhwc": match[2], "shape": list(rv.shape)})
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
        # remember hard failures so later CI runs don't spend 20 min re-trying them (cleared by a converter change)
        src = os.path.join(models_dir, f"{mid}.onnx")
        if os.path.exists(src) and not status.startswith(("ok", "cached", "skipped", "missing")):
            with open(os.path.join(models_dir, f"{mid}.tflite.failed"), "w") as f:
                f.write(f"{MAPPING_VERSION}:{os.path.getsize(src)}\n{status}")
        print(f"{mid:28s} {status}", flush=True)


if __name__ == "__main__":
    main()
