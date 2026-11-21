#!/usr/bin/env python3
"""Fix dynamic input shapes so NNAPI (Pixel TPU / Snapdragon NPU) can take whole graphs.

Usage: python3 tools/prepare_models.py <models_dir>
Idempotent: models whose inputs are already static are left untouched. Writes in place.
"""
import os
import subprocess
import sys

import onnx

FIXED = {
    "scrfd_2.5g.onnx": ("input", "1,3,640,640"),
    "arcface_w600k_r50.onnx": ("input", "1,3,112,112"),
    "open_nsfw.onnx": ("input", "1,224,224,3"),
}


def is_static(path: str) -> bool:
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    for inp in m.graph.input:
        if inp.name in inits:
            continue
        for d in inp.type.tensor_type.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                return False
    return True


def main(models_dir: str) -> None:
    for name, (inp, shape) in FIXED.items():
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            print(f"skip {name} (missing)")
            continue
        real = os.path.realpath(path)
        if is_static(real):
            print(f"ok   {name} already static")
            continue
        tmp = real + ".fixed"
        subprocess.check_call([sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
                               "--input_name", inp, "--input_shape", shape, real, tmp])
        os.replace(tmp, real)
        print(f"fixed {name} -> {shape}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/faceswap-models"))
