#!/usr/bin/env python3
"""Fetch Qualcomm AI Hub models (zip with external data) and store them as single-file ONNX in the model cache.

Usage: python3 tools/fetch_qualcomm.py <models_dir>
"""
import io
import os
import sys
import tempfile
import urllib.request
import zipfile

import onnx

MODELS = {
    # BSD-3 (Real-ESRGAN). 1.2 M params, 128² tiles, ×4. Qualcomm: 4.3 ms float / 1.2 ms w8a8 per tile on SD 8 Gen 3.
    "real_esrgan_general_x4v3": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/real_esrgan_general_x4v3/releases/v0.67.0/real_esrgan_general_x4v3-onnx-float.zip",
}


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for mid, url in MODELS.items():
        dst = os.path.join(out_dir, mid + ".onnx")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        print(f"fetching {mid}")
        data = urllib.request.urlopen(url, timeout=120).read()
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
            src = next(os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f.endswith(".onnx"))
            m = onnx.load(src)  # pulls in the external .data file
            onnx.save(m, dst + ".part")
            os.replace(dst + ".part", dst)
        print(f"  {dst} {os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
