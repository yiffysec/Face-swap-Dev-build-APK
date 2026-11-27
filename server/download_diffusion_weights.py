#!/usr/bin/env python3
"""
Downloads weights for one or more diffusion_manifest.yaml backend ids into <weights_dir>/<id>/. Separate from
download_models.py (the ONNX catalog's downloader) because these come from Hugging Face *repos* (via
huggingface_hub, which handles resumable/parallel chunked downloads and repo-relative file paths) rather than
single-file URLs with a sidecar .hash — a different enough shape that bolting it onto download_models.py would
have made that file harder to read for no shared benefit.

Usage:
    python3 download_diffusion_weights.py instantid_sdxl pulid_sdxl       # just these two
    python3 download_diffusion_weights.py --all                          # everything in the manifest
    python3 download_diffusion_weights.py --all --skip reface_sd14       # everything except reface (e.g. to
                                                                          # respect its non-commercial license on
                                                                          # a box that won't use it)

Does not download `base_model` itself (e.g. stabilityai/stable-diffusion-xl-base-1.0) — that's pulled lazily by
`diffusers.from_pretrained` the first time a backend actually loads, same as every other diffusers script does; a
dedicated pre-fetch step for the base would just duplicate diffusers' own caching, which huggingface_hub already
shares with this script's downloads (both honor HF_HOME / the default ~/.cache/huggingface).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion_app import catalog  # noqa: E402


def download_one(entry: catalog.BackendEntry, weights_dir: Path) -> None:
    from huggingface_hub import hf_hub_download
    import requests

    dest = weights_dir / entry.id
    dest.mkdir(parents=True, exist_ok=True)
    print(f"--- {entry.id} ({entry.display_name}) ---")
    if entry.shares_base_with:
        print(f"    shares its base checkpoint with {entry.shares_base_with} (downloaded once, lazily, by whichever loads first)")

    for w in entry.weights:
        if w.repo:
            for filename in (w.files or [None]):
                if filename is None:
                    print(f"    hf repo (whole): {w.repo}")
                    from huggingface_hub import snapshot_download
                    snapshot_download(w.repo, local_dir=str(dest))
                else:
                    print(f"    hf file: {w.repo}/{filename}")
                    local_path = hf_hub_download(w.repo, filename, local_dir=str(dest))
                    # hf_hub_download preserves the repo-relative subfolder structure (e.g.
                    # "ControlNetModel/diffusion_pytorch_model.safetensors") under local_dir, which is exactly the
                    # layout instantid.py's CONTROLNET_SUBFOLDER lookup expects.
                    print(f"      -> {local_path}")
        elif w.url:
            filename = w.filename or w.url.rsplit("/", 1)[-1]
            target = dest / filename
            if target.exists():
                print(f"    already have {filename}, skipping")
                continue
            print(f"    url: {w.url}")
            resp = requests.get(w.url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            print(f"      -> {target}")

    print(f"    done (~{entry.approx_disk_gb:.1f}GB on disk, ~{entry.approx_vram_gb:.1f}GB VRAM at inference time)")
    if "non-commercial" in entry.license.lower() or "research" in entry.license.lower():
        print(f"    NOTE: license says {entry.license!r} — do not enable this backend on a box serving commercial traffic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="backend ids to download (default: none unless --all)")
    ap.add_argument("--all", action="store_true", help="download every backend in diffusion_manifest.yaml")
    ap.add_argument("--skip", nargs="*", default=[], help="backend ids to exclude even with --all")
    ap.add_argument("--manifest", default=None, help="path to diffusion_manifest.yaml (default: alongside this script)")
    ap.add_argument("--weights-dir", default="diffusion/weights", help="where to put <id>/ subdirectories")
    args = ap.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else catalog.DEFAULT_MANIFEST_PATH
    entries = catalog.load_manifest(manifest_path)
    entries_by_id = catalog.by_id(entries)

    if args.all:
        wanted = [e.id for e in entries if e.id not in set(args.skip)]
    else:
        wanted = args.ids
    if not wanted:
        ap.error("nothing to do: pass backend ids, or --all")

    unknown = [i for i in wanted if i not in entries_by_id]
    if unknown:
        ap.error(f"unknown backend id(s): {unknown} — known ids: {sorted(entries_by_id)}")

    weights_dir = Path(args.weights_dir)
    for model_id in wanted:
        download_one(entries_by_id[model_id], weights_dir)
    print(f"\ndone: {len(wanted)} backend(s) downloaded to {weights_dir}/")


if __name__ == "__main__":
    main()
