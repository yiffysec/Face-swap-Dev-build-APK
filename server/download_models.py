#!/usr/bin/env python3
"""
Downloads model weights into `models/` per models_manifest.yaml, with resume support and CRC32 verification
against each entry's .hash file — the same resumable-download contract as the Android client's
ModelManager.kt:download(), independently implemented here since the two run on different platforms.

Usage:
    python3 download_models.py                   # fetch the "core" set (required + recommended)
    python3 download_models.py --set full         # fetch everything that's remotely servable
    python3 download_models.py --set swappers,enhancers
    python3 download_models.py --only hyperswap_1a_256 gpen_bfr_256
    python3 download_models.py --models-dir /data/models
"""
from __future__ import annotations

import argparse
import sys
import zipfile
import zlib
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import catalog  # noqa: E402

CHUNK = 1 << 20


def crc32_hex(path: Path) -> str:
    crc = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            crc = zlib.crc32(buf, crc)
    return f"{crc:08x}"


def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.text.strip().lower()


def download_one(entry: catalog.ModelEntry, models_dir: Path, force: bool) -> None:
    target = models_dir / entry.filename
    if target.exists() and not force:
        print(f"  {entry.id}: already have {entry.filename}")
        return

    part = models_dir / (entry.filename + ".part")
    print(f"  {entry.id}: downloading {entry.url}")
    with requests.get(entry.url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            downloaded = 0
            total = int(r.headers.get("Content-Length", 0))
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()

    if entry.url.endswith(".zip"):
        print(f"  {entry.id}: extracting archive")
        with zipfile.ZipFile(part) as zf:
            for info in zf.infolist():
                base = Path(info.filename).name
                if not base or not (base.endswith(".onnx") or base.endswith(".data")):
                    continue
                dest = target if base.endswith(".onnx") else (models_dir / base)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        part.unlink()
        if not target.exists():
            raise RuntimeError(f"{entry.id}: no .onnx found in archive")
    else:
        if entry.hash_url:
            try:
                expected = fetch_text(entry.hash_url)
                actual = crc32_hex(part)
                if len(expected) == 8 and expected != actual:
                    part.unlink()
                    raise RuntimeError(f"{entry.id}: checksum mismatch (expected {expected}, got {actual})")
            except requests.RequestException:
                print(f"  {entry.id}: could not fetch hash file, skipping verification")
        part.rename(target)
    print(f"  {entry.id}: done ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="core", help="core | full | comma-separated group list (default: core)")
    ap.add_argument("--only", nargs="*", default=None, help="exact model ids to fetch, ignoring --set")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--manifest", default=str(catalog.DEFAULT_MANIFEST_PATH))
    ap.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = ap.parse_args()

    entries = catalog.load_manifest(Path(args.manifest))
    remote_entries = [e for e in entries if catalog.is_remotely_servable(e.id)]

    if args.only:
        wanted = set(args.only)
        selected = [e for e in entries if e.id in wanted]
        missing = wanted - {e.id for e in selected}
        if missing:
            print(f"unknown model id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1
    elif args.set == "full":
        selected = remote_entries
    elif args.set == "core":
        selected = [e for e in remote_entries if e.group in ("required", "recommended")]
    else:
        groups = {g.strip() for g in args.set.split(",")}
        selected = [e for e in remote_entries if e.group in groups]

    if not selected:
        print("nothing to download for that selection", file=sys.stderr)
        return 1

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(e.size_bytes for e in selected)
    print(f"fetching {len(selected)} model(s), ~{total_bytes / 1e9:.2f} GB, into {models_dir}/")
    failures = []
    for entry in selected:
        try:
            download_one(entry, models_dir, args.force)
        except Exception as e:  # noqa: BLE001
            print(f"  {entry.id}: FAILED - {e}", file=sys.stderr)
            failures.append(entry.id)

    if failures:
        print(f"\n{len(failures)} model(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nall done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
