"""
Binary wire format for named float32 tensors — the server side of the protocol. Mirrors the Android client's
`core/src/main/kotlin/com/faceswap/core/TensorWire.kt` byte-for-byte. If you change one, change the other and bump
VERSION in both.

Wire shape (big-endian throughout, matching Java's Data{Output,Input}Stream defaults):

    message := u16(count) frame*count
    frame   := u32(magic) u8(version) u16(name_len) utf8(name) u8(rank) i64(dim)*rank u64(elem_count) f32(elem)*elem_count

This carries tensor data only. A failed request is reported at the HTTP level instead (non-2xx status with a JSON
`{"error": "..."}` body) — see `main.py`.
"""
from __future__ import annotations

import struct
from typing import BinaryIO, Dict

import numpy as np

MAGIC = 0x46535731  # ASCII "FSW1"
VERSION = 1

TensorMap = Dict[str, np.ndarray]

_HEADER = struct.Struct(">IBH")  # magic, version, name_len
_COUNT = struct.Struct(">H")
_U64 = struct.Struct(">Q")
_U8 = struct.Struct(">B")


class WireError(ValueError):
    """Malformed tensor stream – reported to the client as HTTP 400."""


def write_map(out: BinaryIO, tensors: TensorMap) -> None:
    out.write(_COUNT.pack(len(tensors)))
    for name, arr in tensors.items():
        _write_tensor(out, name, arr)


def _write_tensor(out: BinaryIO, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    name_bytes = name.encode("utf-8")
    if len(name_bytes) > 0xFFFF:
        raise WireError(f"tensor name too long: {name}")
    if arr.ndim > 0xFF:
        raise WireError(f"tensor rank too high for {name}")
    out.write(_HEADER.pack(MAGIC, VERSION, len(name_bytes)))
    out.write(name_bytes)
    out.write(_U8.pack(arr.ndim))
    for dim in arr.shape:
        out.write(struct.pack(">q", dim))
    flat = arr.reshape(-1)
    out.write(_U64.pack(flat.size))
    out.write(flat.astype(">f4", copy=False).tobytes())  # big-endian on the wire regardless of host byte order


def read_map(data: BinaryIO) -> TensorMap:
    (count,) = _COUNT.unpack(_read_exact(data, _COUNT.size))
    out: TensorMap = {}
    for _ in range(count):
        name, arr = _read_tensor(data)
        out[name] = arr
    return out


def _read_tensor(data: BinaryIO):
    magic, version, name_len = _HEADER.unpack(_read_exact(data, _HEADER.size))
    if magic != MAGIC:
        raise WireError(f"bad tensor frame magic: 0x{magic:x}")
    if version != VERSION:
        raise WireError(f"unsupported tensor frame version {version} (expected {VERSION})")
    name = _read_exact(data, name_len).decode("utf-8")
    (rank,) = _U8.unpack(_read_exact(data, 1))
    shape = struct.unpack(f">{rank}q", _read_exact(data, 8 * rank)) if rank else ()
    (count,) = _U64.unpack(_read_exact(data, _U64.size))
    raw = _read_exact(data, count * 4)
    arr = np.frombuffer(raw, dtype=">f4").astype(np.float32).reshape(shape)
    return name, arr


def _read_exact(data: BinaryIO, n: int) -> bytes:
    buf = data.read(n)
    if len(buf) != n:
        raise WireError(f"truncated tensor stream: wanted {n} bytes, got {len(buf)}")
    return buf
