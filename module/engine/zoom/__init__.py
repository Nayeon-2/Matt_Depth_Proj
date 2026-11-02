"""Swappable zoom sources -> one ZoomSample per frame -> matte code block 5.

    make_zoom_source("freed", lens="lens/UA107x8.4_placeholder.json", port=40000, delay_ms=80)

Adding a source: subclass ZoomSource (frame-based, e.g. image analysis) or
EncoderZoomSource (timestamped raw encoder, e.g. a serial lens port), then add
it to make_zoom_source below. Nothing else changes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .base import EncoderZoomSource, ManualZoomSource, ReplayZoomSource, TimedBuffer, ZoomSample, ZoomSource
from .freed import FreeDPacket, FreeDZoomSource, build_d1, parse_d1
from .lens_lut import LensLUT
from .sdi import SdiZoomSource

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LENS = ROOT / "lens" / "UA107x8.4_placeholder.json"
SOURCES = ("off", "freed", "manual", "replay", "sdi")

# Matte code block 5 (x 20..23): R,G = ln(scale)/ln(ZOOM_SCALE_MAX) in 16 bit, B = 255 when valid.
ZOOM_BLOCK = 5
ZOOM_SCALE_MAX = 256.0      # UA107x8.4 with 2x extender reaches ~214x


def make_zoom_source(kind: str, lens: str | None = None, **kw) -> ZoomSource | None:
    kind = (kind or "off").lower().strip()
    if kind == "off":
        return None
    lut = LensLUT.load(lens or DEFAULT_LENS)
    if kind == "freed":
        return FreeDZoomSource(lut, **kw).start()
    if kind == "manual":
        return ManualZoomSource(lut).start()
    if kind == "replay":
        return ReplayZoomSource(lut=lut, **kw).start()
    if kind == "sdi":
        return SdiZoomSource(lut, **kw).start()
    raise ValueError(f"unknown zoom source '{kind}' (choose from {', '.join(SOURCES)})")


def encode_zoom(frame_bgra: np.ndarray, z: ZoomSample):
    from ..person_distance import _put16, put_signature
    v = math.log(max(z.scale, 1.0)) / math.log(ZOOM_SCALE_MAX)
    _put16(frame_bgra, ZOOM_BLOCK, v, z.valid)
    put_signature(frame_bgra)


def decode_zoom(frame_bgra: np.ndarray) -> tuple[float, bool]:
    """Inverse of encode_zoom (tests; Unreal does the same in C++)."""
    from ..person_distance import CODE_BLOCK
    px = frame_bgra[0, ZOOM_BLOCK * CODE_BLOCK]
    v = ((int(px[2]) << 8) | int(px[1])) / 65535.0
    return math.exp(v * math.log(ZOOM_SCALE_MAX)), bool(px[0] > 127)


__all__ = [
    "ZoomSample", "ZoomSource", "EncoderZoomSource", "ManualZoomSource", "ReplayZoomSource", "FreeDZoomSource",
    "SdiZoomSource",
    "FreeDPacket", "TimedBuffer", "LensLUT", "build_d1", "parse_d1",
    "make_zoom_source", "encode_zoom", "decode_zoom", "SOURCES", "DEFAULT_LENS",
    "ZOOM_BLOCK", "ZOOM_SCALE_MAX",
]
