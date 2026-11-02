"""Zoom from the lens metadata the camera embeds in the SDI signal (VANC ancillary packet).

Sony HDC-5500V + HDCU-5500 with a Fujinon digital lens: DID 0x51 SDID 0x01, bytes 65-66
(big-endian) = zoom encoder, 0 at full wide .. 65535 at full tele (found with
tools/sdi_anc_probe.py, checked against the picture with tools/anc_video_correlate.py).
Where to read is set in the lens table ("sdi": {...}), so another camera only needs a
new JSON.

Needs --ffmpeg-in sdk: the packets arrive with the video frame, so the value belongs to
exactly the frame being processed (no delay buffer). [ and ] shift by whole frames in
case the camera writes the value a frame early or late.
"""
from __future__ import annotations

import math

from .base import ZoomSample, ZoomSource
from .lens_lut import LensLUT


class SdiZoomSource(ZoomSource):
    name = "sdi"

    def __init__(self, lut: LensLUT, reader=None, extender: float = 1.0, frame_offset: int = 0):
        spec = lut.sdi or {}
        self.lut = lut
        self.reader = reader          # DeckLinkSDKReader (packets_for(frame_index))
        self.did = int(str(spec.get("did", "0x51")), 0)
        self.sdid = int(str(spec.get("sdid", "0x01")), 0)
        self.line = spec.get("line")  # None: first packet found (field 1)
        self.byte = int(spec.get("byte", 65))
        self.nbytes = int(spec.get("bytes", 2))
        self.endian = str(spec.get("endian", "big"))
        self.extender = float(extender)
        self.frame_offset = int(frame_offset)
        self.frames = 0
        self.missing = 0
        self._last = ZoomSample(source=self.name)

    def raw_of(self, packets) -> float | None:
        for did, sdid, line, data in packets or ():
            if did == self.did and sdid == self.sdid and (self.line is None or line == self.line) \
                    and len(data) >= self.byte + self.nbytes:
                return float(int.from_bytes(data[self.byte:self.byte + self.nbytes], self.endian))
        return None

    def sample(self, t: float, frame=None, frame_index=None) -> ZoomSample:
        self.frames += 1
        raw = None
        if self.reader is not None and frame_index is not None:
            raw = self.raw_of(self.reader.packets_for(int(frame_index) - self.frame_offset))
        if raw is None:
            self.missing += 1
            self._last = ZoomSample(self._last.scale, False, self._last.focal_mm, self._last.raw, self.name)
            return self._last
        focal = self.lut.focal(raw) * self.extender
        self._last = ZoomSample(focal / self.lut.wide_mm, True, focal, raw, self.name)
        return self._last

    def command(self, cmd: str) -> bool:
        if cmd == "zoom_delay_inc":
            self.frame_offset += 1
        elif cmd == "zoom_delay_dec":
            self.frame_offset -= 1
        elif cmd == "zoom_extender":
            self.extender = 1.0 if self.extender > 1.0 else 2.0
        else:
            return False
        print(f"[ZOOM] {self.status()}")
        return True

    def status(self) -> str:
        raw = "-" if math.isnan(self._last.raw) else f"{self._last.raw:.0f}"
        return (f"sdi DID 0x{self.did:02X}/0x{self.sdid:02X} byte {self.byte} raw={raw} "
                f"offset={self.frame_offset:+d}f ext=x{self.extender:g} lens='{self.lut.name}'"
                f"{'' if self.lut.calibrated else ' (uncalibrated)'}"
                + ("" if self.reader is not None else " [no SDK input: use --ffmpeg-in sdk]"))
