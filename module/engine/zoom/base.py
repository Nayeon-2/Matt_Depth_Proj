"""Common pieces for every zoom source.

A zoom source turns "something that knows the lens zoom" (FreeD box, serial lens
port, image analysis, keyboard) into a ZoomSample for the current video frame.
The rest of the pipeline only sees ZoomSample, so sources can be swapped with
--zoom-source without touching the loop, the distance estimator or Unreal.

    scale = focal / wide-end focal      (1.0 at full wide, includes the extender)
"""
from __future__ import annotations

import bisect
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .lens_lut import LensLUT


@dataclass
class ZoomSample:
    scale: float = 1.0          # focal / wide focal; held at the last good value when invalid
    valid: bool = False         # False: no fresh data (Unreal keeps the camera as is)
    focal_mm: float = math.nan
    raw: float = math.nan       # encoder value before the lens table (encoder sources)
    source: str = "off"


class ZoomSource:
    """Base class. Subclasses override sample(); the rest is optional."""

    name = "off"

    def start(self) -> "ZoomSource":
        return self

    def stop(self):
        pass

    def sample(self, t: float, frame=None, frame_index=None) -> ZoomSample:
        """Zoom for a frame that arrived at time t (time.perf_counter()).
        frame_index: source frame number when the input knows it (file playback)."""
        return ZoomSample(source=self.name)

    def command(self, cmd: str) -> bool:
        """Runtime key command ("zoom_delay_inc", ...). True if handled."""
        return False

    def status(self) -> str:
        return self.name


class TimedBuffer:
    """Thread-safe (time, value) history with linear interpolation at any time."""

    def __init__(self, keep_s: float = 3.0):
        self.keep_s = keep_s
        self._t: deque[float] = deque()
        self._v: deque[float] = deque()
        self._lock = threading.Lock()

    def push(self, t: float, v: float):
        with self._lock:
            if self._t and t <= self._t[-1]:
                t = self._t[-1] + 1e-6
            self._t.append(t)
            self._v.append(v)
            while self._t and self._t[0] < t - self.keep_s:
                self._t.popleft()
                self._v.popleft()

    def at(self, t: float, max_gap_s: float = 0.25) -> Optional[float]:
        """Value at time t, or None when there is no data within max_gap_s of t."""
        with self._lock:
            if not self._t:
                return None
            ts, vs = list(self._t), list(self._v)
        i = bisect.bisect_left(ts, t)
        if i == 0:
            return vs[0] if ts[0] - t <= max_gap_s else None
        if i == len(ts):
            return vs[-1] if t - ts[-1] <= max_gap_s else None
        t0, t1 = ts[i - 1], ts[i]
        if t1 - t0 > max_gap_s:
            return vs[i - 1] if t - t0 <= max_gap_s else vs[i]
        k = (t - t0) / (t1 - t0)
        return vs[i - 1] + (vs[i] - vs[i - 1]) * k

    def last_time(self) -> Optional[float]:
        with self._lock:
            return self._t[-1] if self._t else None


class EncoderZoomSource(ZoomSource):
    """Base for sources that deliver a raw lens encoder value with a receive time.

    Handles what every encoder needs: history buffer, fixed video delay, lens table,
    extender and hold-last-value. A subclass only calls self.push(raw) from its
    receive thread (FreeD UDP today; a serial lens port or OSC would work the same).

    delay_ms: how much later the video frame arrives than the matching encoder value
    (camera -> SDI -> OBS -> capture). Tune live with [ and ].
    """

    name = "encoder"

    def __init__(self, lut: LensLUT, delay_ms: float = 0.0, extender: float = 1.0,
                 delay_step_ms: float = 5.0, stale_s: float = 0.5):
        self.lut = lut
        self.delay_ms = float(delay_ms)
        self.extender = float(extender)
        self.delay_step_ms = float(delay_step_ms)
        self.stale_s = float(stale_s)
        self.buf = TimedBuffer()
        self.packets = 0
        self._last = ZoomSample(source=self.name)

    def push(self, raw: float, t: Optional[float] = None):
        self.buf.push(time.perf_counter() if t is None else t, float(raw))
        self.packets += 1

    def sample(self, t: float, frame=None, frame_index=None) -> ZoomSample:
        raw = self.buf.at(t - self.delay_ms / 1000.0, max_gap_s=self.stale_s)
        if raw is None:
            self._last = ZoomSample(self._last.scale, False, self._last.focal_mm, self._last.raw, self.name)
            return self._last
        focal = self.lut.focal(raw) * self.extender
        self._last = ZoomSample(focal / self.lut.wide_mm, True, focal, raw, self.name)
        return self._last

    def command(self, cmd: str) -> bool:
        if cmd == "zoom_delay_inc":
            self.delay_ms += self.delay_step_ms
        elif cmd == "zoom_delay_dec":
            self.delay_ms = max(0.0, self.delay_ms - self.delay_step_ms)
        elif cmd == "zoom_extender":
            self.extender = 1.0 if self.extender > 1.0 else 2.0
        else:
            return False
        print(f"[ZOOM] {self.status()}")
        return True

    def status(self) -> str:
        return (f"{self.name} delay={self.delay_ms:.0f}ms ext=x{self.extender:g} "
                f"lens='{self.lut.name}'{'' if self.lut.calibrated else ' (uncalibrated)'}")


class ManualZoomSource(ZoomSource):
    """Keyboard zoom (- / = / z) for testing Unreal without any lens hardware."""

    name = "manual"

    def __init__(self, lut: LensLUT, step: float = 1.1):
        self.lut = lut
        self.step = step
        self.scale = 1.0

    def sample(self, t: float, frame=None, frame_index=None) -> ZoomSample:
        return ZoomSample(self.scale, True, self.lut.wide_mm * self.scale, math.nan, self.name)

    def command(self, cmd: str) -> bool:
        if cmd == "zoom_in":
            self.scale = min(self.scale * self.step, 256.0)
        elif cmd == "zoom_out":
            self.scale = max(self.scale / self.step, 1.0)
        elif cmd == "zoom_reset":
            self.scale = 1.0
        else:
            return False
        print(f"[ZOOM] manual x{self.scale:.2f}")
        return True


class ReplayZoomSource(ZoomSource):
    """Per-frame zoom from a CSV (frame,scale[,...]) for file playback tests.

    The scale of input frame k is row (k + offset) mod N, so a looping clip stays in
    step. [ and ] shift the offset by one frame to check sync by eye.
    """

    name = "replay"

    def __init__(self, path, lut: LensLUT, offset: int = 0):
        import csv
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        self.scales = [float(r["scale"]) for r in sorted(rows, key=lambda r: int(r["frame"]))]
        if not self.scales:
            raise ValueError(f"empty zoom replay file: {path}")
        self.path, self.lut, self.offset = str(path), lut, int(offset)

    def sample(self, t: float, frame=None, frame_index=None) -> ZoomSample:
        if frame_index is None or frame_index < 0:
            return ZoomSample(1.0, False, source=self.name)
        s = self.scales[(int(frame_index) + self.offset) % len(self.scales)]
        return ZoomSample(s, True, self.lut.wide_mm * s, float(frame_index), self.name)

    def command(self, cmd: str) -> bool:
        if cmd not in ("zoom_delay_inc", "zoom_delay_dec"):
            return False
        self.offset += -1 if cmd == "zoom_delay_inc" else 1   # more delay = use an earlier row
        print(f"[ZOOM] {self.status()}")
        return True

    def status(self) -> str:
        return f"replay '{Path(self.path).name}' frames={len(self.scales)} offset={self.offset:+d}"
