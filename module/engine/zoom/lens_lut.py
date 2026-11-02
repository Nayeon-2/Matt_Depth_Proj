"""Lens table: raw zoom encoder value -> focal length (mm).

File format (JSON):
    {
      "name": "Fujinon UA107x8.4",
      "sensor_width_mm": 9.59,          # 2/3" 16:9
      "calibrated": false,              # true once the points come from a real calibration
      "points": [[raw, focal_mm], ...]  # any order, raw must be unique
      "sdi": {"did": "0x51", "sdid": "0x01", "byte": 65, "bytes": 2}   # optional: where --zoom-source sdi reads raw
    }

Focal length is interpolated in log space: zoom encoders are close to linear in
log(focal), so few points are needed for a good fit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class LensLUT:
    def __init__(self, points, sensor_width_mm: float = 9.59, name: str = "lens", calibrated: bool = False):
        pts = sorted((float(r), float(f)) for r, f in points)
        if len(pts) < 2:
            raise ValueError("lens table needs at least 2 points")
        self.raw = np.array([p[0] for p in pts])
        self.log_f = np.log(np.array([p[1] for p in pts]))
        if np.any(np.diff(self.raw) <= 0):
            raise ValueError("lens table raw values must be unique")
        self.sensor_width_mm = float(sensor_width_mm)
        self.name = name
        self.calibrated = bool(calibrated)
        self.wide_mm = float(np.exp(self.log_f.min()))
        self.sdi: dict = {}   # where the raw value sits in the SDI metadata (sdi source)

    @classmethod
    def load(cls, path: str | Path) -> "LensLUT":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        lut = cls(d["points"], d.get("sensor_width_mm", 9.59), d.get("name", Path(path).stem),
                  d.get("calibrated", False))
        lut.sdi = dict(d.get("sdi") or {})
        return lut

    def focal(self, raw: float) -> float:
        return float(np.exp(np.interp(raw, self.raw, self.log_f)))

    def raw_for_focal(self, focal_mm: float) -> float:
        """Inverse lookup (simulator, calibration). The table must be monotonic."""
        lf = math.log(focal_mm)
        order = np.argsort(self.log_f)
        return float(np.interp(lf, self.log_f[order], self.raw[order]))

    def hfov_deg(self, focal_mm: float) -> float:
        return math.degrees(2.0 * math.atan(self.sensor_width_mm / (2.0 * focal_mm)))
