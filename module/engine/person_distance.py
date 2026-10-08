"""Person distance estimation from pose keypoints (method D).

- Body size: shoulder width / torso length in pixels -> distance (pinhole model).
- Foot contact: when the feet are visible, the ankle row on a flat floor gives a
  body-size independent distance. The floor geometry (camera height, pitch) is
  self-calibrated from body-size samples and saved to a JSON file.
- While both are available a per-person scale (foot / body) is learned, so the
  body-size estimate stays correct after the feet leave the frame.
- Spikes are rejected and the result is smoothed with a One Euro filter.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

# COCO keypoint indices
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16

# Encoding of the distance inside the matte frame (top-left block, alpha = 0)
CODE_BLOCK = 4


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _a(cutoff: float, dt: float) -> float:
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def reset(self):
        self.x = None
        self.dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self.x is None:
            self.x = x
            return x
        dt = max(dt, 1e-3)
        dx = (x - self.x) / dt
        self.dx += self._a(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._a(cutoff, dt) * (x - self.x)
        return self.x


class PersonDistanceEstimator:
    def __init__(
        self,
        width: int,
        height: int,
        hfov_deg: float = 60.0,
        shoulder_m: float = 0.39,
        torso_m: float = 0.52,
        kp_conf: float = 0.5,
        cam_height_m: Optional[float] = None,
        cam_pitch_deg: Optional[float] = None,
        calib_path: Optional[str] = None,
        lost_reset_s: float = 1.0,
    ):
        self.W, self.H = int(width), int(height)
        self.f = (self.W / 2) / math.tan(math.radians(hfov_deg / 2))
        self.cy = self.H / 2
        self.shoulder_m, self.torso_m, self.kp_conf = shoulder_m, torso_m, kp_conf
        self.lost_reset_s = lost_reset_s

        self.calib_path = Path(calib_path) if calib_path else None
        self.ground: Optional[tuple[float, float]] = None   # (height m, pitch rad)
        self.ground_fixed = cam_height_m is not None and cam_pitch_deg is not None
        if self.ground_fixed:
            self.ground = (float(cam_height_m), math.radians(float(cam_pitch_deg)))
        elif self.calib_path and self.calib_path.exists():
            try:
                d = json.loads(self.calib_path.read_text(encoding="utf-8"))
                self.ground = (float(d["cam_height_m"]), math.radians(float(d["cam_pitch_deg"])))
                print(f"[DIST] ground calibration loaded: h={d['cam_height_m']:.2f} m, pitch={d['cam_pitch_deg']:.1f} deg")
            except Exception as e:
                print("[DIST] ground calibration load failed:", e)
        self._samples: list[tuple[float, float]] = []   # (foot row angle, body distance)
        self._last_fit_n = 0

        self.filter = OneEuro(min_cutoff=1.0, beta=0.3)
        self.person_scale = 1.0
        self.z: Optional[float] = None
        self.valid = False
        self.source = "none"
        self._last_seen = 0.0
        self._last_t: Optional[float] = None
        self._spike_count = 0

    # ---------- geometry ----------
    def _body_distance(self, k: np.ndarray, c: np.ndarray) -> float:
        vals = []
        if c[L_SH] > self.kp_conf and c[R_SH] > self.kp_conf:
            sh = float(np.linalg.norm(k[L_SH] - k[R_SH]))
            if sh > 2:
                vals.append(self.f * self.shoulder_m / sh)
            if c[L_HIP] > self.kp_conf and c[R_HIP] > self.kp_conf:
                torso = float(np.linalg.norm((k[L_SH] + k[R_SH]) / 2 - (k[L_HIP] + k[R_HIP]) / 2))
                if torso > 2:
                    vals.append(self.f * self.torso_m / torso)
        return float(np.median(vals)) if vals else math.nan

    def _foot_row(self, k: np.ndarray, c: np.ndarray, bbox) -> float:
        rows = [k[i][1] for i in (L_ANK, R_ANK) if c[i] > self.kp_conf]
        if not rows:
            return math.nan
        y = float(max(rows))
        bottom = float(bbox[3]) if bbox is not None else y
        if y > self.H * 0.96 or bottom > self.H * 0.985:
            return math.nan          # feet touching the frame edge: not reliable
        return y

    def _row_angle(self, y: float) -> float:
        return math.atan((y - self.cy) / self.f)

    def _foot_distance(self, y: float) -> float:
        h, pitch = self.ground
        a = pitch + self._row_angle(y)
        return h / math.tan(a) if a > math.radians(0.5) else math.nan

    def _fit_ground(self):
        """Grid-search pitch, closed-form height: minimize sum (h*g_i - z_i)^2, g_i = 1/tan(pitch + phi_i)."""
        phi = np.array([s[0] for s in self._samples])
        z = np.array([s[1] for s in self._samples])
        best = None
        for pitch in np.radians(np.arange(-5.0, 45.0, 0.25)):
            a = pitch + phi
            if np.any(a <= math.radians(0.5)):
                continue
            g = 1.0 / np.tan(a)
            h = float((g @ z) / (g @ g))
            err = float(np.median(np.abs(np.log(h * g / z))))
            if best is None or err < best[0]:
                best = (err, h, float(pitch))
        if best is None or not (0.3 < best[1] < 6.0):
            return
        err, h, pitch = best
        self.ground = (h, pitch)
        print(f"[DIST] ground self-calibrated from {len(z)} samples: h={h:.2f} m, "
              f"pitch={math.degrees(pitch):.1f} deg, median err={err*100:.1f}%")
        if self.calib_path:
            try:
                self.calib_path.write_text(json.dumps(
                    {"cam_height_m": h, "cam_pitch_deg": math.degrees(pitch), "samples": len(z)}, indent=2),
                    encoding="utf-8")
            except Exception as e:
                print("[DIST] ground calibration save failed:", e)

    # ---------- main update ----------
    def update(self, kps: Optional[np.ndarray], conf: Optional[np.ndarray], bbox=None,
               now: Optional[float] = None) -> tuple[float, bool]:
        now = time.time() if now is None else now
        dt = (now - self._last_t) if self._last_t is not None else 1 / 30
        self._last_t = now

        z_body = z_foot = math.nan
        y_foot = math.nan
        if kps is not None and conf is not None:
            k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
            c = np.asarray(conf, dtype=np.float32).reshape(-1)
            if k.shape[0] >= 17:
                z_body = self._body_distance(k, c)
                y_foot = self._foot_row(k, c, bbox)

        if math.isnan(z_body) and math.isnan(y_foot):
            if now - self._last_seen > self.lost_reset_s:
                self.valid = False
                self.source = "none"
                self.person_scale = 1.0
                self.filter.reset()
                self._spike_count = 0
            return (self.z if self.z is not None else 0.0), self.valid
        self._last_seen = now

        # ground self-calibration samples (only while not fixed by the user)
        if not self.ground_fixed and not math.isnan(y_foot) and not math.isnan(z_body):
            self._samples.append((self._row_angle(y_foot), z_body))
            self._samples = self._samples[-3000:]
            n = len(self._samples)
            if n >= 150 and n - self._last_fit_n >= 150:
                zs = np.array([s[1] for s in self._samples])
                # the fit is only well conditioned when the samples cover a range of distances
                if np.percentile(zs, 95) / np.percentile(zs, 5) >= 1.6:
                    self._last_fit_n = n
                    self._fit_ground()

        if self.ground is not None and not math.isnan(y_foot):
            z_foot = self._foot_distance(y_foot)

        if not math.isnan(z_foot) and not math.isnan(z_body):
            ratio = z_foot / z_body
            if 0.5 < ratio < 2.0:
                self.person_scale += 0.1 * (ratio - self.person_scale)

        if not math.isnan(z_foot) and not math.isnan(z_body):
            # geometric mean of both cues: smooth hand-over when the feet appear / disappear
            meas = math.sqrt(z_foot * z_body * self.person_scale)
            self.source = "foot+body"
        elif not math.isnan(z_foot):
            meas, self.source = z_foot, "foot"
        elif not math.isnan(z_body):
            meas, self.source = z_body * self.person_scale, "body"
        else:
            return (self.z if self.z is not None else 0.0), self.valid

        # spike rejection: ignore short jumps larger than ~35 %
        if self.filter.x is not None and abs(math.log(meas / self.filter.x)) > 0.35 and self._spike_count < 3:
            self._spike_count += 1
            return self.z, self.valid
        self._spike_count = 0

        self.z = self.filter(meas, dt)
        self.valid = True
        return self.z, self.valid


def encode_distance(frame_bgra: np.ndarray, z_m: float, valid: bool):
    """Write the distance into the top-left corner of the matte (alpha 0, invisible).

    Block 0 (x 0..3):  R = high byte, G = low byte of millimeters (0..65535), B = 255 when valid.
    Block 1 (x 4..7):  signature, pure green (R=0, G=255, B=0), so receivers can tell
                       an encoded frame from ordinary image content.
    """
    mm = int(min(max(z_m * 1000.0, 0), 65535)) if valid else 0
    blk = frame_bgra[:CODE_BLOCK, :CODE_BLOCK]
    blk[..., 0] = 255 if valid else 0   # B: valid flag
    blk[..., 1] = mm & 0xFF             # G: low byte
    blk[..., 2] = (mm >> 8) & 0xFF      # R: high byte
    blk[..., 3] = 0                     # A: keep invisible
    sig = frame_bgra[:CODE_BLOCK, CODE_BLOCK:2 * CODE_BLOCK]
    sig[..., 0] = 0
    sig[..., 1] = 255
    sig[..., 2] = 0
    sig[..., 3] = 0


def _put16(frame_bgra: np.ndarray, block: int, value01: float, flag: bool):
    v = int(round(min(max(value01, 0.0), 1.0) * 65535))
    blk = frame_bgra[:CODE_BLOCK, block * CODE_BLOCK:(block + 1) * CODE_BLOCK]
    blk[..., 0] = 255 if flag else 0    # B: flag
    blk[..., 1] = v & 0xFF              # G: low byte
    blk[..., 2] = (v >> 8) & 0xFF       # R: high byte
    blk[..., 3] = 0                     # A: keep invisible


def encode_logo(frame_bgra: np.ndarray, phase: float, active: bool, anchor_xy: tuple[float, float]):
    """Logo trigger data next to the distance code (all invisible, alpha 0).

    Block 2 (x 8..11):  R,G = animation phase 0..1 (16 bit), B = 255 while the logo is playing.
    Block 3 (x 12..15): R,G = anchor x in 0..1 of the frame width (person's chest).
    Block 4 (x 16..19): R,G = anchor y in 0..1 of the frame height (from the top).
    """
    _put16(frame_bgra, 2, phase, active)
    _put16(frame_bgra, 3, anchor_xy[0], True)
    _put16(frame_bgra, 4, anchor_xy[1], True)


def chest_anchor(kps: Optional[np.ndarray], conf: Optional[np.ndarray], width: int, height: int,
                 kp_conf: float = 0.5) -> Optional[tuple[float, float]]:
    """Normalized screen point at the middle of the torso (shoulders/hips), or None."""
    if kps is None or conf is None:
        return None
    k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    c = np.asarray(conf, dtype=np.float32).reshape(-1)
    if k.shape[0] < 17 or c[L_SH] < kp_conf or c[R_SH] < kp_conf:
        return None
    sh = (k[L_SH] + k[R_SH]) / 2
    if c[L_HIP] > kp_conf and c[R_HIP] > kp_conf:
        p = sh * 0.6 + ((k[L_HIP] + k[R_HIP]) / 2) * 0.4
    else:
        p = sh + np.array([0.0, float(np.linalg.norm(k[L_SH] - k[R_SH])) * 0.6], dtype=np.float32)
    return float(p[0] / width), float(p[1] / height)
