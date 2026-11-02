"""Zoom test without camera or RVM: sends a synthetic zoomed background to RVM_BG and the
zoom code to RVM_OUTPUT, exactly like rvm_main does.

The background is drawn with the same pinhole camera as Unreal's Zoom_Stage (UA107x8.4 full
wide = 59.44 deg) and contains outlines where the Marker_* cubes should appear. If zoom
tracking works, the CG cubes stay inside their outlines at every zoom level.

    python tools/zoom_spout_test.py                       # built-in sweep x1 -> x20 -> x1
    python tools/zoom_spout_test.py --source freed        # zoom from FreeD (tools/freed_sim.py)
    python tools/zoom_spout_test.py --code-lag 2          # code 2 frames late: shows misalignment
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from freed_sim import scale_at  # noqa: E402
from module.engine.zoom import ZoomSample, encode_zoom, make_zoom_source  # noqa: E402
from module.vedioio.spout_out import SpoutOutputSender  # noqa: E402

W, H = 1920, 1080
WIDE_HFOV = 2 * math.atan(9.59 / (2 * 8.4))
# Same as Tools/setup_zoom_stage.py markers: (forward cm, right cm, up cm, size cm)
MARKERS = [(3000, 0, 0, 15)]
MARKERS += [(3000, y, z, 8) for y, z in [(60, 0), (-60, 0), (0, 35), (0, -35)]]
MARKERS += [(3000, y, z, 60) for y, z in [(1200, 600), (-1200, 600), (1200, -600), (-1200, -600)]]


def draw_bg(scale: float) -> np.ndarray:
    f = (W / 2) / math.tan(WIDE_HFOV / 2) * scale
    img = np.full((H, W, 3), 40, np.uint8)
    # angular grid: a line every 1 deg (fine) and 10 deg (bold)
    for deg in range(-40, 41):
        x = int(round(W / 2 + f * math.tan(math.radians(deg))))
        y = int(round(H / 2 - f * math.tan(math.radians(deg))))
        col, th = ((90, 90, 90), 2) if deg % 10 == 0 else ((60, 60, 60), 1)
        if 0 <= x < W:
            cv2.line(img, (x, 0), (x, H), col, th)
        if 0 <= y < H:
            cv2.line(img, (0, y), (W, y), col, th)
    for X, Y, Z, size in MARKERS:
        u, v = W / 2 + f * Y / X, H / 2 - f * Z / X
        r = f * size / 2 / X
        cv2.rectangle(img, (int(u - r - 3), int(v - r - 3)), (int(u + r + 3), int(v + r + 3)), (255, 255, 255), 2)
    cv2.putText(img, f"BG x{scale:6.2f}  f={8.4 * scale:6.1f}mm", (40, H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["curve", "freed"], default="curve")
    p.add_argument("--freed-port", type=int, default=40000)
    p.add_argument("--zoom-delay-ms", type=float, default=0.0)
    p.add_argument("--max-scale", type=float, default=20.0)
    p.add_argument("--period", type=float, default=12.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--code-lag", type=int, default=0, help="send the code N frames late (sync check)")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
    args = p.parse_args()

    src = None
    if args.source == "freed":
        src = make_zoom_source("freed", port=args.freed_port, delay_ms=args.zoom_delay_ms)
    out = SpoutOutputSender("RVM_OUTPUT", W, H).start()
    bg = SpoutOutputSender("RVM_BG", W, H).start()
    print(f"[TEST] Spout RVM_OUTPUT + RVM_BG {W}x{H} @ {args.fps:g} fps, source={args.source}")

    # A person-like opaque block on the left so the person plate is visible too.
    matte_base = np.zeros((H, W, 4), np.uint8)
    cv2.rectangle(matte_base, (180, 380), (420, 1080), (60, 160, 230, 255), -1)
    cv2.circle(matte_base, (300, 300), 80, (60, 160, 230, 255), -1)

    lagq: deque[ZoomSample] = deque(maxlen=args.code_lag + 1)
    t0 = time.perf_counter()
    n, next_t = 0, t0
    try:
        while True:
            now = time.perf_counter()
            if args.seconds and now - t0 > args.seconds:
                break
            z = (src.sample(now) if src is not None
                 else ZoomSample(scale_at("sweep", now - t0, args.max_scale, args.period), True, source="curve"))
            lagq.append(z)
            matte = matte_base.copy()
            encode_zoom(matte, lagq[0])
            bg.submit(draw_bg(z.scale))
            out.submit(matte)
            n += 1
            if n % int(args.fps) == 0:
                print(f"\r[TEST] t={now - t0:6.1f}s  x{z.scale:6.2f} valid={z.valid}  sent={n}", end="", flush=True)
            next_t += 1.0 / args.fps
            time.sleep(max(0.0, next_t - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        out.stop()
        bg.stop()
        if src is not None:
            src.stop()


if __name__ == "__main__":
    main()
