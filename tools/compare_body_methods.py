"""Compare person distance body measures ("full" vs "upper") on a video.

    python tools/compare_body_methods.py --video "..\\영상소스\\Oddeye_test_end-30_-20s.mp4" \
        --window 0 2.5 --window 4 5.5

Runs YOLO pose once per frame and feeds the same keypoints to both estimators (person
nearest the frame center). Prints, per time window, the mean and the frame-to-frame
noise of the raw body measure and of the filtered output, and writes <out>.csv / .png.
A good measure shows a clear step between windows and little noise inside them.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from module.engine.person_distance import PersonDistanceEstimator  # noqa: E402

METHODS = ("full", "upper")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--hfov", type=float, default=60.0)
    p.add_argument("--pose", default=str(ROOT / "yolov8n-pose.pt"))
    p.add_argument("--window", nargs=2, type=float, action="append", default=[],
                   metavar=("START", "END"), help="time window in seconds (repeatable)")
    p.add_argument("--out", default=None, help="output prefix (default: next to the video)")
    args = p.parse_args()

    from ultralytics import YOLO
    pose = YOLO(args.pose)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video} (OpenCV on Windows cannot open paths with Korean characters: copy it to an English path)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = 1920, 1080
    est = {m: PersonDistanceEstimator(W, H, hfov_deg=args.hfov, body_method=m) for m in METHODS}
    rows = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        f = cv2.resize(f, (W, H))
        r = pose.predict(f, imgsz=640, verbose=False)[0]
        kps = vis = box = None
        if r.keypoints is not None and len(r.keypoints) > 0:
            b = r.boxes.xyxy.cpu().numpy()
            j = int(np.argmin(np.abs((b[:, 0] + b[:, 2]) / 2 - W / 2)))
            kps, vis, box = r.keypoints.xy.cpu().numpy()[j], r.keypoints.conf.cpu().numpy()[j], b[j]
        row = {"t": t}
        for m, e in est.items():
            raw = e._body_distance(np.asarray(kps).reshape(-1, 2), np.asarray(vis).reshape(-1)) if kps is not None else math.nan
            z, ok = e.update(kps, vis, box, now=t)
            row[f"{m}_raw"], row[f"{m}_out"], row[f"{m}_src"] = raw, (z if ok else math.nan), e.source
        rows.append(row)

    out = Path(args.out) if args.out else Path(args.video).with_suffix("")
    with open(str(out) + ".body_compare.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stats(key, a, b):
        v = np.array([r[key] for r in rows if a <= r["t"] < b], float)
        v = v[~np.isnan(v)]
        if len(v) < 3:
            return math.nan, math.nan
        noise = float(np.median(np.abs(np.diff(v))) / np.median(v) * 100)   # % per frame
        return float(np.median(v)), noise

    print(f"{len(rows)} frames @ {fps:.2f} fps")
    for m in METHODS:
        print(f"== {m}")
        meds = []
        for a, b in args.window:
            mr, nr = stats(f"{m}_raw", a, b)
            mo, no = stats(f"{m}_out", a, b)
            meds.append(mo)
            print(f"  {a:5.1f}-{b:5.1f}s  raw {mr:5.2f} m (noise {nr:4.1f}%/frame)   output {mo:5.2f} m (noise {no:4.2f}%/frame)")
        if len(meds) >= 2:
            print(f"  step between first two windows: {abs(meds[0] - meds[1]) / meds[0] * 100:.1f}%")

    # plot
    plot = np.full((360, 1000, 3), 255, np.uint8)
    vals = [r[f"{m}_out"] for r in rows for m in METHODS if not math.isnan(r[f"{m}_out"])]
    lo, hi = min(vals) * 0.9, max(vals) * 1.1
    T = rows[-1]["t"] or 1
    for a, b in args.window:
        cv2.rectangle(plot, (int(40 + 940 * a / T), 20), (int(40 + 940 * b / T), 340), (235, 235, 235), -1)
    for m, col in zip(METHODS, ((200, 80, 0), (0, 0, 220))):
        for key, th in ((f"{m}_raw", 1), (f"{m}_out", 2)):
            pts = [(int(40 + 940 * r["t"] / T), int(340 - 320 * (r[key] - lo) / (hi - lo)))
                   for r in rows if not math.isnan(r[key]) and lo <= r[key] <= hi]
            if len(pts) > 1:
                c = col if th == 2 else tuple(int(x * 0.4 + 255 * 0.6) for x in col)
                cv2.polylines(plot, [np.array(pts, np.int32)], False, c, th)
    cv2.putText(plot, f"blue=full  red=upper  (thin=raw, thick=filtered)  {lo:.1f}-{hi:.1f} m",
                (40, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(out) + ".body_compare.png", plot)


if __name__ == "__main__":
    main()
