"""Run PersonDistanceEstimator (method D) on a video file and plot the result."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from module.engine.person_distance import PersonDistanceEstimator  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pose", default="yolov8n-pose.pt")
    p.add_argument("--hfov", type=float, default=60.0)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    pose = YOLO(args.pose)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = 1920, 1080
    calib = out / "ground_calib.json"
    calib.unlink(missing_ok=True)
    est = PersonDistanceEstimator(W, H, hfov_deg=args.hfov, calib_path=str(calib))

    t, z_body, z_d, src = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        r = pose.predict(frame, imgsz=args.imgsz, conf=0.25, max_det=1, half=True, verbose=False)[0]
        kps = conf = bbox = None
        if r.keypoints is not None and len(r.keypoints) > 0:
            kps = r.keypoints.xy[0].cpu().numpy()
            conf = r.keypoints.conf[0].cpu().numpy()
            bbox = r.boxes.xyxy[0].cpu().numpy()
        ts = i / fps
        body = est._body_distance(kps, conf) if kps is not None else np.nan
        z, valid = est.update(kps, conf, bbox, now=ts)
        t.append(ts); z_body.append(body); z_d.append(z if valid else np.nan); src.append(est.source)
        i += 1
    cap.release()

    t = np.array(t); z_body = np.array(z_body); z_d = np.array(z_d); src = np.array(src)
    lz = np.log(z_d[np.isfinite(z_d)])
    print(f"frames={i} foot+body={np.mean(src == 'foot+body')*100:.0f}% body={np.mean(src == 'body')*100:.0f}% "
          f"range={np.nanmin(z_d):.2f}..{np.nanmax(z_d):.2f} m "
          f"jitter={np.median(np.abs(np.diff(lz, n=2)))*100:.2f}% person_scale={est.person_scale:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, z_body, color="#bbbbbb", lw=1, label="body size only (raw)")
    for s, col in (("foot+body", "#2a9d5b"), ("foot", "#1b6fb5"), ("body", "#e0822a")):
        zz = np.where(src == s, z_d, np.nan)
        ax.plot(t, zz, color=col, lw=2.2, label=f"method D (source: {s})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance to person (m)")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    ax.set_title("Method D: body size + self-calibrated foot contact + One Euro filter")
    fig.tight_layout(); fig.savefig(out / "method_d.png", dpi=110)


if __name__ == "__main__":
    main()
