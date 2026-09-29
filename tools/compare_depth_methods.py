"""Compare person-distance estimation methods on a video file.

A: body size from YOLO pose keypoints (shoulder width, torso length)
B: Metric Video Depth Anything (streaming mode), median depth inside the RVM matte

Outputs (in --out): per-frame CSV, comparison plot, and sample visualization frames.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VDA_ROOT = ROOT / "third_party" / "Video-Depth-Anything"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VDA_ROOT))

from module.models.rvm import load_rvm  # noqa: E402
from video_depth_anything.video_depth_stream import VideoDepthAnything  # noqa: E402

# Typical adult dimensions (m). Absolute error of method A scales with these.
SHOULDER_M = 0.39
TORSO_M = 0.52
KP_CONF = 0.5
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def pose_distance(kps: np.ndarray, conf: np.ndarray, focal_px: float) -> tuple[float, float, float]:
    """Return (distance, from_shoulders, from_torso) in meters; NaN when unavailable."""
    z_sh = z_to = math.nan
    if conf[L_SH] > KP_CONF and conf[R_SH] > KP_CONF:
        sh_px = float(np.linalg.norm(kps[L_SH] - kps[R_SH]))
        if sh_px > 2:
            z_sh = focal_px * SHOULDER_M / sh_px
        if conf[L_HIP] > KP_CONF and conf[R_HIP] > KP_CONF:
            mid_sh = (kps[L_SH] + kps[R_SH]) / 2
            mid_hip = (kps[L_HIP] + kps[R_HIP]) / 2
            torso_px = float(np.linalg.norm(mid_sh - mid_hip))
            if torso_px > 2:
                z_to = focal_px * TORSO_M / torso_px
    vals = [v for v in (z_sh, z_to) if not math.isnan(v)]
    return (float(np.median(vals)) if vals else math.nan), z_sh, z_to


def jitter(z: np.ndarray) -> float:
    """Median |second difference| of log-distance, in percent. Lower is smoother."""
    lz = np.log(z[np.isfinite(z) & (z > 0)])
    if lz.size < 5:
        return math.nan
    return float(np.median(np.abs(np.diff(lz, n=2))) * 100.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hfov", type=float, default=60.0, help="assumed horizontal FOV of the source camera (deg)")
    p.add_argument("--width", type=int, default=1280, help="processing width")
    p.add_argument("--vda-input", type=int, default=518)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    rvm = load_rvm(str(ROOT / "rvm_mobilenetv3.pth"), "mobilenetv3", dev, True)
    from ultralytics import YOLO
    pose = YOLO(str(out / "yolov8n-pose.pt") if (out / "yolov8n-pose.pt").exists() else "yolov8n-pose.pt")
    vda = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    vda.load_state_dict(torch.load(VDA_ROOT / "checkpoints" / "metric_video_depth_anything_vits.pth", map_location="cpu"), strict=True)
    vda = vda.to(dev).eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = args.width
    H = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * W / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    focal_px = (W / 2) / math.tan(math.radians(args.hfov / 2))

    rec = [None] * 4
    rows, t_pose, t_vda = [], [], []
    samples = {int(n_total * f) for f in (0.1, 0.5, 0.8, 0.97)}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # RVM matte (shared by both methods for visualization / B sampling)
        t = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float().div(255)[None]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            _, pha, *rec = rvm(t, *rec, downsample_ratio=min(1.0, 512 / max(W, H)))
        alpha = pha[0, 0].float().cpu().numpy()

        # A: pose
        torch.cuda.synchronize(); t0 = time.perf_counter()
        res = pose.predict(frame, imgsz=640, conf=0.25, max_det=1, half=True, verbose=False)[0]
        torch.cuda.synchronize(); t_pose.append(time.perf_counter() - t0)
        z_a = z_sh = z_to = math.nan
        kps = None
        if res.keypoints is not None and len(res.keypoints) > 0:
            kps = res.keypoints.xy[0].cpu().numpy()
            kc = res.keypoints.conf[0].cpu().numpy() if res.keypoints.conf is not None else np.ones(17)
            z_a, z_sh, z_to = pose_distance(kps, kc, focal_px)

        # B: metric video depth, streaming (one frame at a time, like the live pipeline)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        depth = vda.infer_video_depth_one(rgb, input_size=args.vda_input, device=dev)
        torch.cuda.synchronize(); t_vda.append(time.perf_counter() - t0)
        core = cv2.erode((alpha > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        z_b = float(np.median(depth[core])) if core.sum() > 50 else math.nan
        person_px = float((alpha > 0.5).mean())

        rows.append(dict(frame=idx, t=idx / fps, pose_z=z_a, pose_shoulder_z=z_sh, pose_torso_z=z_to,
                         vda_z=z_b, person_area=person_px))

        if idx in samples:
            dvis = cv2.applyColorMap(
                (255 - np.clip(depth / max(1e-3, np.percentile(depth, 98)), 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)
            vis = frame.copy()
            if kps is not None:
                for x, y in kps[[L_SH, R_SH, L_HIP, R_HIP]]:
                    cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 255), -1)
            for img, txt in ((vis, f"A pose: {z_a:.2f} m"), (dvis, f"B VDA: {z_b:.2f} m")):
                cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"sample_{idx:04d}.jpg"),
                        cv2.resize(np.hstack([vis, dvis]), (1280, 360)))
        idx += 1
        if idx % 100 == 0:
            print(f"frame {idx}/{n_total}", flush=True)
    cap.release()

    with open(out / "distances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    t_arr = np.array([r["t"] for r in rows])
    za = np.array([r["pose_z"] for r in rows])
    zb = np.array([r["vda_z"] for r in rows])
    both = np.isfinite(za) & np.isfinite(zb) & (za > 0) & (zb > 0)
    scale = float(np.exp(np.median(np.log(zb[both] / za[both])))) if both.any() else math.nan

    summary = {
        "frames": len(rows),
        "A_valid_pct": float(np.isfinite(za).mean() * 100),
        "B_valid_pct": float(np.isfinite(zb).mean() * 100),
        "A_jitter_pct": jitter(za),
        "B_jitter_pct": jitter(zb),
        "A_ms_median": float(np.median(t_pose[5:]) * 1000),
        "B_ms_median": float(np.median(t_vda[5:]) * 1000),
        "A_range_m": (float(np.nanmin(za)), float(np.nanmax(za))),
        "B_range_m": (float(np.nanmin(zb)), float(np.nanmax(zb))),
        "A_to_B_scale": scale,
        "hfov_assumed": args.hfov,
    }
    for k, v in summary.items():
        print(f"SUMMARY {k} = {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t_arr, zb, color="#2a6fdb", lw=2, label="B  Metric Video Depth Anything (m)")
    ax[0].plot(t_arr, za, color="#e0822a", lw=1.5, label=f"A  pose body size (m, assumed HFOV {args.hfov:.0f}°)")
    if math.isfinite(scale):
        ax[0].plot(t_arr, za * scale, color="#e0822a", lw=1, ls="--", alpha=0.7,
                   label=f"A  scaled to B (x{scale:.2f})")
    ax[0].set_ylabel("distance to person (m)")
    ax[0].legend(loc="upper right", frameon=False)
    ax[0].grid(alpha=0.25)
    ax[1].plot(t_arr, [r["person_area"] * 100 for r in rows], color="#777")
    ax[1].set_ylabel("person area (%)")
    ax[1].set_xlabel("time (s)")
    ax[1].grid(alpha=0.25)
    fig.suptitle("Person distance: pose body-size (A) vs metric video depth (B)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=110)


if __name__ == "__main__":
    main()
