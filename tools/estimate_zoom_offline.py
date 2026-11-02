"""Estimate a clip's per-frame zoom (offline CV) for --zoom-source replay.

Background features (people masked out with YOLO-seg) are tracked frame to frame and a
similarity transform (scale + rotation + shift) is fitted with RANSAC. The scales are
chained and normalized so the widest frame of the clip is x1.

    python tools/estimate_zoom_offline.py --video test_videos/hatsuha5994_0-8s.mp4

Writes <video>.zoom.csv (frame,scale,dx,dy,inliers) and, for checking by eye,
<video>.zoom.png (curve) and <video>.zoom_check.jpg: frames warped back into the
widest frame's view. If the estimate is right, the background in the check sheet
stays put and only the frame border shrinks or grows.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def person_mask(seg, frame, size):
    """1 = usable background, 0 = person (dilated)."""
    m = np.ones(size[::-1], np.uint8)
    if seg is None:
        return m
    r = seg.predict(frame, imgsz=640, classes=[0], conf=0.25, verbose=False)[0]
    if r.masks is not None:
        for pm in r.masks.data.cpu().numpy():
            pm = cv2.resize(pm, size, interpolation=cv2.INTER_NEAREST)
            m[pm > 0.5] = 0
    return cv2.erode(m, np.ones((25, 25), np.uint8))


def pair_transform(g0, g1, mask):
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=1200, qualityLevel=0.005, minDistance=7, mask=mask)
    if p0 is None or len(p0) < 20:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=4)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < 0.7)
    if ok.sum() < 20:
        return None, int(ok.sum())
    M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                         maxIters=4000, confidence=0.999)
    return M, int(inl.sum()) if inl is not None else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--work-width", type=int, default=960)
    p.add_argument("--no-mask", action="store_true", help="do not mask people (faster, less robust)")
    p.add_argument("--seg-model", default=str(ROOT / "yolov8n-seg.pt"))
    args = p.parse_args()

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = args.work_width / W
    size = (args.work_width, int(round(H * k)))
    seg = None
    if not args.no_mask:
        from ultralytics import YOLO
        seg = YOLO(args.seg_model)

    frames_small, cum = [], [np.eye(3)]
    rows = []
    ok, f0 = cap.read()
    g_prev = cv2.cvtColor(cv2.resize(f0, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m_prev = person_mask(seg, f0, size)
    frames_small.append(cv2.resize(f0, size, interpolation=cv2.INTER_AREA))
    rows.append((0, 0))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        small = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        M, inl = pair_transform(g_prev, g, m_prev)
        A = np.eye(3)
        if M is not None:
            A[:2] = M
        else:
            print(f"[WARN] frame {i}: no transform (inliers {inl}), holding")
        cum.append(A @ cum[-1])      # maps frame-0 pixels -> frame-i pixels
        rows.append((i, inl))
        frames_small.append(small)
        g_prev, m_prev = g, person_mask(seg, f, size)
        if i % 60 == 0:
            s = np.sqrt(np.linalg.det(cum[-1][:2, :2]))
            print(f"[EST] frame {i}/{n}  scale(vs frame 0) x{s:.3f}  inliers {inl}")

    scale0 = np.array([np.sqrt(abs(np.linalg.det(C[:2, :2]))) for C in cum])
    ref = int(np.argmin(scale0))                     # widest frame -> x1
    scale = scale0 / scale0[ref]
    c = np.array([size[0] / 2, size[1] / 2, 1.0])
    out_csv = video.with_suffix(".zoom.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["frame", "scale", "dx", "dy", "inliers"])
        for (idx, inl), C, s in zip(rows, cum, scale):
            # where the widest frame's center lands in this frame, as a fraction of width/height
            Cr = C @ np.linalg.inv(cum[ref])
            q = Cr @ c
            w.writerow([idx, f"{s:.5f}", f"{(q[0] - c[0]) / size[0]:.5f}", f"{(q[1] - c[1]) / size[1]:.5f}", inl])
    print(f"[EST] {len(rows)} frames, widest = frame {ref}, max zoom x{scale.max():.2f} -> {out_csv}")

    # curve plot
    plot = np.full((300, 900, 3), 255, np.uint8)
    smax = max(scale.max(), 1.01)
    pts = [(int(20 + 860 * j / max(1, len(scale) - 1)), int(280 - 260 * (s - 1) / (smax - 1))) for j, s in enumerate(scale)]
    cv2.polylines(plot, [np.array(pts, np.int32)], False, (200, 80, 0), 2)
    cv2.putText(plot, f"zoom x1..x{smax:.2f}  ({len(scale)} frames)", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(video.with_suffix(".zoom.png")), plot)

    # check sheet: warp sampled frames back into the widest frame's view
    tiles = []
    for j in np.linspace(0, len(frames_small) - 1, 12).astype(int):
        Cr = cum[j] @ np.linalg.inv(cum[ref])
        back = cv2.warpAffine(frames_small[j], np.linalg.inv(Cr)[:2], size)
        cv2.putText(back, f"#{j} x{scale[j]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(back, (480, 270)))
    sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)])
    cv2.imwrite(str(video.with_suffix(".zoom_check.jpg")), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    sys.exit(main())
