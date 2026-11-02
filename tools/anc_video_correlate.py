"""Match the lens values in the SDI metadata with what the picture actually did.

Input: a log + video pair from
    python tools/sdi_anc_probe.py --watch 0x51:0x01 --record-video
Then:
    python tools/anc_video_correlate.py anc_logs/anc_20261027_120000.jsonl

For every moment the metadata moved (an "event"), the zoom of the picture is measured with
background feature tracking (same method as estimate_zoom_offline.py) and compared with each
16-bit word of the packet. The word that follows the picture's zoom is the zoom encoder;
a word that moves while the picture scale stays put is something else (focus, iris ...).

Writes next to the log:
  <log>.events.csv   one row per event: time, picture zoom change, change of every word
  <log>.frames.csv   every frame inside events: words + measured picture scale (lens table data)
  <log>.events.png   per event: picture log-scale (black) and each word (colors), normalized
  <log>.ev<N>.jpg    first / last frame of each event side by side
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from estimate_zoom_offline import pair_transform, person_mask  # noqa: E402

WORDS = {"w1": 63, "w2": 65, "w3": 67, "w4": 69}   # byte offset of each big-endian 16-bit word
COLORS = {"w1": (180, 180, 180), "w2": (0, 0, 220), "w3": (0, 160, 0), "w4": (220, 120, 0), "v": (160, 0, 160)}


def load(log: Path, did: int, line: int):
    rows, header = [], {}
    with open(log, encoding="utf-8") as f:
        for s in f:
            r = json.loads(s)
            if "p" not in r:
                header = r
                continue
            if "vf" not in r:
                continue
            pk = next((p for p in r["p"] if p[2] == did and p[1] == line), None)
            if pk is None:
                continue
            d = bytes.fromhex(pk[5])
            w = {k: int.from_bytes(d[o:o + 2], "big") for k, o in WORDS.items()}
            w["v"] = int.from_bytes(d[71:73], "big", signed=True)
            rows.append({"t": r["t"], "vf": r["vf"], "ns": r["ns"], **w})
    return header, rows


def find_events(rows, noise: int, gap: float, pad: float):
    """Spans where w2..w4 move more than noise, merged when closer than gap seconds."""
    spans, base = [], rows[0]
    for r in rows:
        if any(abs(r[k] - base[k]) > noise for k in ("w2", "w3", "w4")):
            if spans and r["t"] - spans[-1][1] < gap:
                spans[-1][1] = r["t"]
            else:
                spans.append([r["t"], r["t"]])
            base = r
    return [(a - pad, b + pad) for a, b in spans]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--video", default=None, help="기본: 로그와 같은 이름의 .mp4")
    ap.add_argument("--did", default="0x51")
    ap.add_argument("--line", type=int, default=9)
    ap.add_argument("--noise", type=int, default=16)
    ap.add_argument("--gap", type=float, default=1.5, help="이보다 가까운 움직임은 한 이벤트로 합침 (초)")
    ap.add_argument("--pad", type=float, default=1.0, help="이벤트 앞뒤 여유 (초)")
    ap.add_argument("--no-mask", action="store_true", help="사람 가리기 끔 (빠름)")
    args = ap.parse_args()

    log = Path(args.log)
    header, rows = load(log, int(args.did, 0), args.line)
    video = Path(args.video or header.get("video") or log.with_suffix(".mp4"))
    if not rows:
        sys.exit("로그에 영상 프레임 번호(vf)가 있는 패킷이 없습니다 (--record-video 로 기록했는지 확인).")
    events = find_events(rows, args.noise, args.gap, args.pad)
    print(f"[INFO] {len(rows)} frames, {len(events)} events, video {video}")

    seg = None
    if not args.no_mask:
        try:
            from ultralytics import YOLO
            seg = YOLO(str(ROOT / "yolov8n-seg.pt"))
        except Exception as e:
            print(f"[WARN] person mask off: {e}")

    by_vf = {r["vf"]: r for r in rows}
    windows = []   # (event index, first vf, last vf)
    for i, (a, b) in enumerate(events):
        vfs = [r["vf"] for r in rows if a <= r["t"] <= b]
        if vfs:
            windows.append((i, vfs[0], vfs[-1]))

    # one pass through the video, measuring frame-to-frame zoom inside the event windows
    cap = cv2.VideoCapture(str(video))
    results = {i: [] for i, _, _ in windows}
    firstlast = {}
    wi, vf, prev = 0, -1, None
    while wi < len(windows):
        ok, f = cap.read()
        if not ok:
            break
        vf += 1
        i, v0, v1 = windows[wi]
        if vf < v0:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if vf == v0 or prev is None:
            logs = 0.0
            firstlast[i] = [f, f]
        else:
            M, inl = pair_transform(prev[0], g, prev[1])
            ds = math.log(math.sqrt(abs(np.linalg.det(M[:, :2])))) if M is not None else 0.0
            logs = results[i][-1][1] + ds
            firstlast[i][1] = f
        results[i].append((vf, logs))
        prev = (g, person_mask(seg, f, (g.shape[1], g.shape[0])) if seg else None)
        if vf >= v1:
            wi += 1
            prev = None
            print(f"[EV {i}] frames {v0}-{v1}  picture zoom x{math.exp(results[i][-1][1]):.3f}")

    # tables
    ev_csv, fr_csv = log.with_suffix(".events.csv"), log.with_suffix(".frames.csv")
    with open(ev_csv, "w", newline="", encoding="utf-8") as fe, open(fr_csv, "w", newline="", encoding="utf-8") as ff:
        we, wf = csv.writer(fe), csv.writer(ff)
        we.writerow(["event", "start_s", "end_s", "clock", "picture_zoom", *[f"d_{k}" for k in WORDS],
                     *[f"corr_{k}" for k in WORDS]])
        wf.writerow(["event", "vf", "t", "log_scale", *WORDS, "v"])
        start_clock = header.get("start")
        for i, res in results.items():
            if len(res) < 3:
                continue
            ls = np.array([x[1] for x in res])
            ws = {k: np.array([by_vf[v][k] for v, _ in res], float) for k in WORDS}
            corr = {k: (np.corrcoef(ls, ws[k])[0, 1] if ws[k].std() > 0 and ls.std() > 1e-4 else float("nan"))
                    for k in WORDS}
            t0 = by_vf[res[0][0]]["t"]
            clock = ""
            if start_clock:
                import datetime as dt
                clock = (dt.datetime.strptime(start_clock, "%Y-%m-%d %H:%M:%S") + dt.timedelta(seconds=t0)).strftime("%H:%M:%S")
            we.writerow([i, f"{t0:.2f}", f"{by_vf[res[-1][0]]['t']:.2f}", clock, f"{math.exp(ls[-1]):.4f}",
                         *[int(ws[k][-1] - ws[k][0]) for k in WORDS], *[f"{corr[k]:+.3f}" for k in WORDS]])
            for (v, l) in res:
                r = by_vf[v]
                wf.writerow([i, v, r["t"], f"{l:.5f}", *[r[k] for k in WORDS], r["v"]])
            print(f"[EV {i}] {clock} picture x{math.exp(ls[-1]):.3f}  " +
                  "  ".join(f"{k} {int(ws[k][-1] - ws[k][0]):+6d} (r {corr[k]:+.2f})" for k in WORDS))
            a, b = firstlast[i]
            cv2.imwrite(str(log.with_suffix(f".ev{i}.jpg")), np.hstack([a, b]))

    # plot: one strip per event
    strips = []
    for i, res in results.items():
        if len(res) < 3:
            continue
        H, W = 160, 900
        img = np.full((H, W, 3), 255, np.uint8)
        n = len(res)
        series = {"pic": np.array([x[1] for x in res])}
        series.update({k: np.array([by_vf[v][k] for v, _ in res], float) for k in (*WORDS, "v")})
        for k, s in series.items():
            if s.max() - s.min() < 1e-6:
                continue
            y = (s - s.min()) / (s.max() - s.min())
            pts = np.array([[int(j * (W - 1) / max(n - 1, 1)), int((1 - y[j]) * (H - 20) + 10)] for j in range(n)])
            cv2.polylines(img, [pts], False, (0, 0, 0) if k == "pic" else COLORS[k], 3 if k == "pic" else 1)
        cv2.putText(img, f"event {i}  black=picture zoom  red=w2 green=w3 orange=w4 purple=v", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        strips.append(img)
    if strips:
        cv2.imwrite(str(log.with_suffix(".events.png")), np.vstack(strips))
    print(f"[INFO] {ev_csv}\n[INFO] {fr_csv}")


if __name__ == "__main__":
    main()
