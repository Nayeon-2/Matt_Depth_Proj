"""Lens table from a calibration recording: SDI zoom value -> focal length (lens/<name>.json).

Field steps (camera fixed, textured background that does not move, no one walking through):
    1. python tools/sdi_anc_probe.py --watch 0x51:0x01 --record-video --seconds 120
       (zoom full wide -> full tele -> full wide, twice, slowly)
    2. python tools/anc_video_correlate.py anc_logs/anc_<time>.jsonl
    3. python tools/make_lens_table.py anc_logs/anc_<time>.jsonl --name UA24x7.8B --wide-mm 7.8 --tele-mm 187

Uses every zoom sweep in <log>.frames.csv that covers most of the encoder range, sets the wide
end of each sweep to x1, averages the measured picture scale per encoder bin, then (with
--tele-mm) bends the curve slightly in log so full tele = the lens spec: the picture
measurement drifts a few % over a 20x+ zoom, the spec end points do not.
Prints the --zoom-lens / --dist-hfov options and the Unreal Wide Focal Length to use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="anc_logs/anc_<time>.jsonl (its .frames.csv from anc_video_correlate.py is read)")
    ap.add_argument("--name", required=True, help="렌즈 이름 (파일 이름이 됨), 예: UA24x7.8B")
    ap.add_argument("--wide-mm", type=float, required=True, help="렌즈 사양의 와이드 끝 초점거리 (mm)")
    ap.add_argument("--tele-mm", type=float, default=None, help="렌즈 사양의 텔레 끝 초점거리 (mm). 주면 끝점을 사양에 맞춤")
    ap.add_argument("--sensor-width", type=float, default=9.59, help="센서 가로 (mm). 2/3\" 16:9 = 9.59")
    ap.add_argument("--word", default="w2", help="줌 값 열 (sdi_anc_probe 기준 w2 = byte 65-66)")
    ap.add_argument("--byte", type=int, default=65, help="렌즈 JSON에 기록할 SDI 바이트 위치")
    ap.add_argument("--min-span", type=float, default=0.6, help="이 비율 이상 줌 범위를 쓴 구간만 사용")
    ap.add_argument("--bin", type=int, default=2048)
    ap.add_argument("--out", default=None, help="기본 lens/<name>.json")
    args = ap.parse_args()

    frames = Path(args.log).with_suffix(".frames.csv")
    rows = list(csv.DictReader(open(frames, encoding="utf-8")))
    events = defaultdict(list)
    for r in rows:
        events[r["event"]].append((int(r[args.word]), float(r["log_scale"])))

    bins, used = defaultdict(list), []
    for ev, pts in events.items():
        raws = [p[0] for p in pts]
        if max(raws) - min(raws) < args.min_span * 65535:
            continue   # focus / small zoom moves: not a sweep
        lo = min(raws)
        wide = [ls for raw, ls in pts if raw <= lo + 300]
        ref = sum(wide) / len(wide)
        for raw, ls in pts:
            bins[raw // args.bin].append((raw, ls - ref))
        used.append(ev)
    if not used:
        raise SystemExit("와이드-텔레를 크게 오간 구간이 없습니다. 줌을 끝에서 끝까지 움직이며 다시 기록하세요.")

    curve = []
    for k in sorted(bins):
        v = bins[k]
        if len(v) >= 4:
            curve.append((sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)))
    measured_tele = math.exp(curve[-1][1])
    k = math.log(args.tele_mm / args.wide_mm) / curve[-1][1] if args.tele_mm else 1.0
    points = [[0, args.wide_mm]]
    points += [[round(raw), round(args.wide_mm * math.exp(k * ls), 3)] for raw, ls in curve[1:]]
    tele = args.tele_mm or points[-1][1]
    if points[-1][0] < 65535:
        points.append([65535, round(tele, 3)])

    hfov = math.degrees(2 * math.atan(args.sensor_width / (2 * args.wide_mm)))
    out = Path(args.out) if args.out else ROOT / "lens" / f"{args.name}.json"
    out.write_text(json.dumps({
        "name": args.name,
        "note": (f"Measured from {Path(args.log).name} (sweeps {', '.join(used)}): picture scale vs SDI zoom value. "
                 f"Picture gave x{measured_tele:.2f} at full tele"
                 + (f", curve log-scaled by {k:.4f} to the spec {args.wide_mm:g}-{args.tele_mm:g} mm." if args.tele_mm else ".")
                 + " Extender not included (X key / --zoom-extender)."),
        "sensor_width_mm": args.sensor_width,
        "calibrated": True,
        "sdi": {"did": "0x51", "sdid": "0x01", "byte": args.byte, "bytes": 2, "endian": "big"},
        "points": points,
    }, indent=1), encoding="utf-8")

    print(f"[LENS] sweeps used: {', '.join(used)} ({sum(len(events[e]) for e in used)} frames)")
    print(f"[LENS] picture zoom at full tele: x{measured_tele:.2f}"
          + (f"  (spec x{args.tele_mm / args.wide_mm:.2f}, log-scale {k:.4f})" if args.tele_mm else ""))
    if args.tele_mm and abs(k - 1) > 0.05:
        print("[WARN] 측정과 사양이 5% 넘게 다릅니다. 렌즈 이름·익스텐더·초점거리를 확인하세요.")
    print(f"[LENS] {len(points)} points -> {out}")
    print()
    print("적용:")
    print(f"  RVM     : --zoom-source sdi --zoom-lens {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} --dist-hfov {hfov:.1f}")
    print(f"  Unreal  : ZoomCamera > Zoom > Wide Focal Length = {args.wide_mm:g}")


if __name__ == "__main__":
    main()
