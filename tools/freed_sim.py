"""FreeD D1 simulator: sends a fake zoom lens over UDP so the pipeline can be
tested without the real lens / converter box.

    python tools/freed_sim.py                         # sweep x1 -> x20 -> x1 every 10 s
    python tools/freed_sim.py --profile steps --max-scale 8
    python tools/freed_sim.py --profile hold --max-scale 3
    python tools/freed_sim.py --profile keys          # - / = / z / q in this window

The zoom curve is made in "scale" (focal / wide focal) and converted to raw encoder
values through the lens table, like a real lens would report it.
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from module.engine.zoom import DEFAULT_LENS, FreeDPacket, LensLUT, build_d1  # noqa: E402


def scale_at(profile: str, t: float, smax: float, period: float) -> float:
    """Zoom scale at time t. Curves are smooth in log(scale), like an operator's zoom."""
    lmax = math.log(smax)
    ph = (t % period) / period
    if profile == "sweep":      # wide -> tele -> wide, eased
        k = 0.5 - 0.5 * math.cos(2 * math.pi * ph)
        return math.exp(lmax * k)
    if profile == "sine":       # gentle breathing around the middle
        return math.exp(lmax * (0.5 + 0.25 * math.sin(2 * math.pi * ph)))
    if profile == "steps":      # hold 1, 2, 4 ... smax with quick 0.5 s moves between them
        levels = [1.0]
        while levels[-1] * 2 < smax:
            levels.append(levels[-1] * 2)
        levels.append(smax)
        seg = period / len(levels)
        i = int((t % period) // seg)
        a, b = levels[i - 1], levels[i]          # move a -> b in the first 0.5 s of segment i
        k = min(1.0, ((t % period) - i * seg) / 0.5)
        k = 0.5 - 0.5 * math.cos(math.pi * k)
        return math.exp(math.log(a) + (math.log(b) - math.log(a)) * k)
    if profile == "hold":
        return smax
    raise ValueError(profile)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=40000)
    p.add_argument("--camera-id", type=int, default=1)
    p.add_argument("--rate", type=float, default=60.0, help="packets per second")
    p.add_argument("--lens", default=str(DEFAULT_LENS))
    p.add_argument("--profile", choices=["sweep", "sine", "steps", "hold", "keys"], default="sweep")
    p.add_argument("--max-scale", type=float, default=20.0, help="largest zoom (x)")
    p.add_argument("--period", type=float, default=10.0, help="seconds per cycle")
    p.add_argument("--jitter-ms", type=float, default=0.0, help="random send jitter (network test)")
    p.add_argument("--drop", type=float, default=0.0, help="fraction of packets to drop (0..1)")
    args = p.parse_args()

    lut = LensLUT.load(args.lens)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[SIM] FreeD -> udp {args.host}:{args.port} cam={args.camera_id} {args.rate:g} Hz "
          f"profile={args.profile} lens='{lut.name}'")

    keys_scale = 1.0
    msvcrt = None
    if args.profile == "keys":
        import msvcrt  # Windows console
        print("[SIM] keys: = zoom in, - zoom out, z wide, q quit")

    t0 = time.perf_counter()
    dt = 1.0 / args.rate
    n, next_t, last_print = 0, t0, 0.0
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if args.profile == "keys":
                while msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    if ch == "=":
                        keys_scale = min(keys_scale * 1.1, args.max_scale)
                    elif ch == "-":
                        keys_scale = max(keys_scale / 1.1, 1.0)
                    elif ch == "z":
                        keys_scale = 1.0
                    elif ch == "q":
                        return
                scale = keys_scale
            else:
                scale = scale_at(args.profile, t, args.max_scale, args.period)
            raw = int(round(lut.raw_for_focal(lut.wide_mm * scale)))
            if random.random() >= args.drop:
                sock.sendto(build_d1(FreeDPacket(camera_id=args.camera_id, zoom=raw, focus=0x8000)),
                            (args.host, args.port))
            n += 1
            if t - last_print >= 0.5:
                last_print = t
                f = lut.focal(raw)
                print(f"\r[SIM] t={t:6.1f}s  x{scale:6.2f}  raw={raw:6d}  f={f:6.1f}mm  "
                      f"hfov={lut.hfov_deg(f):5.2f}deg  sent={n}", end="", flush=True)
            next_t += dt
            sleep = next_t - time.perf_counter() + (random.uniform(-1, 1) * args.jitter_ms / 1000.0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        print()


if __name__ == "__main__":
    main()
