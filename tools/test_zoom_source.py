"""Offline checks for module/engine/zoom (no camera, no Unreal).

    python tools/test_zoom_source.py
"""
from __future__ import annotations

import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from module.engine.person_distance import PersonDistanceEstimator  # noqa: E402
from module.engine.zoom import (  # noqa: E402
    DEFAULT_LENS, FreeDPacket, LensLUT, ZoomSample, build_d1, decode_zoom, encode_zoom,
    make_zoom_source, parse_d1,
)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        check.failed += 1
check.failed = 0


def test_packet():
    p = FreeDPacket(camera_id=7, pan=-12.5, tilt=3.25, roll=0.5, x=1234.5, y=-20.0, z=1500.0,
                    zoom=0x00ABCD, focus=0x123456, user=0xBEEF)
    data = build_d1(p)
    q = parse_d1(data)
    check("D1 packet length", len(data) == 29)
    check("D1 round trip", q == p, str(q))
    bad = bytearray(data); bad[10] ^= 1
    check("D1 bad checksum rejected", parse_d1(bytes(bad)) is None)


def test_lut():
    lut = LensLUT.load(DEFAULT_LENS)
    check("LUT wide / tele", abs(lut.focal(0) - 8.4) < 1e-6 and abs(lut.focal(65535) - 900) < 1e-3)
    errs = [abs(lut.focal(lut.raw_for_focal(f)) / f - 1) for f in np.geomspace(8.4, 900, 50)]
    check("LUT inverse", max(errs) < 1e-9)
    check("LUT clamps outside range", abs(lut.focal(-100) - 8.4) < 1e-6 and abs(lut.focal(1e6) - 900) < 1e-3)
    check("UA107 wide hfov ~59.4 deg", abs(lut.hfov_deg(8.4) - 59.4) < 0.2, f"{lut.hfov_deg(8.4):.2f}")


def test_code_block():
    worst = 0.0
    frame = np.zeros((8, 64, 4), np.uint8)
    for s in np.geomspace(1, 214, 200):
        encode_zoom(frame, ZoomSample(scale=float(s), valid=True))
        d, ok = decode_zoom(frame)
        worst = max(worst, abs(d / s - 1))
    check("block 5 round trip < 0.01%", worst < 1e-4, f"worst {worst * 100:.4f}%")
    check("block 1 signature written", tuple(frame[0, 4, :3]) == (0, 255, 0))
    check("code pixels stay invisible", frame[:4, :24, 3].max() == 0)
    encode_zoom(frame, ZoomSample(scale=3.0, valid=False))
    check("invalid flag", decode_zoom(frame)[1] is False)


def test_freed_udp_delay():
    """Real loopback UDP: send a known curve, sample with a delay, compare."""
    port = 40991
    lut = LensLUT.load(DEFAULT_LENS)
    src = make_zoom_source("freed", port=port, delay_ms=100.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    curve = lambda t: math.exp(math.log(20) * (0.5 - 0.5 * math.cos(2 * math.pi * t / 2.0)))
    sent = []  # (time, scale)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        t = time.perf_counter()
        s = curve(t - t0)
        tx.sendto(build_d1(FreeDPacket(zoom=int(round(lut.raw_for_focal(8.4 * s))))), ("127.0.0.1", port))
        sent.append((t, s))
        time.sleep(1 / 60)
    time.sleep(0.05)
    errs = []
    for t, s in sent[30:-10]:
        z = src.sample(t + 0.100)          # frame arrived 100 ms after the lens value
        errs.append(abs(z.scale / s - 1))
    check("FreeD packets received", src.packets >= len(sent) - 2, f"{src.packets}/{len(sent)}")
    check("delayed sample matches sent curve (<1.5%)", max(errs) < 0.015,
          f"max {max(errs) * 100:.3f}% (includes UDP receive time)")
    z = src.sample(time.perf_counter() + 2.0)
    check("stale data -> invalid, scale held", (not z.valid) and z.scale > 1.0)
    src.command("zoom_extender")
    z = src.sample(sent[-20][0] + 0.100)
    check("extender doubles scale", abs(z.scale / sent[-20][1] - 2) < 0.05, f"{z.scale:.2f}")
    src.stop()


def test_distance_zoom():
    est = PersonDistanceEstimator(1920, 1080, hfov_deg=59.4)
    k = np.zeros((17, 2)); c = np.ones(17)
    k[5], k[6] = (900, 400), (1020, 400)     # shoulders 120 px apart
    k[11], k[12] = (920, 560), (1000, 560)
    z1 = est._body_distance(k, c)
    k3 = (k - (960, 540)) * 3 + (960, 540)   # same person, 3x zoom
    est.set_zoom(3.0)
    z3 = est._body_distance(k3, c)
    check("distance unchanged under 3x zoom", abs(z3 / z1 - 1) < 1e-6, f"{z1:.2f} m vs {z3:.2f} m")


def test_sdi_source():
    """SDI lens metadata: the value of frame k comes from frame k's packet, through the lens table."""
    class FakeReader:   # stands in for DeckLinkSDKReader.packets_for
        def __init__(self, raws):
            self.raws = raws

        def packets_for(self, k):
            if k is None or not 0 <= k < len(self.raws):
                return None
            data = bytearray(91)
            data[65:67] = self.raws[k].to_bytes(2, "big")
            return [(0x53, 0x03, 16, b"info"), (0x51, 0x01, 9, bytes(data))]

    lens = ROOT / "lens" / "UA24x7.8B.json"
    src = make_zoom_source("sdi", lens=str(lens), reader=FakeReader([0, 65535, 32768]))
    z0, z1 = src.sample(0, None, 0), src.sample(0, None, 1)
    check("sdi wide = 7.8 mm, x1", z0.valid and abs(z0.focal_mm - 7.8) < 1e-3 and abs(z0.scale - 1) < 1e-6,
          f"{z0.focal_mm:.2f} mm x{z0.scale:.3f}")
    check("sdi tele = 187 mm, x24", abs(z1.focal_mm - 187.0) < 1e-3, f"{z1.focal_mm:.1f} mm x{z1.scale:.2f}")
    z2 = src.sample(0, None, 2)
    check("sdi mid-range", 3.0 < z2.scale < 6.0, f"raw 32768 -> x{z2.scale:.2f}")
    zm = src.sample(0, None, 7)
    check("sdi missing frame holds, invalid", not zm.valid and zm.scale == z2.scale)


if __name__ == "__main__":
    test_packet()
    test_lut()
    test_code_block()
    test_freed_udp_delay()
    test_distance_zoom()
    test_sdi_source()
    print("ALL PASS" if check.failed == 0 else f"{check.failed} FAILED")
    sys.exit(1 if check.failed else 0)
