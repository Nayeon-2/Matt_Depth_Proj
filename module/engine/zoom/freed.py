"""FreeD (type D1) camera tracking packets over UDP.

D1 packet, 29 bytes, big-endian 24-bit fields:
    0      0xD1
    1      camera id
    2-4    pan    (signed, 1/32768 deg)
    5-7    tilt   (signed, 1/32768 deg)
    8-10   roll   (signed, 1/32768 deg)
    11-13  X      (signed, 1/64 mm)
    14-16  Y      (signed, 1/64 mm)
    17-19  Z      (signed, 1/64 mm)
    20-22  zoom   (unsigned raw encoder)
    23-25  focus  (unsigned raw encoder)
    26-27  user / spare
    28     checksum = (0x40 - sum(bytes 0..27)) mod 256
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .base import EncoderZoomSource
from .lens_lut import LensLUT

D1 = 0xD1
D1_LEN = 29


@dataclass
class FreeDPacket:
    camera_id: int = 1
    pan: float = 0.0      # deg
    tilt: float = 0.0
    roll: float = 0.0
    x: float = 0.0        # mm
    y: float = 0.0
    z: float = 0.0
    zoom: int = 0         # raw
    focus: int = 0        # raw
    user: int = 0


def _checksum(body: bytes) -> int:
    return (0x40 - sum(body)) & 0xFF


def _s24(b: bytes) -> int:
    v = int.from_bytes(b, "big")
    return v - (1 << 24) if v & 0x800000 else v


def _p24(v: int) -> bytes:
    return (int(v) & 0xFFFFFF).to_bytes(3, "big")


def parse_d1(data: bytes) -> Optional[FreeDPacket]:
    """Packet, or None when the data is not a valid D1 message."""
    if len(data) < D1_LEN or data[0] != D1 or _checksum(data[:28]) != data[28]:
        return None
    return FreeDPacket(
        camera_id=data[1],
        pan=_s24(data[2:5]) / 32768.0,
        tilt=_s24(data[5:8]) / 32768.0,
        roll=_s24(data[8:11]) / 32768.0,
        x=_s24(data[11:14]) / 64.0,
        y=_s24(data[14:17]) / 64.0,
        z=_s24(data[17:20]) / 64.0,
        zoom=int.from_bytes(data[20:23], "big"),
        focus=int.from_bytes(data[23:26], "big"),
        user=int.from_bytes(data[26:28], "big"),
    )


def build_d1(p: FreeDPacket) -> bytes:
    body = (bytes([D1, p.camera_id & 0xFF])
            + _p24(round(p.pan * 32768)) + _p24(round(p.tilt * 32768)) + _p24(round(p.roll * 32768))
            + _p24(round(p.x * 64)) + _p24(round(p.y * 64)) + _p24(round(p.z * 64))
            + _p24(p.zoom) + _p24(p.focus) + (int(p.user) & 0xFFFF).to_bytes(2, "big"))
    return body + bytes([_checksum(body)])


class FreeDZoomSource(EncoderZoomSource):
    """Listens for FreeD D1 on UDP and keeps the zoom encoder history."""

    name = "freed"

    def __init__(self, lut: LensLUT, port: int = 40000, host: str = "0.0.0.0",
                 camera_id: int = -1, **kw):
        super().__init__(lut, **kw)
        self.port, self.host, self.camera_id = int(port), host, int(camera_id)
        self.bad_packets = 0
        self.last_packet: Optional[FreeDPacket] = None
        self._sock: Optional[socket.socket] = None
        self._alive = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FreeDZoomSource":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.2)
        self._alive = True
        self._thread = threading.Thread(target=self._loop, name="FreeDZoom", daemon=True)
        self._thread.start()
        print(f"[ZOOM] FreeD listening on udp {self.host}:{self.port} "
              f"(camera {'any' if self.camera_id < 0 else self.camera_id}) | {self.status()}")
        return self

    def stop(self):
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()

    def _loop(self):
        while self._alive:
            try:
                data, _ = self._sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            t = time.perf_counter()
            # Some senders pack several messages into one datagram.
            for off in range(0, len(data) - D1_LEN + 1, D1_LEN):
                pkt = parse_d1(data[off:off + D1_LEN])
                if pkt is None:
                    self.bad_packets += 1
                    continue
                if self.camera_id >= 0 and pkt.camera_id != self.camera_id:
                    continue
                self.last_packet = pkt
                self.push(pkt.zoom, t)
