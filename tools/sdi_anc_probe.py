"""SDI ANC probe: lists every ancillary packet arriving on a DeckLink input, to find out
whether the camera carries lens zoom / focus / iris in the SDI signal.

    python tools/sdi_anc_probe.py --list                    # DeckLink devices
    python tools/sdi_anc_probe.py                           # device 0, until Ctrl+C
    python tools/sdi_anc_probe.py --device 0 --seconds 60   # zoom the lens while it runs
    python tools/sdi_anc_probe.py --hanc 0x43:0x05          # also capture this DID:SDID from HANC
    python tools/sdi_anc_probe.py --analyze anc_logs/anc_20261023_150000.jsonl
    python tools/sdi_anc_probe.py --watch 0x51:0x01         # keep running, one line when the value moves

Every second it prints one row per packet type (data space, line, DID, SDID) with packets
per second, payload length, how many different payloads were seen and which byte
positions changed. The raw VANC lines are also scanned for ANC flags (RAWY / RAWC rows),
and the RP188 timecode the driver decodes is shown. Zoom the lens wide <-> tele while it runs: a packet whose bytes change
only while zooming is the lens data. Everything is logged to anc_logs/*.jsonl.

Uses the DeckLink COM API from Blackmagic Desktop Video (no SDK needed) through comtypes.
The device must not be open in another program (rvm_main ffmpeg/dshow, Media Express,
Unreal Blackmagic Media). VANC packets come through automatically; HANC packets only for
the DID:SDID pairs given with --hanc (driver limit, up to 4).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECKLINK_TLB = ("{D864517A-EDD5-466D-867D-C819F1C052BB}", 1, 0)

# SMPTE ST 291 registered DID/SDID (subset). 0x50-0x5F = user application (manufacturer).
KNOWN = {
    (0x41, 0x01): "ST 352 payload ID",
    (0x41, 0x05): "AFD / bar data",
    (0x41, 0x06): "pan-scan",
    (0x41, 0x07): "SCTE-104",
    (0x43, 0x02): "OP-47 SDP (teletext)",
    (0x43, 0x03): "OP-47 multipacket",
    (0x43, 0x05): "RDD 18 camera/lens metadata",
    (0x44, 0x04): "KLV metadata (RP 214)",
    (0x44, 0x14): "KLV metadata HANC",
    (0x44, 0x44): "UMID (RP 223)",
    (0x60, 0x60): "ATC timecode (ST 12-2)",
    (0x61, 0x01): "CEA-708 captions",
    (0x61, 0x02): "CEA-608 captions",
    (0x64, 0x64): "LTC (RP 196)",
    (0x64, 0x7F): "VITC (RP 196)",
}
for _s in range(1, 10):
    KNOWN[(0x45, _s)] = "audio metadata (ST 2020)"


def describe(did: int, sdid: int) -> str:
    if (did, sdid) in KNOWN:
        return KNOWN[(did, sdid)]
    if 0x50 <= did <= 0x5F:
        return "USER / manufacturer  <-- check"
    if did in (0x80, 0x84, 0x88):
        return "deletion / end marker"
    return "?"


# ---------------------------------------------------------------- statistics
class PacketStats:
    """Per packet type: count, distinct payloads, which byte positions ever changed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.types: dict[tuple, dict] = {}
        self.frames = 0
        self.frames_no_signal = 0
        self.timecode = None   # RP188 timecode the driver decoded (it does not show up as a packet)

    def add_frame(self, packets: list[tuple], no_signal: bool):
        with self.lock:
            self.frames += 1
            self.frames_no_signal += int(no_signal)
            for key, data in packets:
                st = self.types.get(key)
                if st is None:
                    st = self.types[key] = {"count": 0, "window": 0, "first": data, "last": data,
                                            "lens": set(), "distinct": set(), "changed": set(),
                                            "changes": 0, "window_changes": 0}
                st["count"] += 1
                st["window"] += 1
                st["lens"].add(len(data))
                if len(st["distinct"]) < 4096:
                    st["distinct"].add(data)
                if data != st["last"]:
                    st["changes"] += 1
                    st["window_changes"] += 1
                    prev = st["last"]
                    for i in range(max(len(prev), len(data))):
                        if i >= len(prev) or i >= len(data) or prev[i] != data[i]:
                            st["changed"].add(i)
                st["last"] = data

    def report(self, dt: float | None, final=False) -> str:
        with self.lock:
            rows = [f"frames={self.frames} (no signal {self.frames_no_signal})  packet types={len(self.types)}"
                    f"  RP188 timecode={self.timecode}"]
            if not self.types:
                rows.append("  (ANC 패킷 없음)")
            rows.append("  space line  DID  SDID  " + ("  /s  " if dt else " count") +
                        " len  distinct chg/s  changed bytes            name / last payload")
            for key in sorted(self.types):
                space, line, did, sdid, stream = key
                st = self.types[key]
                rate = f"{st['window'] / dt:5.1f}" if dt else f"{st['count']:6d}"
                chg = f"{st['window_changes'] / dt:5.1f}" if dt else f"{st['changes']:5d}"
                lens = "/".join(str(n) for n in sorted(st["lens"]))
                ch = _ranges(sorted(st["changed"]))
                mark = " *" if st["window_changes"] and dt else "  "
                rows.append(f"{mark}{space:4s} {line:4d}  0x{did:02X} 0x{sdid:02X} {rate} {lens:>4s} "
                            f"{len(st['distinct']):8d} {chg} {ch[:24]:24s} {describe(did, sdid)}")
                rows.append(f"          last: {st['last'][:48].hex(' ')}{' ...' if len(st['last']) > 48 else ''}")
                st["window"] = 0
                st["window_changes"] = 0
            if final:
                rows.append("* = 이번 구간에 내용이 바뀐 패킷. 줌을 돌릴 때만 바뀌는 패킷이 렌즈 데이터 후보.")
            return "\n".join(rows)


def _ranges(idx: list[int]) -> str:
    if not idx:
        return "-"
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == b + 1:
            b = i
            continue
        out.append(f"{a}" if a == b else f"{a}-{b}")
        a = b = i
    out.append(f"{a}" if a == b else f"{a}-{b}")
    return ",".join(out)


# ---------------------------------------------------------------- video recording
class VideoRecorder:
    """Writes the incoming 10-bit YUV (v210) frames as a small gray video next to the log,
    so lens values can be compared with what the picture did. Takes the first field's luma
    at half width (960x540): cheap enough to do in the capture callback."""

    def __init__(self, api, path: Path, fps: float = 30000 / 1001):
        import queue

        import cv2
        self.api, self.path, self.cv2 = api, path, cv2
        self.q: queue.Queue = queue.Queue(maxsize=120)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
        self.count = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, frame) -> int | None:
        if frame.GetPixelFormat() != self.api.bmdFormat10BitYUV or frame.GetWidth() != 1920:
            return None
        h, rb = frame.GetHeight(), frame.GetRowBytes()
        buf = frame.QueryInterface(self.api.IDeckLinkVideoBuffer)
        buf.StartAccess(self.api.bmdBufferAccessRead)
        try:
            ptr = buf.GetBytes()
            a = np.ctypeslib.as_array((ctypes.c_uint32 * (rb // 4 * h)).from_address(ptr))
            w = a.reshape(h, rb // 4)[::2, :1280].reshape(h // 2, 320, 4)
            # v210 words: [Cb0 Y0 Cr0] [Y1 Cb1 Y2] [Cr1 Y3 Cb2] [Y4 Cr2 Y5] -> Y0, Y2, Y4 = every other pixel
            y = np.stack([(w[..., 0] >> 10) & 0x3FF, (w[..., 1] >> 20) & 0x3FF, w[..., 3] & 0x3FF], -1)
            gray = (y.reshape(h // 2, 960) >> 2).astype(np.uint8)
        finally:
            buf.EndAccess(self.api.bmdBufferAccessRead)
        try:
            self.q.put_nowait(gray)
        except Exception:
            self.dropped += 1
            return None
        self.count += 1
        return self.count - 1

    def _run(self):
        while True:
            g = self.q.get()
            if g is None:
                break
            self.writer.write(self.cv2.cvtColor(g, self.cv2.COLOR_GRAY2BGR))

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.writer.release()
        print(f"[INFO] video: {self.path} frames={self.count} dropped={self.dropped}")


# ---------------------------------------------------------------- DeckLink capture
def load_api():
    sys.coinit_flags = 0  # COINIT_MULTITHREADED: DeckLink calls back on its own threads
    import comtypes  # noqa: F401
    import comtypes.client as cc
    return cc.GetModule(DECKLINK_TLB)


def list_devices(api):
    import comtypes.client as cc
    it = cc.CreateObject(api.CDeckLinkIterator, interface=api.IDeckLinkIterator)
    devs = []
    while True:
        try:
            d = it.Next()
        except Exception:
            break
        if not d:
            break
        devs.append((d.GetDisplayName(), d))
    return devs


def capture(args):
    import comtypes
    api = load_api()
    devs = list_devices(api)
    if not devs:
        sys.exit("DeckLink 장치를 찾지 못했습니다 (Desktop Video 드라이버 확인).")
    if args.list:
        for i, (name, _) in enumerate(devs):
            print(f"[{i}] {name}")
        return
    if args.device.isdigit():
        name, dev = devs[int(args.device)]
    else:
        name, dev = next((n, d) for n, d in devs if args.device.lower() in n.lower())
    print(f"[INFO] device: {name}")

    attrs = dev.QueryInterface(api.IDeckLinkProfileAttributes)
    try:
        print(f"[INFO] HANC input supported: {bool(attrs.GetFlag(api.BMDDeckLinkSupportsHANCInput))}")
    except Exception:
        pass
    cfg = dev.QueryInterface(api.IDeckLinkConfiguration)
    for i, pair in enumerate(args.hanc[:4]):
        did, sdid = (int(x, 0) for x in pair.split(":"))
        key = getattr(api, f"bmdDeckLinkConfigHANCInputFilter{i + 1}")
        cfg.SetInt(key, (did << 8) | sdid)
        print(f"[INFO] HANC filter {i + 1}: DID 0x{did:02X} SDID 0x{sdid:02X}")

    inp = dev.QueryInterface(api.IDeckLinkInput)
    stats = PacketStats()
    log_path = Path(args.log) if args.log else ROOT / "anc_logs" / time.strftime("anc_%Y%m%d_%H%M%S.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    log_lock = threading.Lock()
    t0 = time.perf_counter()
    spaces = {api.bmdAncillaryDataSpaceVANC: "VANC", api.bmdAncillaryDataSpaceHANC: "HANC"}
    watch_id = tuple(int(x, 0) for x in args.watch.split(":")) if args.watch else None
    if watch_id:
        args.vanc_lines = 0   # the packet API already gives it; skip the raw scan
    video = None
    if args.record_video:
        vpath = log_path.with_suffix(".mp4") if args.record_video == "auto" else Path(args.record_video)
        video = VideoRecorder(api, vpath)
        print(f"[INFO] recording video (gray 960x540, one field) -> {vpath}")
    log.write(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "video": str(video.path) if video else None}) + "\n")

    def read_raw_vanc(frame) -> list[tuple]:
        """Scan the raw v210 VANC lines for ANC flags (000 3FF 3FF), in case the driver
        does not hand a packet out through the packet API."""
        out = []
        try:
            anc = frame.GetAncillaryData()
        except comtypes.COMError:
            return out
        rb = frame.GetRowBytes()
        for ln in range(1, args.vanc_lines + 1):
            try:
                buf = anc.GetBufferForVerticalBlankingLine(ln)
            except comtypes.COMError:
                continue
            a = np.frombuffer(ctypes.string_at(buf, rb), dtype=np.uint32)
            w = np.stack([a & 0x3FF, (a >> 10) & 0x3FF, (a >> 20) & 0x3FF], 1).reshape(-1)   # C Y C Y ...
            for sname, s in (("RAWY", w[1::2]), ("RAWC", w[0::2])):
                for i in np.where((s[:-5] == 0) & (s[1:-4] == 0x3FF) & (s[2:-3] == 0x3FF))[0]:
                    did, sdid, dc = (int(x) & 0xFF for x in s[i + 3:i + 6])
                    data = bytes(int(x) & 0xFF for x in s[i + 6:i + 6 + dc])
                    out.append(((sname, ln, did, sdid, 0), data))
        return out

    def read_timecode(frame) -> str | None:
        try:
            tc = frame.GetTimecode(api.bmdTimecodeRP188Any)
            return tc.GetString() if tc else None
        except comtypes.COMError:
            return None

    def read_packets(frame) -> list[tuple]:
        out = []
        try:
            anc = frame.QueryInterface(api.IDeckLinkVideoFrameAncillaryPackets)
            it = anc.GetPacketIterator()
        except comtypes.COMError:
            return out
        while True:
            try:
                p = it.Next()
            except comtypes.COMError:
                break
            if not p:
                break
            try:
                ptr, size = p.GetBytes(api.bmdAncillaryPacketFormatUInt8)
                data = ctypes.string_at(ptr, size) if ptr and size else b""
            except comtypes.COMError:
                data = b""
            key = (spaces.get(p.GetDataSpace(), "?"), int(p.GetLineNumber()), int(p.GetDID()),
                   int(p.GetSDID()), int(p.GetDataStreamIndex()))
            out.append((key, data))
        return out

    class Callback(comtypes.COMObject):
        _com_interfaces_ = [api.IDeckLinkInputCallback]

        # ctypes already AddRefs COM in-arguments (_needs_com_addref_) and comtypes releases them,
        # so no AddRef here: an extra one leaks every frame.
        def VideoInputFormatChanged(self, events, mode, flags):
            try:
                pix = api.bmdFormat10BitRGB if flags & api.bmdDetectedVideoInputRGB444 else api.bmdFormat10BitYUV
                print(f"\n[INFO] input format: {mode.GetName()} ({'RGB' if pix == api.bmdFormat10BitRGB else 'YUV'})")
                inp.PauseStreams()
                inp.EnableVideoInput(mode.GetDisplayMode(), pix, api.bmdVideoInputEnableFormatDetection)
                inp.FlushStreams()
                inp.StartStreams()
            except Exception as e:  # never let an exception cross into the driver
                print(f"[WARN] format change: {e}")
            return 0

        def VideoInputFrameArrived(self, frame, audio):
            if not frame:
                return 0
            try:
                no_signal = bool(frame.GetFlags() & api.bmdFrameHasNoInputSource)
                vf = video.add(frame) if video else None
                pk = read_packets(frame)
                if args.vanc_lines:
                    pk += read_raw_vanc(frame)
                stats.timecode = read_timecode(frame)
                stats.add_frame(pk, no_signal)
                if watch_id:   # long runs: log only the watched packet (about 20 MB/hour)
                    pk = [(k, d) for k, d in pk if k[0] == "VANC" and (k[2], k[3]) == watch_id]
                rec = {"t": round(time.perf_counter() - t0, 4), "ns": no_signal, "tc": stats.timecode,
                       "p": [[*k, d.hex()] for k, d in pk]}
                if vf is not None:
                    rec["vf"] = vf   # frame number in the recorded video
                with log_lock:
                    log.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[WARN] frame: {e}")
            return 0

    cb = Callback()
    inp.SetCallback(cb)
    mode = getattr(api, args.mode)
    inp.EnableVideoInput(mode, api.bmdFormat10BitYUV, api.bmdVideoInputEnableFormatDetection)
    inp.StartStreams()
    print(f"[INFO] capturing ({args.mode}, format detection on) -> {log_path}")
    print("[INFO] 렌즈를 와이드 <-> 텔레로 돌리고, 포커스·아이리스도 움직여 보세요. Ctrl+C 로 종료.")

    end = time.perf_counter() + args.seconds if args.seconds > 0 else None
    last = time.perf_counter()
    try:
        if args.watch:
            watch(args, stats, t0, end)
        while not args.watch and (end is None or time.perf_counter() < end):
            time.sleep(args.interval)
            now = time.perf_counter()
            print(f"\n--- {now - t0:6.1f} s ---")
            print(stats.report(now - last))
            last = now
    except KeyboardInterrupt:
        pass
    finally:
        inp.StopStreams()
        inp.DisableVideoInput()
        inp.SetCallback(None)
        with log_lock:
            log.close()
        if video:
            video.close()
    print("\n=== 전체 ===")
    print(stats.report(None, final=True))
    print(f"[INFO] log: {log_path}")


def watch(args, stats: PacketStats, t0: float, end: float | None):
    """Long-running view of one packet: the chosen bytes as big-endian 16-bit words, one line
    only when a word moves more than the noise threshold (plus a heartbeat), so it can stay
    open during a shoot and show when the lens moves."""
    did, sdid = (int(x, 0) for x in args.watch.split(":"))
    a, b = (int(x) for x in args.watch_bytes.split("-"))
    print(f"[INFO] watching DID 0x{did:02X} SDID 0x{sdid:02X} bytes {a}-{b} as 16-bit words, "
          f"noise {args.watch_noise}. 값이 움직일 때만 한 줄씩 찍힙니다.")
    printed, lo, hi, last_beat = None, None, None, 0.0
    while end is None or time.perf_counter() < end:
        time.sleep(0.05)
        with stats.lock:
            keys = sorted(k for k in stats.types if k[0] == "VANC" and k[2] == did and k[3] == sdid)
            data = stats.types[keys[0]]["last"] if keys else None
        now = time.perf_counter() - t0
        if data is None or len(data) <= b:
            if now - last_beat > args.watch_heartbeat:
                last_beat = now
                print(f"{time.strftime('%H:%M:%S')} {now:8.1f}s  (패킷 없음)")
            continue
        seg = data[a:b + 1]
        words = [int.from_bytes(seg[i:i + 2], "big") for i in range(0, len(seg) - 1, 2)]
        lo = words if lo is None else [min(x, y) for x, y in zip(lo, words)]
        hi = words if hi is None else [max(x, y) for x, y in zip(hi, words)]
        moved = printed is None or any(abs(x - y) > args.watch_noise for x, y in zip(words, printed))
        if moved or now - last_beat > args.watch_heartbeat:
            mark = "MOVE" if moved and printed is not None else "    "
            delta = " ".join(f"{x - y:+6d}" for x, y in zip(words, printed or words))
            print(f"{time.strftime('%H:%M:%S')} {now:8.1f}s {mark}  " + " ".join(f"{w:04x}" for w in words)
                  + f"  | d {delta}  | range " + " ".join(f"{y - x:5d}" for x, y in zip(lo, hi)))
            last_beat = now
            if moved:
                printed = words


def analyze(path: str):
    """Summary of a saved log, plus the value of every changing packet over time."""
    stats = PacketStats()
    timeline: dict[tuple, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "p" not in rec:   # header line
                continue
            pk = [(tuple(p[:5]), bytes.fromhex(p[5])) for p in rec["p"]]
            stats.timecode = rec.get("tc")
            stats.add_frame(pk, rec["ns"])
            for k, d in pk:
                tl = timeline.setdefault(k, [])
                if not tl or tl[-1][1] != d:
                    tl.append((rec["t"], d))
    print(stats.report(None, final=True))
    for k, tl in sorted(timeline.items()):
        if len(tl) < 2:
            continue
        print(f"\n{k[0]} line {k[1]} DID 0x{k[2]:02X} SDID 0x{k[3]:02X}: {len(tl)} 번 바뀜 (처음 20개)")
        for t, d in tl[:20]:
            print(f"  {t:8.3f} s  {d[:48].hex(' ')}")


def main():
    ap = argparse.ArgumentParser(description="DeckLink SDI ANC 패킷 확인 (렌즈 줌 데이터 찾기)")
    ap.add_argument("--list", action="store_true", help="DeckLink 장치 목록")
    ap.add_argument("--device", default="0", help="장치 번호 또는 이름 일부")
    ap.add_argument("--mode", default="bmdModeHD1080p5994", help="시작 모드 (형식 자동 감지로 바뀜)")
    ap.add_argument("--seconds", type=float, default=0, help="캡처 시간 (0 = Ctrl+C 까지)")
    ap.add_argument("--interval", type=float, default=1.0, help="화면 출력 간격 (초)")
    ap.add_argument("--vanc-lines", type=int, default=41, help="원시 VANC 를 직접 검사할 라인 수 (0 = 끔). RAWY/RAWC 로 표시")
    ap.add_argument("--hanc", action="append", default=[], help="HANC 에서 받을 DID:SDID (예 0x43:0x05, 최대 4)")
    ap.add_argument("--log", default=None, help="JSONL 로그 경로 (기본 anc_logs/anc_<시각>.jsonl)")
    ap.add_argument("--analyze", default=None, help="저장된 로그 분석")
    ap.add_argument("--record-video", nargs="?", const="auto", default=None,
                    help="영상도 함께 녹화 (회색 960x540, 로그와 같은 이름의 .mp4). 로그의 vf = 영상 프레임 번호")
    ap.add_argument("--watch", default=None, help="이 DID:SDID 패킷만 계속 감시 (예 0x51:0x01)")
    ap.add_argument("--watch-bytes", default="63-70", help="감시할 바이트 범위 (16비트 단위로 표시)")
    ap.add_argument("--watch-noise", type=int, default=16, help="이보다 크게 바뀔 때만 출력 (떨림 무시)")
    ap.add_argument("--watch-heartbeat", type=float, default=30.0, help="변화가 없어도 현재 값을 찍는 간격 (초)")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        capture(args)


if __name__ == "__main__":
    main()
