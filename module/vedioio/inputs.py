# io/inputs.py
from __future__ import annotations
import os, sys, time, threading, queue, subprocess, shlex, shutil
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2  # <- 필요 시 색공간 변환에 사용

# --- 작은 유틸 ---
def parse_fps(s) -> float:
    s = str(s)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def resolve_ffmpeg_bin(ffmpeg_bin: str | os.PathLike | None = None) -> str:
    """Resolve FFmpeg without depending on the current working directory.

    Resolution order for the default value is:
      1. RVM_FFMPEG_BIN environment variable
      2. project-local/sibling ``FFmpeg\\bin\\ffmpeg.exe``
      3. PATH

    An explicitly supplied path is always tried first. Relative explicit paths
    are resolved against both the project root and the current directory.
    """
    project_root = Path(__file__).resolve().parents[2]
    requested = os.path.expandvars(os.path.expanduser(str(ffmpeg_bin or "ffmpeg"))).strip()
    default_names = {"ffmpeg", "ffmpeg.exe"}
    candidates: list[Path] = []

    if requested.lower() not in default_names:
        explicit = Path(requested)
        if explicit.is_absolute():
            candidates.append(explicit)
        else:
            candidates.extend((project_root / explicit, Path.cwd() / explicit))

    env_value = os.environ.get("RVM_FFMPEG_BIN", "").strip()
    if env_value:
        candidates.append(Path(os.path.expandvars(os.path.expanduser(env_value))))

    candidates.extend((
        project_root / "FFmpeg" / "bin" / "ffmpeg.exe",
        project_root.parent / "FFmpeg" / "bin" / "ffmpeg.exe",
    ))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    found = shutil.which(requested if requested.lower() in default_names else "ffmpeg")
    if found:
        return str(Path(found).resolve())

    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "FFmpeg 실행 파일을 찾지 못했습니다. --ffmpeg-bin 또는 "
        "RVM_FFMPEG_BIN을 지정하세요.\n  - " + searched
    )

def _bytes_per_px(fmt: str) -> int:
    fmt = (fmt or "").lower()
    # rawvideo 파이프에서 가장 흔한 포맷들
    if fmt in ("bgr24", "rgb24"):     return 3
    if fmt in ("bgra", "rgba", "bgr0", "rgb0"): return 4
    if fmt in ("yuyv422", "yuy2", "uyvy422", "uyvy"): return 2
    # 기본값은 3바이트로 보수 처리
    return 3

def _ffmpeg_pix_fmt(fmt: str) -> str:
    fmt = (fmt or "").lower()
    # OpenCV 변환 코드와 매칭되는 ffmpeg 픽셀포맷 명칭
    if fmt in ("bgr24", "rgb24"): return "bgr24"
    if fmt in ("bgra", "rgba", "bgr0", "rgb0"): return "bgra"
    if fmt in ("yuyv422", "yuy2"): return "yuyv422"
    if fmt in ("uyvy422", "uyvy"): return "uyvy422"
    # 모르면 bgr24
    return "bgr24"

def _is_interlaced_code(code: Optional[str]) -> bool:
    """1080i5994, 1080i5000 등 'i' 포함이면 인터레이스 가정"""
    if not code: return False
    c = code.lower()
    # 간단 휴리스틱: 'i' 들어있고 'p'는 없는 경우
    return ('i' in c) and ('p' not in c)

def _read_exact(pipe, n: int) -> bytes:
    """파이프에서 정확히 n바이트를 모을 때까지 반복 읽기 (저지연, 부분읽기 대응)"""
    chunks = []
    got = 0
    while got < n:
        chunk = pipe.read(n - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)

# ---------- FFmpeg DeckLink 파이프 ----------
def _open_ffmpeg_decklink_pipe(
    device_name: str,
    out_size: Tuple[int, int],
    out_fps: str | float,
    raw_format: str = "bgr24",
    ffmpeg_bin: str = "ffmpeg",
    decklink_format: Optional[str] = None,   # 예: '1080p5994', '1080i5994', '2160p2997'
):
    """
    변경점:
      - (A-1) scale/format 필터 제거 → CPU 변환 비용 최소화
      - (A-1) 저지연 플래그 추가: -fflags nobuffer -flags low_delay -avioflags direct
      - (A-3) 프레임 바이트 고정 계산 후 정확 읽기
    """
    reqW, reqH = int(out_size[0]), int(out_size[1])

    pix_fmt = _ffmpeg_pix_fmt(raw_format)  # 가급적 'bgr24'를 권장(파이썬/torch로 바로 사용)
    bpp = _bytes_per_px(pix_fmt)

    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
        # 저지연 핵심 플래그
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-avioflags", "direct",
        # DeckLink 입력
        "-f", "decklink",
    ]
    if decklink_format:
        cmd += ["-format_code", str(decklink_format)]

    cmd += [
        "-i", device_name,
        "-an",
        "-vsync", "0",
        # 필터 체인: 기본적으로 제거. 인터레이스일 때만 yadif 적용.
    ]

    # 인터레이스이면 1프레임 디인터레이스(yadif=1). scale/format은 넣지 않음.
    if _is_interlaced_code(decklink_format):
        cmd += ["-vf", "yadif=1"]

    cmd += [
        "-vcodec", "rawvideo",
        "-pix_fmt", pix_fmt,       # 출력 rawvideo 픽셀포맷 지정
        "-f", "rawvideo",
        "pipe:1"
    ]

    try:
        subprocess.run([ffmpeg_bin, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print("[WARN] ffmpeg 실행 확인 실패:", e)

    print("[INFO] Starting ffmpeg DeckLink pipe:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"[INFO] Using DeckLink input: {device_name}" + (f" (format_code={decklink_format})" if decklink_format else ""))
    print(f"[INFO] Expect pixel format: {pix_fmt} (bpp={bpp}) ~{reqW}x{reqH} (입력 소스 해상도에 따름)")

    # 파이썬 파이프 버퍼는 프레임 몇 장 분량으로 넉넉히
    frame_bytes = reqW * reqH * bpp
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout 파이프 생성 실패")

    return proc, frame_bytes, reqW, reqH, pix_fmt

def _open_ffmpeg_dshow_pipe(
    dev_name: str,
    out_size: Tuple[int, int],
    out_fps: str | float,
    raw_format: str = "bgr24",
    ffmpeg_bin: str = "ffmpeg",
    rtbufsize: str = "256M",
):
    """
    변경점:
      - (A-1) scale/format 필터 제거 (가능하면 원본 크기, 원본 포맷에서 ffmpeg가 최소 변환)
      - (A-1) 저지연 플래그 유지/강화
      - (A-3) 프레임 바이트 고정 계산 후 정확 읽기
    """
    reqW, reqH = int(out_size[0]), int(out_size[1])
    pix_fmt = _ffmpeg_pix_fmt(raw_format)   # 가급적 'bgr24' 권장
    bpp = _bytes_per_px(pix_fmt)

    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",

        # ★ 저지연 핵심 플래그
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-avioflags", "direct",

        # dshow 입력
        "-f", "dshow",
        "-rtbufsize", str(rtbufsize),
        "-thread_queue_size", "8",
        "-use_wallclock_as_timestamps", "1",
        "-analyzeduration", "0",
        "-probesize", "32",
        "-video_size", f"{reqW}x{reqH}",
        "-framerate", str(out_fps),
        "-i", f"video={dev_name}",

        # rawvideo 출력
        "-vsync", "0",
        "-vcodec", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-f", "rawvideo", "pipe:1",
    ]

    print("[INFO] Starting ffmpeg dshow pipe:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))

    frame_bytes = reqW * reqH * bpp
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 2)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe open failed")

    return proc, frame_bytes, reqW, reqH, pix_fmt

class _DeckLinkGrab:
    """
    FFmpeg rawvideo 파이프에서 프레임 단위로 정확히 읽고(np view) 반환.
    픽셀포맷을 bgr24로 맞추면 cv2 변환이 필요 없으므로 CPU 부담이 줄어듭니다.
    """
    def __init__(self, proc, frame_bytes, W, H, pix_fmt: str):
        self.proc = proc
        self.frame_bytes = frame_bytes
        self.W, self.H = W, H
        self.pix_fmt = (pix_fmt or "bgr24").lower()

    def read(self, timeout: float = 1.0):
        buf = _read_exact(self.proc.stdout, self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return False, None

        if self.pix_fmt == "bgra":
            arr = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 4)
            # BGRA -> BGR (필요 시)
            frm = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif self.pix_fmt in ("yuyv422", "yuy2"):
            arr = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 2)
            frm = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
        elif self.pix_fmt in ("uyvy422", "uyvy"):
            arr = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 2)
            frm = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
        else:
            # bgr24 (또는 기타 3B/px 가정) → 변환 없이 바로 사용
            frm = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3)

        return True, frm

    def release(self):
        try:
            if self.proc.stdout is not None:
                self.proc.stdout.close()
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

# ---------- DirectShow Reader (PyAV) ----------
class _AVReader:
    def __init__(self, dev_name="OBS Virtual Camera", size="1920x1080",
                 fps="60000/1001", rtbufsize="1024M"):
        import av
        opts = {"rtbufsize": str(rtbufsize)}
        if size: opts["video_size"] = size
        if fps:  opts["framerate"]  = str(fps)
        url = f"video={dev_name}"
        self.container = av.open(url, format="dshow", options=opts)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._iter = self.container.demux(self.stream)
        print(f"[INFO] AVReader open: name='{dev_name}', size={size}, fps={fps}, rtbufsize={rtbufsize}")
        try:
            print(f"[INFO] AV stream avg_rate={self.stream.average_rate} time_base={self.stream.time_base}")
        except Exception:
            pass
    def isOpened(self): return True
    def read(self):
        for pkt in self._iter:
            for frame in pkt.decode():
                img = frame.to_ndarray(format="bgr24")
                return True, img
        return False, None
    def release(self):
        try: self.container.close()
        except Exception: pass

class _ThreadedGrabber:
    """
    (D-3) oldest-drop 정책: 큐가 가득 차면 가장 오래된 프레임을 버리고 최신 프레임을 유지.
    """
    def __init__(self, reader, max_queue=1):
        self.reader = reader
        self.q = queue.Queue(maxsize=max_queue)
        self.alive = False
        self.t: Optional[threading.Thread] = None
        self.frames_read = 0
        self.frames_dropped = 0
        self.last_error: Optional[BaseException] = None
    def start(self):
        self.alive = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
    def _loop(self):
        while self.alive:
            try:
                ok, frm = self.reader.read()
            except BaseException as exc:
                self.last_error = exc
                break
            if not ok:
                time.sleep(0.001); continue
            self.frames_read += 1
            if self.q.full():
                try:
                    self.q.get_nowait()   # oldest-drop
                    self.frames_dropped += 1
                except queue.Empty: pass
            try: self.q.put_nowait(frm)
            except queue.Full: pass
    def read(self, timeout=1.0):
        try:
            frm = self.q.get(timeout=timeout); return True, frm
        except queue.Empty:
            return False, None
    def release(self):
        self.alive = False
        # Close the underlying pipe first so a blocking read can return.
        try: self.reader.release()
        except Exception: pass
        if self.t and self.t.is_alive():
            self.t.join(timeout=1.0)

    def stats(self) -> dict[str, int]:
        return {
            "captured": int(self.frames_read),
            "stale_dropped": int(self.frames_dropped),
        }

# ---------- Public Factory ----------
def create_input_source(
    *,
    ffmpeg_in: str = "dshow",                  # "decklink" | "dshow"
    dshow_name: str = "OBS Virtual Camera",
    force_size: str = "1920x1080",
    force_fps: str = "60000/1001",
    rtbufsize: str = "1024M",
    raw_format: str = "bgr24",                 # "bgr24" | "bgra" | "yuyv422" | "uyvy422"
    ffmpeg_bin: str = "ffmpeg",
    decklink_device: Optional[str] = None,
    decklink_format: Optional[str] = None,     # '1080p5994' 등
):
    """
    Returns:
        grab  : 객체 (read(timeout)->(ok, frame), release())
        W, H  : 입력 프레임 해상도(첫 프레임 기준)
        obs_fps_from_input : OBS 가상카메라에 사용할 권장 fps(입력기반)
    Raises:
        RuntimeError on open failure

    변경 요약:
      - (A-1) FFmpeg 단계의 scale/format 제거로 CPU 부하 감소
      - (A-1) 저지연 플래그 추가/유지
      - (A-3) 정확한 바이트 읽기 적용
      - (D-3) oldest-drop 큐 정책 유지
    """
    ffmpeg_bin = resolve_ffmpeg_bin(ffmpeg_bin)
    print(f"[INFO] FFmpeg: {ffmpeg_bin}")

    # 해상도 파싱
    try:
        w_str, h_str = str(force_size).lower().split("x")
        req_W, req_H = int(w_str), int(h_str)
    except Exception:
        raise ValueError(f"--force-size 형식 오류: {force_size} (예: 1920x1080)")

    if ffmpeg_in == "decklink":
        if not (decklink_device and force_size and force_fps):
            raise AssertionError("DeckLink 입력은 --decklink-device, --force-size, --force-fps 필요")
        proc, frame_bytes, W, H, pix_fmt = _open_ffmpeg_decklink_pipe(
            decklink_device, (req_W, req_H), force_fps,
            raw_format=raw_format, ffmpeg_bin=ffmpeg_bin, decklink_format=decklink_format
        )
        grab = _ThreadedGrabber(_DeckLinkGrab(proc, frame_bytes, W, H, pix_fmt=pix_fmt), max_queue=1)
        grab.start()
        ok, frm = grab.read(timeout=2.0)
        if not ok:
            raise RuntimeError("프레임 읽기 실패 (ffmpeg decklink)")
        H, W = frm.shape[:2]
        print(f"[INFO] Using DeckLink input: {decklink_device}, format_code={decklink_format}")
        print(f"[INFO] Input size (first frame): {W}x{H}")
        obs_fps_from_input = parse_fps(force_fps)   # 입력 fps를 그대로 추천
        return grab, W, H, obs_fps_from_input

    if ffmpeg_in == "dshow":
        proc, frame_bytes, W, H, pix_fmt = _open_ffmpeg_dshow_pipe(
            dshow_name, (req_W, req_H), force_fps, raw_format, ffmpeg_bin,
            rtbufsize=str(rtbufsize),
        )
        grab = _ThreadedGrabber(_DeckLinkGrab(proc, frame_bytes, W, H, pix_fmt=pix_fmt), max_queue=1)
        grab.start()
        ok, frm = grab.read(timeout=2.0)
        if not ok:
            raise RuntimeError("프레임 읽기 실패 (ffmpeg dshow)")
        H, W = frm.shape[:2]
        print(f"[INFO] Using dshow(ffmpeg) input: {dshow_name}")
        print(f"[INFO] Input size (first frame): {W}x{H}")
        obs_fps_from_input = parse_fps(force_fps)
        return grab, W, H, obs_fps_from_input
    
    # dshow (PyAV)
    cap = _AVReader(dshow_name, force_size, str(force_fps), rtbufsize=str(rtbufsize))
    grab = _ThreadedGrabber(cap); grab.start()
    ok, frm = grab.read(timeout=2.0)
    if not ok:
        raise RuntimeError("프레임 읽기 실패")
    H, W = frm.shape[:2]
    print(f"[INFO] Input size (first frame): {W}x{H}")
    # DirectShow는 설정값을 추천값으로 반환
    obs_fps_from_input = parse_fps(force_fps)
    return grab, W, H, obs_fps_from_input
