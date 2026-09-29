# module/vedioio/decklink_out.py
# -*- coding: utf-8 -*-

import subprocess, shlex, threading, sys, time, queue, atexit
from typing import Optional, Union, Sequence
import numpy as np


class DeckLinkOut:
    """
    DeckLink 출력용 FFmpeg 파이프 송출기 (지연 누적 방지 버전)

    핵심 변경점
    - stdin에 직접 write하지 않고, maxsize=1 큐에 '최신 프레임만 유지' → 백그라운드 라이터가 ffmpeg에 전달
    - 파이프 입력은 `-re -framerate <fps>`로 설정하여 과도한 읽기/타임스탬프 왜곡 방지
    - 옵션으로 scale, 픽셀 포맷 변환, 추가 -vf 체인 지원

    Args:
        out_device: DeckLink 장치 이름 (예: "DeckLink Quad (2)")
        format_code: DeckLink 포맷 코드 (예: "1080p2997", "1080p5994" 등)
        size: 입력(rawvideo) 프레임 크기 (W, H)
        fps: 입력(rawvideo) 프레임레이트 (예: "30000/1001" 또는 29.97 또는 30)
        in_pix_fmt: 파이프 입력 포맷 ("bgr24" | "bgra" | "uyvy422")
        ffmpeg_bin: ffmpeg 실행 경로
        out_pix_fmt: DeckLink 출력 픽셀 포맷 (None 또는 "uyvy422"/"v210" 등)
        scale_to: 출력 전 스케일 목표 (예: "1920x1080"). None이면 스케일 생략
        extra_vf: 추가 비디오 필터 문자열(들). scale/format 뒤에 이어붙음
        decklink_opts: decklink muxer에 추가할 옵션 리스트 (예: ["-duplex_mode", "full"])
    """

    def __init__(
        self,
        out_device: str,
        format_code: str,
        size: tuple[int, int],
        fps: Union[str, float, int],
        in_pix_fmt: str = "bgr24",
        ffmpeg_bin: str = "ffmpeg",
        out_pix_fmt: Optional[str] = "uyvy422",
        scale_to: Optional[str] = None,
        extra_vf: Optional[Union[str, Sequence[str]]] = None,
        decklink_opts: Optional[Sequence[str]] = None,
    ):
        W, H = size
        self.W, self.H = int(W), int(H)
        self.in_pix_fmt = (in_pix_fmt or "bgr24").lower()
        self.out_pix_fmt = (out_pix_fmt.lower() if out_pix_fmt else None)
        self.scale_to = (scale_to.lower() if scale_to else None)
        self.proc: Optional[subprocess.Popen] = None

        # 외부 루프용 힌트 플래그(기존 코드 호환)
        self.use_sleep = False         # DL 클럭이 있으니 외부 슬립은 보통 불필요
        self.dl = True                 # "DeckLink 클럭 있음" 신호
        self._sent = 0

        # 최신 프레임만 보관하는 큐
        self.q: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stop = False

        # --------- 비디오 필터 체인 구성 ---------
        vf_chain = []
        # 1) scale
        if self.scale_to:
            try:
                out_w, out_h = map(int, self.scale_to.split("x"))
            except Exception:
                raise ValueError(f"[DeckLinkOut] scale_to 형식 오류: {self.scale_to} (예: '1920x1080')")
            if (out_w != self.W) or (out_h != self.H):
                vf_chain.append(f"scale={out_w}:{out_h}")

        # 2) format (출력 픽셀포맷을 DeckLink에 넘길 때 ffmpeg 필터로 맞춤)
        if self.out_pix_fmt and (self.out_pix_fmt != self.in_pix_fmt):
            vf_chain.append(f"format={self.out_pix_fmt}")

        # 3) 추가 사용자 필터
        if extra_vf:
            if isinstance(extra_vf, str):
                vf_chain.append(extra_vf)
            else:
                vf_chain.extend(extra_vf)

        # --------- FFmpeg 커맨드 구성 ---------
        fps_str = str(fps)
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y",
            "-re",                                  # ★ 실시간 속도로 읽기(버퍼 폭주 방지)
            "-f", "rawvideo",
            "-pix_fmt", self.in_pix_fmt,
            "-video_size", f"{self.W}x{self.H}",
            "-framerate", fps_str,                  # ★ rawvideo 입력 fps
            "-i", "pipe:0",
            "-an",
        ]

        if vf_chain:
            cmd += ["-vf", ",".join(vf_chain)]

        # DeckLink 출력 (포맷 코드 + 옵션)
        cmd += ["-f", "decklink", "-format_code", str(format_code)]
        if decklink_opts:
            # 예: ["-duplex_mode","full","-queue_size","16"]
            cmd += list(decklink_opts)

        if self.out_pix_fmt:
            cmd += ["-pix_fmt", self.out_pix_fmt]

        cmd += [out_device]

        print("[INFO] Starting FFmpeg DeckLink OUT:")
        print("  " + " ".join(shlex.quote(str(c)) for c in cmd))

        # 버퍼가 쌓이지 않도록 bufsize=0 (파이프 무버퍼링)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        # GPU 패스스루 경로 식별용
        self.accepts_uyvy_bytes = (self.in_pix_fmt == "uyvy422") and (self.out_pix_fmt in (None, "uyvy422"))
        self.is_decklink_raw = self.accepts_uyvy_bytes

        # stderr 로그 펌프
        threading.Thread(target=self._stderr_pump, args=(self.proc,), daemon=True).start()
        # 전송 스레드
        threading.Thread(target=self._writer_loop, daemon=True).start()
        # 송출률 모니터(옵션)
        threading.Thread(target=self._heartbeat, daemon=True).start()

        atexit.register(self.close)

    # ---------------- 내부 스레드 ----------------

    def _stderr_pump(self, p: subprocess.Popen):
        while True:
            if p.poll() is not None:
                break
            line = p.stderr.readline()
            if not line:
                break
            try:
                sys.stderr.write("[DL-OUT] " + line.decode("utf-8", "ignore"))
            except Exception:
                pass

    def _writer_loop(self):
        """최신 프레임만 ffmpeg stdin으로 전송. 큐가 비면 잠깐 대기."""
        last = b""
        stdin = self.proc.stdin if (self.proc and self.proc.stdin) else None
        while not self._stop and self.proc and (self.proc.poll() is None) and stdin:
            try:
                item = self.q.get(timeout=0.05)  # 새 프레임이 오면 교체
                last = item
            except queue.Empty:
                # 새 프레임이 없으면 직전 프레임 유지(필요시 정지)
                if not last:
                    continue
                item = last
            try:
                stdin.write(item)
                self._sent += 1
            except Exception as e:
                print("[DL-OUT] write failed:", e)
                self._stop = True
                if self.proc is not None:
                    code = self.proc.poll()
                    print(f"[DL-OUT] ffmpeg poll() after write fail → {code}")
                break

    def _heartbeat(self):
        last = 0
        while not self._stop and self.proc and (self.proc.poll() is None):
            time.sleep(2.0)
            cur = self._sent
            fps_val = (cur - last) / 2.0
            last = cur
            # 필요시 모니터링 출력
            # print(f"[DL-OUT] sending ~{fps_val:.1f} fps", flush=True)

    # ---------------- 외부 API ----------------

    def submit(self, frame: Union[np.ndarray, bytes, bytearray]) -> bool:
        """
        최신 프레임 제출. 큐가 가득 차 있으면 오래된 프레임을 버리고 교체.
        - BGR/BGRA 경로: frame.shape == (H,W,3|4), dtype=uint8
        - UYVY 경로: frame.shape == (H, W*2) 또는 bytes, dtype=uint8
        """
        if self.proc is None or self.proc.poll() is not None:
            return False
        try:
            buf = frame if isinstance(frame, (bytes, bytearray)) else frame.tobytes()
            if self.q.full():
                try:
                    self.q.get_nowait()  # 오래된 프레임 드롭
                except Exception:
                    pass
            self.q.put_nowait(buf)
            return True
        except Exception as e:
            print("[DL-OUT] submit failed:", e)
            return False

    # 호환용: 기존 코드가 send()를 호출해도 동작하게 유지
    def send(self, frame: Union[np.ndarray, bytes, bytearray]) -> bool:
        return self.submit(frame)

    def is_alive(self) -> bool:
        return self.proc is not None and (self.proc.poll() is None)

    def close(self):
        """정상 종료 시그널 전송 및 프로세스 정리"""
        if self._stop:
            return
        self._stop = True
        try:
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
        except Exception:
            pass
        # 프로세스 종료 대기 후 강제 종료
        if self.proc:
            try:
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
