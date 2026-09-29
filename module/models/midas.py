#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module/models/midas.py
- MiDaS Depth 모델 래퍼
  • MidasDepth: 동기 추론(기존 인터페이스와 호환)
  • DepthWorker: 완전 비동기(멀티프로세스+공유메모리) 워커, DepthProcess를 래핑
- 메인 예시:
    from module.models.midas import MidasDepth, DepthWorker
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import torch
import torch.nn.functional as F

__all__ = [
    "MidasDepthConfig",
    "MidasDepth",          # 동기 추론용 (메인 호환)
    "DepthWorker",         # 비동기 워커(메인 호환)
    "DepthProcess",        # 저수준 프로세스 워커
]

# -------------------- 설정 --------------------

@dataclass
class MidasDepthConfig:
    device: str = "cuda"
    model_name: str = "MiDaS_small"       # "MiDaS_small" | "DPT_Large" | "DPT_Hybrid" ...
    work_size: Tuple[int,int] = (384,384) # (Ww,Hw) 워커 내부 추론 크기
    out_size:  Tuple[int,int] = (720,1280)# (H,W) 결과 해상도(메인 프레임과 동일 권장)
    max_fps: float = 8.0                  # 워커 추론 최대 FPS
    contrast_clip: Tuple[float,float] = (2.0, 98.0)

# -------------------- (내부) 모델 구현 --------------------

class _MidasModel:
    def __init__(self, cfg: MidasDepthConfig):
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("intel-isl/MiDaS", cfg.model_name, trust_repo=True).to(self.device).eval()
        tx = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if cfg.model_name.lower() in ("midas_small", "midas_v21_small", "midas_small_256", "midas_small_384"):
            self.transform = tx.small_transform
        else:
            self.transform = tx.dpt_transform

        Ww, Hw = cfg.work_size
        self._rgb_cpu = torch.empty((3, Hw, Ww), dtype=torch.float32)
        self._rgb_pinned = torch.empty((3, Hw, Ww), dtype=torch.float32, pin_memory=True) \
                           if self.device.type == "cuda" else None
        print(f"[MiDaS] Loaded '{cfg.model_name}' on {self.device}, work_size={cfg.work_size}, out_size={cfg.out_size}")

    @torch.inference_mode()
    def infer_fixed(self, bgr_np: np.ndarray) -> np.ndarray:
        """
        고정 출력 크기(self.cfg.out_size)에 맞춰 depth를 생성 (비동기 워커에서 사용)
        """
        H_out, W_out = self.cfg.out_size
        Ww, Hw = self.cfg.work_size

        rgb = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (Ww, Hw), interpolation=cv2.INTER_AREA)

        inp = torch.from_numpy(rgb).permute(2,0,1).float().div_(255.0)  # (3,Hw,Ww)
        if self._rgb_pinned is not None:
            self._rgb_pinned.copy_(inp, non_blocking=False)
            inp = self._rgb_pinned
        else:
            self._rgb_cpu.copy_(inp)
            inp = self._rgb_cpu
        inp = inp.unsqueeze(0)

        if self.device.type == "cuda":
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream), torch.amp.autocast("cuda", dtype=torch.float16):
                pred = self.model(inp.to(self.device, non_blocking=True))
                pred = F.interpolate(pred.unsqueeze(1), size=(H_out, W_out),
                                     mode="bilinear", align_corners=False).squeeze(1)
                pred = pred[0].detach().float().cpu()
        else:
            pred = self.model(inp)
            pred = F.interpolate(pred.unsqueeze(1), size=(H_out, W_out),
                                 mode="bilinear", align_corners=False).squeeze(1)
            pred = pred[0].detach().float()

        d = pred.numpy()
        lo, hi = np.percentile(d, self.cfg.contrast_clip[0]), np.percentile(d, self.cfg.contrast_clip[1])
        d = np.clip((d - lo) / max(1e-6, (hi - lo)), 0.0, 1.0).astype(np.float32)
        return d

# -------------------- 동기 추론(호환용): MidasDepth --------------------

class MidasDepth:
    """
    기존 메인 코드와 호환되는 동기 추론 래퍼.
    - 생성자: (device="cuda", model_name="MiDaS_small", max_long=384)
    - infer(bgr_np) -> depth [0..1], shape=(H,W)
      * 입력 긴 변을 max_long으로 리사이즈해 추론 후 원본 크기로 보간
    """
    def __init__(self, device: str = "cuda", model_name: str = "MiDaS_small", max_long: int = 384):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_long = int(max_long)
        torch.backends.cudnn.benchmark = True

        self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True).to(self.device).eval()
        tx = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = tx.small_transform if model_name.lower() in ("midas_small", "midas_v21_small") else tx.dpt_transform
        print(f"[INFO] MiDaS (sync) loaded: {model_name} on {self.device}, max_long={self.max_long}")

    @torch.inference_mode()
    def infer(self, bgr_np: np.ndarray) -> np.ndarray:
        H, W = bgr_np.shape[:2]
        scale = min(1.0, float(self.max_long) / float(max(H, W)))
        src = bgr_np if scale >= 1.0 else cv2.resize(bgr_np, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb).to(self.device)

        if self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = self.model(inp)
        else:
            pred = self.model(inp)

        pred = F.interpolate(pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        d = pred[0].detach().float().cpu().numpy()

        d_min, d_max = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - d_min) / max(1e-6, (d_max - d_min)), 0.0, 1.0).astype(np.float32)
        return d

# -------------------- 멀티프로세스 워커 --------------------

class DepthProcess:
    """
    완전 비동기 MiDaS 워커(멀티프로세스 + 공유메모리).
    메인은 submit()으로 프레임을 던지고, get_latest()로 최신 결과를 읽는다.
    """
    def __init__(self, cfg: MidasDepthConfig):
        self.cfg = cfg
        H_out, W_out = cfg.out_size
        self._inH, self._inW = H_out, W_out  # 입력 프레임 크기를 출력과 동일로 가정

        self._slots = 2
        self._in_bytes = self._inH * self._inW * 3
        self._shm_in  = SharedMemory(create=True, size=self._in_bytes * self._slots)
        self._shm_out = SharedMemory(create=True, size=(self._inH * self._inW * 4))  # float32

        self._idx_write = mp.Value('i', 0)
        self._idx_read  = mp.Value('i', -1)
        self._ts_out    = mp.Value('d', 0.0)
        self._alive     = mp.Value('b', True)

        self._proc: Optional[mp.Process] = None

    def start(self):
        self._proc = mp.Process(
            target=_depth_worker_main,
            args=(self.cfg,
                  self._shm_in.name, self._inH, self._inW, self._slots,
                  self._shm_out.name,
                  self._idx_write, self._idx_read, self._ts_out, self._alive),
            daemon=True
        )
        self._proc.start()
        print(f"[MiDaS] DepthProcess started (pid={self._proc.pid})")
        return self

    def submit(self, bgr_np: np.ndarray):
        if not self._alive.value:
            return
        if bgr_np.shape[0] != self._inH or bgr_np.shape[1] != self._inW:
            bgr_np = cv2.resize(bgr_np, (self._inW, self._inH), interpolation=cv2.INTER_AREA)
        w = (self._idx_write.value + 1)
        slot = w % self._slots
        np.frombuffer(self._shm_in.buf, dtype=np.uint8, count=self._in_bytes,
                      offset=(slot * self._in_bytes)).reshape(self._inH, self._inW, 3)[:] = bgr_np
        self._idx_write.value = w

    def get_latest(self, copy: bool = False):
        ts = self._ts_out.value
        if ts <= 0.0:
            return None, 0.0
        arr = np.frombuffer(self._shm_out.buf, dtype=np.float32,
                            count=self._inH*self._inW).reshape(self._inH, self._inW)
        return (arr.copy() if copy else arr), ts

    def stop(self, timeout: float = 1.0):
        if self._alive.value:
            self._alive.value = False
        if self._proc is not None:
            self._proc.join(timeout=timeout)
        try: self._shm_in.close();  self._shm_in.unlink()
        except Exception: pass
        try: self._shm_out.close(); self._shm_out.unlink()
        except Exception: pass
        print("[MiDaS] DepthProcess stopped")

def _depth_worker_main(cfg: MidasDepthConfig,
                       shm_in_name: str, inH: int, inW: int, slots: int,
                       shm_out_name: str,
                       idx_write: mp.Value, idx_read: mp.Value, ts_out: mp.Value, alive: mp.Value):
    shm_in  = SharedMemory(name=shm_in_name)
    shm_out = SharedMemory(name=shm_out_name)
    in_bytes = inH * inW * 3

    model = _MidasModel(cfg)

    interval = (1.0 / cfg.max_fps) if cfg.max_fps > 0 else 0.0
    last_time = 0.0

    try:
        while alive.value:
            w = idx_write.value
            r = idx_read.value
            if w <= r:
                time.sleep(0.002); continue

            now = time.time()
            if interval > 0 and (now - last_time) < interval:
                time.sleep(0.001); continue

            slot = w % slots
            frm = np.frombuffer(shm_in.buf, dtype=np.uint8, count=in_bytes,
                                offset=(slot * in_bytes)).reshape(inH, inW, 3)

            try:
                d = model.infer_fixed(frm)
                np.frombuffer(shm_out.buf, dtype=np.float32,
                              count=inH*inW).reshape(inH, inW)[:] = d
                idx_read.value = w
                ts_out.value = time.time()
                last_time = ts_out.value
            except Exception as e:
                print(f"[MiDaS/worker] infer error: {e}")
                time.sleep(0.005)
    finally:
        try: shm_in.close()
        except Exception: pass
        try: shm_out.close()
        except Exception: pass

# -------------------- 메인 호환 래퍼(비동기): DepthWorker --------------------

class DepthWorker:
    """
    메인 루프가 기대하는 인터페이스로 DepthProcess를 감싼 래퍼.
    - submit(frame)           : 즉시 반환
    - latest (property)       : 최신 depth(np.float32 [0..1]) 또는 None
    - get_latest(copy=False)  : (depth | None, ts)
    - stop()                  : 정리
    """
    def __init__(self, model: MidasDepth,
                 max_fps: float = 10.0,
                 work_size: Tuple[int,int] = (384,384),
                 out_size: Tuple[int,int] = (720,1280),
                 device: str = "cuda"):
        # model은 시그니처 유지용(미사용). config로 실제 워커 생성
        self._H, self._W = out_size
        cfg = MidasDepthConfig(
            device=device,
            model_name="MiDaS_small",   # 동작엔 영향 없음(허브에서 동일 계열 로드)
            work_size=work_size,
            out_size=out_size,
            max_fps=max_fps,
        )
        self._proc = DepthProcess(cfg).start()

    def start(self):  # 메인 호환 (체인 호출 대응)
        return self

    def submit(self, bgr_np: np.ndarray):
        self._proc.submit(bgr_np)

    @property
    def latest(self) -> Optional[np.ndarray]:
        arr, ts = self._proc.get_latest(copy=False)
        return arr

    def get_latest(self, copy: bool = False):
        return self._proc.get_latest(copy=copy)

    def stop(self, timeout: float = 1.0):
        self._proc.stop(timeout=timeout)
