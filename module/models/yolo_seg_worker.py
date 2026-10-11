#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module/models/yolo_seg_worker.py
- YOLOv8-seg 비동기 워커: 최신 프레임만 처리해 사람 인스턴스 마스크를 만든다.
- 마스크는 masks.xyn(정규화 폴리곤)을 작업 해상도(work_w x work_h)에 래스터라이즈한다.
  (masks.data는 레터박스 패딩이 포함된 모델 입력 좌표라 원본과 정렬이 어긋날 수 있음)

state (property) -> dict:
    ts:    결과 생성 시각(time.time())
    masks: np.ndarray [N, work_h, work_w] uint8(0/1) 또는 None
    boxes: np.ndarray [N, 4] xyxy (원본 픽셀 좌표) 또는 None
"""

from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _YOLO_IMPORT_ERROR = e
else:
    _YOLO_IMPORT_ERROR = None


@dataclass
class YoloSegConfig:
    model_name: str = "yolov8n-seg.pt"
    imgsz: int = 640
    conf: float = 0.4
    iou: float = 0.5
    max_det: int = 6
    device: Any = 0
    half: bool = True
    fps_cap: float = 30.0
    work_w: int = 480                  # 마스크 작업 해상도(가로). 세로는 입력 비율로 결정
    min_area_frac: float = 0.002       # 화면 대비 최소 마스크 면적


class YoloSegWorker:
    """완전 비동기 YOLO-seg 워커. submit()으로 최신 프레임을 넣고 state로 결과를 읽는다."""

    def __init__(self, cfg: YoloSegConfig):
        if YOLO is None:
            raise ImportError(f"[YOLO-seg] ultralytics import 실패: {_YOLO_IMPORT_ERROR}")
        self.cfg = cfg
        self.model = YOLO(cfg.model_name)
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {"ts": 0.0, "masks": None, "boxes": None}
        self._alive = False
        self._thread: Optional[threading.Thread] = None
        self._min_dt = 1.0 / float(cfg.fps_cap) if cfg.fps_cap > 0 else 0.0
        print(f"[INFO] YOLO-seg worker: model={cfg.model_name}, imgsz={cfg.imgsz}, "
              f"conf={cfg.conf}, work_w={cfg.work_w}")

    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self._alive = True
        self._thread = threading.Thread(target=self._run, name="YoloSegWorker", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._alive = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(self, frame_bgr: np.ndarray):
        if not self._alive or frame_bgr is None:
            return
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait(frame_bgr)
        except queue.Full:
            pass

    @property
    def state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        H, W = frame_bgr.shape[:2]
        ww = int(self.cfg.work_w)
        wh = max(1, int(round(ww * H / float(W))))
        res = self.model.predict(
            source=frame_bgr, imgsz=int(self.cfg.imgsz), conf=float(self.cfg.conf),
            iou=float(self.cfg.iou), max_det=int(self.cfg.max_det), classes=[0],
            device=self.cfg.device, half=bool(self.cfg.half), verbose=False,
        )
        if not res or res[0].masks is None or res[0].boxes is None or len(res[0].boxes) == 0:
            return {"ts": time.time(), "masks": None, "boxes": None}

        r = res[0]
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        min_area = float(self.cfg.min_area_frac) * ww * wh
        masks, keep_boxes = [], []
        for i, poly in enumerate(r.masks.xyn):
            if poly is None or len(poly) < 3:
                continue
            m = np.zeros((wh, ww), np.uint8)
            pts = np.round(poly * np.array([ww, wh], np.float32)).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
            if float(m.sum()) < min_area:
                continue
            masks.append(m)
            keep_boxes.append(boxes[i])
        if not masks:
            return {"ts": time.time(), "masks": None, "boxes": None}
        return {"ts": time.time(), "masks": np.stack(masks, 0), "boxes": np.stack(keep_boxes, 0)}

    def _run(self):
        last = 0.0
        while self._alive:
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            wait = self._min_dt - (time.perf_counter() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.perf_counter()
            try:
                st = self._infer(frame)
            except Exception as e:
                print(f"[WARN] YOLO-seg 추론 실패: {e}")
                st = {"ts": time.time(), "masks": None, "boxes": None}
            with self._lock:
                self._state = st
