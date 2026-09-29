#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module/models/yolo_pose.py
- YOLOv8 Pose 모델 래퍼
  • YoloPose: 동기 추론(간단 테스트/디버그용)
  • PoseWorker: 완전 비동기(온디맨드) 워커, 최신 프레임만 처리
    - select="center"일 때 화면 중앙에 가장 가까운 사람 우선
    - lock_enable=True면 일정 시간(lock_ttl) IoU로 같은 대상 유지(락)
  • draw_pose_bones: 출력단에서 스켈레톤 오버레이

메인 예시:
    from module.models.yolo_pose import PoseWorker, draw_pose_bones, YoloPoseConfig
    pw = PoseWorker(YoloPoseConfig(model_name="yolov8n-pose.pt", select="center")).start()
    ...
    pw.need_once(seconds=0.5)     # 필요할 때만 추론 허용 (HUD on 등)
    pw.submit(frame_bgr)          # 최신 프레임 공급
    st = pw.state                 # 최신 상태 (논블로킹)
    if st["ok"]:
        frame_bgr = draw_pose_bones(frame_bgr, st["kps"], alpha_u8=alpha_u8)
    ...
    pw.stop()
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import threading, queue
import torch

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _YOLO_IMPORT_ERROR = e
else:
    _YOLO_IMPORT_ERROR = None

__all__ = [
    "YoloPoseConfig",
    "YoloPose",       # 동기 추론
    "PoseWorker",     # 비동기 워커(온디맨드)
    "draw_pose_bones" # 스켈레톤 드로잉 유틸
]

# -------------------- 유틸 --------------------

def _resolve_device(dev: str | int) -> Any:
    if isinstance(dev, int):
        return dev
    if dev == "cuda" and torch.cuda.is_available():
        return 0
    return "cpu"

def _check_ultralytics():
    if YOLO is None:
        raise ImportError(f"[YOLO] ultralytics import 실패: {_YOLO_IMPORT_ERROR}")

def _iou_xyxy(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (float(a[2])-float(a[0])) * (float(a[3])-float(a[1])))
    area_b = max(0.0, (float(b[2])-float(b[0])) * (float(b[3])-float(b[1])))
    union = area_a + area_b - inter
    return (inter / union) if union > 0.0 else 0.0

class _LatestFrameQueue(queue.Queue):
    """항상 최신 프레임 1개만 유지 (지연 누수 방지)"""
    def put(self, item):
        with self.mutex:
            self.queue.clear()
        super().put(item)

# -------------------- 설정 --------------------

@dataclass
class YoloPoseConfig:
    device: str = "cuda"              # "cuda" | "cpu" | gpu index(int) 허용
    model_name: str = "yolov8n-pose.pt"
    imgsz: int = 416                  # 내부 추론 해상도(긴 변)
    conf: float = 0.25
    iou: float = 0.50
    max_det: int = 1                  # 단일 인물 우선
    use_fp16: bool = True             # CUDA 시 half 사용
    fuse: bool = True                 # 모델 fuse
    fps_cap: float = 12.0             # 비동기 워커 최대 추론 빈도
    ema_alpha: float = 0.7            # 키포인트 EMA 스무딩 계수(0..1, 클수록 이전값 가중)

    # 선택/락 옵션
    select: str = "center"            # "largest" | "center"
    lock_enable: bool = True          # 대상 고정
    lock_ttl: float = 0.6             # 락 유지 시간(초)
    lock_iou: float = 0.30            # 같은 대상으로 간주할 최소 IoU

# -------------------- 동기 추론 --------------------

class YoloPose:
    """
    간단 동기 추론 래퍼 (테스트/디버그용).
    - infer(frame_bgr) -> dict {ok, kps:(1,K,2) or None, bbox: (4,) or None, vis: (K,) or None}
    """
    def __init__(self, cfg: YoloPoseConfig = YoloPoseConfig()):
        _check_ultralytics()
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model = YOLO(cfg.model_name)
        if cfg.fuse and hasattr(self.model, "fuse"):
            try: self.model.fuse()
            except Exception: pass

        self.kw = dict(conf=cfg.conf, iou=cfg.iou, max_det=cfg.max_det,
                       verbose=False, device=self.device,
                       half=(cfg.use_fp16 and torch.cuda.is_available()))
        print(f"[YOLO-Pose(sync)] Loaded '{cfg.model_name}' on {self.device}, imgsz={cfg.imgsz}")

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> Dict[str,Any]:
        H0, W0 = frame_bgr.shape[:2]
        scale = float(self.cfg.imgsz) / float(max(H0, W0))
        inp = cv2.resize(frame_bgr, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_AREA)

        res = self.model.predict(inp, imgsz=self.cfg.imgsz, **self.kw)
        ok = (len(res) and getattr(res[0], "keypoints", None) is not None and len(res[0].keypoints) > 0)
        if not ok:
            return {"ok": False, "kps": None, "bbox": None, "vis": None}

        kps_xy = res[0].keypoints.xy.cpu().numpy()        # [N,K,2]
        boxes  = res[0].boxes.xyxy.cpu().numpy() if len(res[0].boxes) else np.empty((0,4), dtype=np.float32)
        # 선택 기준
        idx = 0
        if boxes.shape[0] > 1:
            if self.cfg.select == "center":
                cx = (boxes[:,0] + boxes[:,2]) * 0.5
                cy = (boxes[:,1] + boxes[:,3]) * 0.5
                Wi, Hi = inp.shape[1], inp.shape[0]
                dx = cx - (Wi * 0.5); dy = cy - (Hi * 0.5)
                idx = int(np.argmin(dx*dx + dy*dy))
            else:
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                idx = int(np.argmax(areas))

        sx, sy = W0/inp.shape[1], H0/inp.shape[0]
        person_kps = kps_xy[idx:idx+1]
        person_kps[...,0] *= sx; person_kps[...,1] *= sy
        vis = res[0].keypoints.conf[idx].cpu().numpy() if res[0].keypoints.conf is not None else None

        return {"ok": True, "kps": person_kps, "bbox": (boxes[idx] if boxes.shape[0] > 0 else None), "vis": vis}

# -------------------- 비동기 워커 (온디맨드 + 중앙고정 락) --------------------

class PoseWorker:
    """
    완전 비동기 YOLOv8 Pose 워커 (스레드)
    - 최신 프레임만 유지, 메인 루프 블로킹 없음
    - 필요시에만 추론: need_once(seconds)로 '허용 창'을 열어두는 방식
    - 중앙 우선(select='center') + 락(lock_iou/lock_ttl)로 대상 유지
    - 공유 상태(self._state)는 최신값만 유지

    인터페이스:
      start() / stop()
      submit(frame_bgr)                 : 최신 프레임 공급 (항상 즉시 반환)
      need_once(seconds=0.0)            : 다음 1회 또는 지정 초 동안 추론 허용
      state (property) -> dict          : {ok, kps, bbox, vis, ts}
    """
    def __init__(self, cfg: YoloPoseConfig = YoloPoseConfig()):
        _check_ultralytics()
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)

        self._q = _LatestFrameQueue(maxsize=1)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

        self._model = YOLO(cfg.model_name)
        if cfg.fuse and hasattr(self._model, "fuse"):
            try: self._model.fuse()
            except Exception: pass

        self._kwargs = dict(conf=cfg.conf, iou=cfg.iou, max_det=cfg.max_det,
                            verbose=False, device=self.device,
                            half=(cfg.use_fp16 and torch.cuda.is_available()))
        self._imgsz = int(cfg.imgsz)
        self._min_interval = 1.0 / max(cfg.fps_cap, 1.0)
        self._needed_until = 0.0
        self._last_infer   = 0.0

        self._ema = None
        self._alpha = float(cfg.ema_alpha)

        # 락 상태
        self._lock_box = None
        self._lock_ts  = 0.0

        # 공유 상태
        self._state: Dict[str,Any] = {"ts":0.0, "ok":False, "kps":None, "bbox":None, "vis":None}

        print(f"[YOLO-Pose(async)] Loaded '{cfg.model_name}' on {self.device}, "
              f"imgsz={self._imgsz}, fps_cap={cfg.fps_cap}, on-demand mode, select={cfg.select}")

    # ---- 외부 API ----
    def start(self):
        if self._th is not None:
            return self
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def stop(self, timeout: float = 1.0):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=timeout)
            self._th = None

    def submit(self, frame_bgr: np.ndarray):
        if frame_bgr is not None:
            self._q.put(frame_bgr)

    def need_once(self, seconds: float = 0.0):
        """다음 1회 또는 지정 초 동안 추론 허용(허용 창 갱신)"""
        now = time.time()
        self._needed_until = max(self._needed_until, now + max(0.0, seconds))

    @property
    def state(self) -> Dict[str,Any]:
        return self._state

    # ---- 내부 루프 ----
    def _should_infer(self) -> bool:
        now = time.time()
        if (now - self._last_infer) < self._min_interval:
            return False
        return now <= self._needed_until

    @torch.inference_mode()
    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.002); continue

            if not self._should_infer():
                # 허용 창이 아니면 스킵 (상태 유지)
                continue

            H0, W0 = frame.shape[:2]
            scale = float(self._imgsz) / float(max(H0, W0))
            inp = cv2.resize(frame, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_AREA)

            res = self._model.predict(inp, imgsz=self._imgsz, **self._kwargs)
            self._last_infer = time.time()

            ok = (len(res) and getattr(res[0], "keypoints", None) is not None and len(res[0].keypoints) > 0)
            if ok:
                kps_xy = res[0].keypoints.xy.cpu().numpy()        # [N,K,2]

                if len(res[0].boxes):
                    boxes = res[0].boxes.xyxy.cpu().numpy()
                else:
                    boxes = np.empty((0,4), dtype=np.float32)

                idx = 0
                if boxes.shape[0] > 0:
                    now = time.time()
                    use_center = (self.cfg.select == "center")

                    # 1) 락 유지 시도
                    if self.cfg.lock_enable and self._lock_box is not None and (now - self._lock_ts) <= self.cfg.lock_ttl:
                        ious = np.array([_iou_xyxy(self._lock_box, b) for b in boxes], dtype=np.float32)
                        j = int(np.argmax(ious))
                        if ious[j] >= self.cfg.lock_iou:
                            idx = j
                        else:
                            # 2) 중앙 우선 or 최대 박스
                            if use_center:
                                cx = (boxes[:,0] + boxes[:,2]) * 0.5
                                cy = (boxes[:,1] + boxes[:,3]) * 0.5
                                Wi, Hi = inp.shape[1], inp.shape[0]
                                dx = cx - (Wi * 0.5); dy = cy - (Hi * 0.5)
                                d2 = dx*dx + dy*dy
                                idx = int(np.argmin(d2))
                            else:
                                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                                idx = int(np.argmax(areas))
                    else:
                        # 락 없음 → 중앙/최대
                        if use_center:
                            cx = (boxes[:,0] + boxes[:,2]) * 0.5
                            cy = (boxes[:,1] + boxes[:,3]) * 0.5
                            dx = cx - (W0 * 0.5); dy = cy - (H0 * 0.5)
                            d2 = dx*dx + dy*dy
                            idx = int(np.argmin(d2))
                        else:
                            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                            idx = int(np.argmax(areas))

                    # --- 좌표 스케일 복원 + EMA ---
                    sx, sy = W0/inp.shape[1], H0/inp.shape[0]
                    person_kps = kps_xy[idx:idx+1]
                    person_kps[...,0] *= sx; person_kps[...,1] *= sy

                    if self._ema is None: self._ema = person_kps
                    else: self._ema = self._alpha*self._ema + (1.0-self._alpha)*person_kps

                    vis = res[0].keypoints.conf[idx].cpu().numpy() if res[0].keypoints.conf is not None else None

                    # --- 락 갱신 ---
                    self._lock_box = boxes[idx].copy()
                    self._lock_ts  = now

                    self._state = {
                        "ts": time.time(), "ok": True,
                        "kps": self._ema.copy(), "bbox": boxes[idx], "vis": vis
                    }
                else:
                    # keypoints는 있으나 boxes 없음(희박) → 방어
                    self._state = {"ts": time.time(), "ok": False, "kps": None, "bbox": None, "vis": None}
                    self._lock_box = None
            else:
                self._state = {"ts": time.time(), "ok": False, "kps": None, "bbox": None, "vis": None}
                self._lock_box = None

# -------------------- 스켈레톤 드로잉 유틸 --------------------

# YOLOv8(17 keypoints, COCO 순서)용 기본 연결쌍
_COCO_PAIRS = [
    (5,7),(7,9), (6,8),(8,10), (5,6),
    (11,13),(13,15), (12,14),(14,16), (11,12),
    (5,11),(6,12), (0,5),(0,6)
]

def draw_pose_bones(img_bgr: np.ndarray,
                    kps: Optional[np.ndarray],
                    alpha_u8: Optional[np.ndarray] = None,
                    label: bool = True) -> np.ndarray:
    """
    kps: (1,K,2) float xy, alpha_u8: 전경 알파(0..255, 선택)
    출력단(미리 합성 끝난 프레임)에 호출하여 스켈레톤 확인.
    """
    if kps is None or len(kps) == 0:
        if label:
            cv2.putText(img_bgr, "POSE: none", (16,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        return img_bgr

    pts = kps[0]
    H, W = img_bgr.shape[:2]

    # 라인
    for a,b in _COCO_PAIRS:
        if a < len(pts) and b < len(pts):
            p1 = tuple(np.clip(np.int32(pts[a]), [0,0], [W-1,H-1]))
            p2 = tuple(np.clip(np.int32(pts[b]), [0,0], [W-1,H-1]))
            if alpha_u8 is not None:
                if not (alpha_u8[p1[1], p1[0]]>127 or alpha_u8[p2[1], p2[0]]>127):
                    continue
            cv2.line(img_bgr, p1, p2, (0,255,0), 2, cv2.LINE_AA)
    # 점
    for p in pts:
        x,y = map(int, p[:2])
        x = min(max(x,0), W-1); y = min(max(y,0), H-1)
        if alpha_u8 is None or alpha_u8[y, x] > 127:
            cv2.circle(img_bgr, (x,y), 3, (255,0,0), -1, cv2.LINE_AA)

    if label:
        cv2.putText(img_bgr, "AI POSE (YOLOv8)", (16,36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    return img_bgr
