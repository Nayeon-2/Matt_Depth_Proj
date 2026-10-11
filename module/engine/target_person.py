#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module/engine/target_person.py
- YOLO-seg 인스턴스 중 효과를 줄 "타겟 한 명"을 고르고, 그 사람의 소프트 가중치 맵을 만든다.
- RVM 알파(경계 품질) × 타겟 가중치(누구인지) = 타겟 알파.

선택 모드:
    left    : 화면 가장 왼쪽 사람
    right   : 화면 가장 오른쪽 사람
    center  : 화면 중앙에 가장 가까운 사람
    largest : 마스크 면적이 가장 큰 사람
    nearest : 발(bbox 아래)이 화면 아래쪽에 있는 사람 = 카메라에 가까운 사람

한 번 고른 타겟은 IoU/중심거리로 계속 추적(락)하고, lock_ttl 초 동안 못 찾으면 모드 기준으로 다시 고른다.
cycle()은 다음 사람(왼→오른쪽 순환)으로 타겟을 넘기고, reset()은 락을 풀어 모드 기준으로 다시 고른다.
"""

from __future__ import annotations
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("left", "right", "center", "largest", "nearest")


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _blur(x: torch.Tensor, r: int) -> torch.Tensor:
    """separable box blur 2회 ≈ 가우시안. x: [N,1,h,w]"""
    if r <= 0:
        return x
    k = 2 * r + 1
    for _ in range(2):
        x = F.avg_pool2d(F.pad(x, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
        x = F.avg_pool2d(F.pad(x, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)
    return x


class TargetPersonSelector:
    def __init__(self, W: int, H: int, *, mode: str = "left", lock_ttl_s: float = 1.0,
                 grow_px: int = 40, feather_px: int = 6, device="cuda"):
        if mode not in MODES:
            raise ValueError(f"target mode must be one of {MODES}: {mode}")
        self.W, self.H = int(W), int(H)
        self.mode = mode
        self.lock_ttl_s = float(lock_ttl_s)
        self.grow_px = int(grow_px)
        self.feather_px = int(feather_px)
        self.device = device

        self._last_ts = 0.0
        self._boxes: Optional[np.ndarray] = None    # 최신 인스턴스 bbox들 [N,4]
        self._target_box: Optional[np.ndarray] = None
        self._target_seen_ts = 0.0
        self._weight: Optional[torch.Tensor] = None  # [1,H,W] float 0..1
        self._pending_cycle = False

    # ---------- 외부 제어 ----------
    def cycle(self):
        self._pending_cycle = True

    def reset(self):
        self._target_box = None
        self._weight = None

    @property
    def target_box(self) -> Optional[np.ndarray]:
        return None if self._weight is None else self._target_box

    # ---------- 선택 ----------
    def _pick_by_mode(self, boxes: np.ndarray, areas: np.ndarray) -> int:
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
        if self.mode == "left":
            return int(np.argmin(cx))
        if self.mode == "right":
            return int(np.argmax(cx))
        if self.mode == "center":
            return int(np.argmin(np.abs(cx - 0.5 * self.W)))
        if self.mode == "largest":
            return int(np.argmax(areas))
        return int(np.argmax(boxes[:, 3]))  # nearest

    def _match_locked(self, boxes: np.ndarray) -> int:
        tb = self._target_box
        ious = np.array([_iou(tb, b) for b in boxes])
        best = int(np.argmax(ious))
        if ious[best] >= 0.2:
            return best
        # 빠르게 움직여 IoU가 낮으면 중심거리로 매칭(타겟 bbox 폭 이내)
        tcx, tcy = 0.5 * (tb[0] + tb[2]), 0.5 * (tb[1] + tb[3])
        d = np.hypot(0.5 * (boxes[:, 0] + boxes[:, 2]) - tcx, 0.5 * (boxes[:, 1] + boxes[:, 3]) - tcy)
        best = int(np.argmin(d))
        return best if d[best] <= max(1.0, tb[2] - tb[0]) else -1

    def _choose(self, boxes: np.ndarray, areas: np.ndarray, now: float) -> int:
        if self._pending_cycle and self._target_box is not None:
            self._pending_cycle = False
            order = np.argsort(0.5 * (boxes[:, 0] + boxes[:, 2]))
            cur = self._match_locked(boxes)
            if cur < 0 or len(order) == 1:
                return int(order[0])
            pos = int(np.where(order == cur)[0][0])
            return int(order[(pos + 1) % len(order)])
        self._pending_cycle = False

        if self._target_box is not None:
            idx = self._match_locked(boxes)
            if idx >= 0:
                return idx
            if now - self._target_seen_ts < self.lock_ttl_s:
                return -1  # 잠깐 놓침: 이전 가중치 유지
        return self._pick_by_mode(boxes, areas)

    # ---------- 업데이트 ----------
    @torch.no_grad()
    def update(self, seg_state: dict) -> Optional[torch.Tensor]:
        """YoloSegWorker.state를 받아 타겟 가중치 맵 [1,H,W](0..1) 또는 None을 반환."""
        now = time.time()
        ts = float(seg_state.get("ts", 0.0))
        if ts <= self._last_ts:
            return self._expire(now)
        self._last_ts = ts

        masks, boxes = seg_state.get("masks"), seg_state.get("boxes")
        if masks is None or boxes is None or len(boxes) == 0:
            self._boxes = None
            return self._expire(now)
        self._boxes = boxes

        areas = masks.reshape(masks.shape[0], -1).sum(1).astype(np.float32)
        idx = self._choose(boxes, areas, now)
        if idx < 0:
            return self._expire(now)

        self._target_box = boxes[idx].copy()
        self._target_seen_ts = now
        self._weight = self._build_weight(masks, idx)
        return self._weight

    def _expire(self, now: float) -> Optional[torch.Tensor]:
        if self._weight is not None and now - self._target_seen_ts > self.lock_ttl_s:
            self._weight = None
            self._target_box = None
        return self._weight

    def _build_weight(self, masks: np.ndarray, idx: int) -> torch.Tensor:
        N, h, w = masks.shape
        scale = w / float(self.W)                      # 작업 해상도 / 원본
        grow = max(1, int(round(self.grow_px * scale)))
        feather = max(1, int(round(self.feather_px * scale)))

        m = torch.from_numpy(masks).to(self.device, non_blocking=True).float().unsqueeze(1)  # [N,1,h,w]
        soft = _blur(m, feather)
        grown = _blur(F.max_pool2d(m, 2 * grow + 1, stride=1, padding=grow), feather)

        # 사람 몸 안쪽은 원본 마스크가 우선, 머리카락처럼 마스크 바깥 RVM 알파는 확장 마스크로 귀속
        k = 0.05
        score = soft + k * grown
        wt = score[idx] / (score.sum(0) + 1e-3)        # [1,h,w]
        wt = ((wt - 0.3) / 0.4).clamp(0, 1)            # 경쟁 구간을 좁게
        wt = wt * wt * (3 - 2 * wt)                    # smoothstep
        wt = F.interpolate(wt.unsqueeze(0), size=(self.H, self.W), mode="bilinear", align_corners=False)[0]
        return wt.clamp_(0, 1)
