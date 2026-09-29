#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module/effects/hand_trail.py
- YOLOv8 Pose의 손목 키포인트(좌/우)를 받아 '손을 따라가는 라인' 잔상 이펙트를 그립니다.
- 입력은 numpy BGR 프레임 기준(필요 시 torch 텐서도 허용)으로 설계.
- 옵션:
  • which: "auto" | "left" | "right"  (auto는 가시도 높은 손 선택)
  • max_len: 잔상 길이(포인트 개수)
  • thickness: 기준 선 두께
  • color: (B,G,R)
  • fade: 뒤로 갈수록 선을 얇고 어둡게
  • only_foreground: 알파 마스크(전경) 위에서만 그리기
  • alpha_thr/dilate_px: 전경 마스크 기준
  • smooth: 간단 1차 저역통과(EMA)로 흔들림 줄이기 (0=끄기)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import cv2
import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]

# COCO keypoints index (YOLOv8): 17개
# 0-nose, 5-left_shoulder, 6-right_shoulder, 9-left_wrist, 10-right_wrist ...
LEFT_WRIST  = 9
RIGHT_WRIST = 10


@dataclass
class HandTrailConfig:
    which: str = "auto"                 # "auto" | "left" | "right"
    max_len: int = 60
    thickness: int = 4
    color: Tuple[int,int,int] = (0,255,255)  # BGR
    fade: bool = True
    only_foreground: bool = False       # True면 전경 알파 안쪽에서만 그림
    alpha_thr: int = 127
    dilate_px: int = 1
    smooth: float = 0.0                 # 0=off, (0,1] EMA 강화(값이 클수록 더 천천히 이동)


def _alpha_to_mask(alpha_like: ArrayLike, thr: int, dilate_px: int) -> Optional[np.ndarray]:
    """
    alpha_like: [H,W] (np.uint8 0..255 또는 torch 텐서)
    반환: np.uint8 [H,W] {0,255} (전경=255), 없으면 None
    """
    if alpha_like is None:
        return None
    if isinstance(alpha_like, torch.Tensor):
        a = alpha_like
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.dtype != torch.uint8:
            # 0..1 float일 가능성
            a = (a > (thr/255.0)).to(torch.uint8) * 255
        else:
            a = (a > thr).to(torch.uint8) * 255
        a = a.cpu().numpy()
    else:
        a = np.asarray(alpha_like)
        if a.dtype != np.uint8:
            # float 가정
            a = (a > (thr/255.0)).astype(np.uint8) * 255
        else:
            a = (a > thr).astype(np.uint8) * 255
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        a = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k)))
    return a


class HandTrailEffect:
    def __init__(self, cfg: HandTrailConfig = HandTrailConfig()):
        self.cfg = cfg
        self._pts: list[Tuple[int,int]] = []
        self._ema_pt: Optional[np.ndarray] = None  # EMA된 좌표 (2,)

    def reset(self):
        self._pts.clear()
        self._ema_pt = None

    def _pick_wrist(self, kps1: np.ndarray, vis: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        kps1: (K,2) numpy float
        vis : (K,) or None (가시도/conf)
        반환: 선택 손목 좌표 (2,) or None
        """
        lw_ok = (LEFT_WRIST  < len(kps1))
        rw_ok = (RIGHT_WRIST < len(kps1))
        if not lw_ok and not rw_ok:
            return None

        if self.cfg.which == "left" and lw_ok:
            return kps1[LEFT_WRIST]
        if self.cfg.which == "right" and rw_ok:
            return kps1[RIGHT_WRIST]

        # auto: 가시도 높은 손 선택 → 없으면 오른손 우선
        if vis is not None and len(vis) > max(LEFT_WRIST, RIGHT_WRIST):
            lv = float(vis[LEFT_WRIST])  if lw_ok else -1.0
            rv = float(vis[RIGHT_WRIST]) if rw_ok else -1.0
            if lv >= rv and lw_ok:
                return kps1[LEFT_WRIST]
            if rw_ok:
                return kps1[RIGHT_WRIST]
        # 가시도 정보 없으면 오른손 우선, 없으면 왼손
        if rw_ok:
            return kps1[RIGHT_WRIST]
        if lw_ok:
            return kps1[LEFT_WRIST]
        return None

    def update(self, kps: Optional[np.ndarray], vis: Optional[np.ndarray] = None):
        """
        kps: (1,K,2) 또는 (K,2) numpy float
        vis: (K,) or None
        """
        if kps is None:
            return
        kps1 = kps[0] if (kps.ndim == 3 and kps.shape[0] == 1) else kps
        pt = self._pick_wrist(kps1, vis)
        if pt is None:
            return

        # 간단 스무딩(EMA)
        if self.cfg.smooth and self.cfg.smooth > 0.0:
            if self._ema_pt is None:
                self._ema_pt = np.asarray(pt, dtype=np.float32)
            else:
                a = float(self.cfg.smooth)  # 0..1
                self._ema_pt = a*self._ema_pt + (1.0-a)*np.asarray(pt, dtype=np.float32)
            pt_use = self._ema_pt
        else:
            pt_use = np.asarray(pt, dtype=np.float32)

        x, y = int(round(float(pt_use[0]))), int(round(float(pt_use[1])))
        self._pts.append((x, y))
        if len(self._pts) > self.cfg.max_len:
            self._pts = self._pts[-self.cfg.max_len:]

    def draw(self, frame_bgr: ArrayLike, alpha_u8: Optional[ArrayLike] = None) -> ArrayLike:
        """
        frame_bgr: np.ndarray (H,W,3) 또는 torch CHW/HWC
        alpha_u8 : 전경 마스크(선택)
        """
        # torch도 허용하되, 편의상 numpy로 변환하여 그림
        was_torch = False
        if isinstance(frame_bgr, torch.Tensor):
            was_torch = True
            t = frame_bgr
            if t.ndim == 3 and t.shape[0] == 3:  # CHW
                img = (t.permute(1,2,0).contiguous().detach().cpu().numpy())
            elif t.ndim == 3 and t.shape[2] == 3:  # HWC
                img = t.detach().cpu().numpy()
            else:
                raise ValueError("Unsupported torch image shape.")
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 1)
                img = (img*255.0).round().astype(np.uint8)
        else:
            img = np.asarray(frame_bgr)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 1)
                img = (img*255.0).round().astype(np.uint8)

        if len(self._pts) >= 2:
            H, W = img.shape[:2]
            mask_fg = None
            if self.cfg.only_foreground and alpha_u8 is not None:
                mask_fg = _alpha_to_mask(alpha_u8, self.cfg.alpha_thr, self.cfg.dilate_px)

            n = len(self._pts)
            for i in range(1, n):
                p1 = self._pts[i-1]; p2 = self._pts[i]
                # 화면 경계 내 점만 허용
                if not (0 <= p1[0] < W and 0 <= p1[1] < H and 0 <= p2[0] < W and 0 <= p2[1] < H):
                    continue
                if self.cfg.fade:
                    t = i / (n - 1 + 1e-6)  # 0..1
                    thick = max(1, int(self.cfg.thickness * (0.4 + 0.6*t)))
                    c = tuple(int(v * (0.25 + 0.75*t)) for v in self.cfg.color)
                else:
                    thick = self.cfg.thickness
                    c = self.cfg.color

                if mask_fg is not None:
                    # 선분 샘플링하여 전경 픽셀만 그리기(간단 샘플 방식)
                    num_samp = max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1])) + 1
                    xs = np.linspace(p1[0], p2[0], num_samp).astype(np.int32)
                    ys = np.linspace(p1[1], p2[1], num_samp).astype(np.int32)
                    valid = (mask_fg[ys, xs] > 0)
                    if not np.any(valid):
                        continue
                    # 전경 구간만 분할 그리기
                    seg_start = None
                    for j in range(num_samp):
                        if valid[j] and seg_start is None:
                            seg_start = j
                        if (not valid[j] or j == num_samp-1) and seg_start is not None:
                            j2 = j if not valid[j] else j  # 포함 종료
                            s1 = (int(xs[seg_start]), int(ys[seg_start]))
                            s2 = (int(xs[j2-1]),     int(ys[j2-1])) if j2>seg_start else s1
                            cv2.line(img, s1, s2, c, thick, cv2.LINE_AA)
                            seg_start = None
                else:
                    cv2.line(img, p1, p2, c, thick, cv2.LINE_AA)

        if was_torch:
            # torch CHW로 되돌려 주는 편이 안전
            out = torch.from_numpy(img).to(torch.uint8).permute(2,0,1)
            # 호출자가 float[0..1]을 기대한다면 여기서 후처리 가능
            return out
        return img
