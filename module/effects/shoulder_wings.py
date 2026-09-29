# module/effects/image_wing.py
from __future__ import annotations
import os, math
from typing import Optional, Tuple
import numpy as np
import cv2

# YOLOv8 COCO keypoints index 참고:
# 5=L.shoulder, 7=L.elbow, 9=L.wrist
# 6=R.shoulder, 8=R.elbow, 10=R.wrist

def _load_png_rgba(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.shape[2] == 3:
        # 알파 없으면 255로 추가
        im = np.concatenate([im, np.full((*im.shape[:2],1), 255, dtype=np.uint8)], axis=2)
    return im

def _composite_rgba(dst_bgr: np.ndarray, src_rgba: np.ndarray, x: int, y: int, alpha_mul: float=1.0):
    """
    src_rgba를 (x,y)를 좌상단으로 dst_bgr에 합성. 범위 벗어나면 자동 클리핑.
    """
    H, W = dst_bgr.shape[:2]
    h, w = src_rgba.shape[:2]
    if w<=0 or h<=0: return dst_bgr

    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1<=x0 or y1<=y0: return dst_bgr

    roi_dst = dst_bgr[y0:y1, x0:x1]
    roi_src = src_rgba[(y0-y):(y0-y)+(y1-y0), (x0-x):(x0-x)+(x1-x0)]

    src_rgb = roi_src[..., :3].astype(np.float32)
    src_a   = (roi_src[..., 3:4].astype(np.float32) / 255.0) * float(alpha_mul)
    dst_rgb = roi_dst.astype(np.float32)

    out = src_rgb * src_a + dst_rgb * (1.0 - src_a)
    roi_dst[...] = np.clip(out, 0, 255).astype(np.uint8)
    return dst_bgr

def _rotate_scale_rgba(img_rgba: np.ndarray, angle_deg: float, scale: float, pivot: Tuple[float,float]) -> Tuple[np.ndarray, Tuple[int,int]]:
    """
    주어진 pivot(이미지 좌표계 기준)을 회전 중심으로 회전/스케일 후,
    결과 이미지를 새 캔버스에 놓고, pivot이 결과에서도 동일한 상대 위치가 되도록 translate 반환.
    """
    h, w = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D(center=pivot, angle=angle_deg, scale=scale)

    # 회전 후 bounding box 계산
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int(w * cos + h * sin + 0.5)
    new_h = int(h * sin + w * cos + 0.5)

    # pivot이 동일 위치로 오도록 이동량 보정
    # 회전 전 pivot -> 회전 후 pivot 좌표
    M_pivot = np.array([ [M[0,0], M[0,1], M[0,2]],
                         [M[1,0], M[1,1], M[1,2]] ], dtype=np.float32)
    px, py = float(pivot[0]), float(pivot[1])
    rx = M_pivot[0,0]*px + M_pivot[0,1]*py + M_pivot[0,2]
    ry = M_pivot[1,0]*px + M_pivot[1,1]*py + M_pivot[1,2]
    # 결과 캔버스에서 pivot이 동일 좌표(px,py)에 위치하도록 translation 추가
    tx = (px - rx) + (new_w - w)/2.0
    ty = (py - ry) + (new_h - h)/2.0
    M[0,2] += tx
    M[1,2] += ty

    out = cv2.warpAffine(img_rgba, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return out, (new_w, new_h)

class ImageWingFX:
    """
    PNG 날개 이미지를 어깨에 붙여서 회전/스케일/펼침(팔 각도 기반) 표현.
    - left_path만 주면 오른쪽은 자동 미러링하여 사용.
    - pivot_px: PNG 내에서 어깨가 위치할 좌표(기본 왼쪽날개는 이미지 좌측 안쪽/중앙 높이로 가정)
    - offset_px: 최종 합성 시 어깨 좌표에서의 추가 오프셋(x,y), 좌/우에 자동 부호 적용 옵션 제공.

    스케일/펼침:
      spread ∈ [1, max_spread]  ; elbow가 shoulder보다 높을수록(팔 올릴수록) spread 증가
      spread = 1 + (max_spread-1) * clip( (shoulder_y - elbow_y) / spread_norm, 0..1 )^gamma
    """
    def __init__(self,
                 left_path: str,
                 right_path: Optional[str] = None,
                 pivot_px: Tuple[float,float] = (40.0, 0.5),  # 비율 지원: (x_px or ratio, y_ratio)
                 offset_px: Tuple[int,int] = (0,-20),
                 mirror_right: bool = True,
                 base_scale: float = 1.0,
                 alpha: float = 0.9,
                 max_spread: float = 1.6,
                 spread_norm_px: float = 240.0,
                 spread_gamma: float = 1.0,
                 ema: float = 0.35):
        self.imgL = _load_png_rgba(left_path) if left_path else None
        self.imgR = _load_png_rgba(right_path) if right_path else None
        self.mirror_right = mirror_right
        if self.imgL is None and self.imgR is None:
            raise FileNotFoundError("Wing PNG not found")

        # pivot_px: x가 0~1이면 비율로 간주, y는 0~1 비율(이미지 높이)
        def _pivot_to_px(img, pivot):
            h, w = img.shape[:2]
            px = pivot[0]*w if 0 <= pivot[0] <= 1 else float(pivot[0])
            py = pivot[1]*h if 0 <= pivot[1] <= 1 else float(pivot[1])
            return (float(px), float(py))

        img_ref = self.imgL if (self.imgL is not None) else self.imgR
        if img_ref is None:
            raise FileNotFoundError("Wing PNG not found")
        self.pivotL = _pivot_to_px(img_ref, pivot_px)
        self.pivotR = None  # 오른쪽은 나중에 계산

        self.offset = offset_px
        self.base_scale = float(base_scale)
        self.alpha = float(alpha)
        self.max_spread = float(max_spread)
        self.spread_norm = float(spread_norm_px)
        self.spread_gamma = float(spread_gamma)
        self.ema = float(ema)

        self._W = None; self._H = None
        self._kps_ema = None

    def with_size(self, W:int, H:int):
        self._W, self._H = int(W), int(H)
        return self

    def _get_imgR(self):
        if self.imgR is not None:
            return self.imgR
        # mirror_right가 꺼져 있으면 자동 미러 생성하지 않음
        if (self.imgL is None) or (not self.mirror_right):
            return None
        return cv2.flip(self.imgL, 1)

    def _arm_raise_factor(self, shoulder, elbow) -> float:
        """
        어깨보다 팔꿈치가 얼마나 위에 있나로 펼침 정도 결정.
        """
        if shoulder is None or elbow is None:
            return 0.0
        dy = float(shoulder[1] - elbow[1])  # elbow가 위에 있으면 양수
        f = max(0.0, min(1.0, (dy / self.spread_norm)))
        if self.spread_gamma != 1.0:
            f = f ** self.spread_gamma
        return f

    def _angle_from_arm(self, shoulder, elbow, right_side: bool) -> float:
        """
        어깨→팔꿈치 벡터 각도(도). 오른쪽/왼쪽에 따라 살짝 보정(날개 기본 방향).
        """
        if shoulder is None or elbow is None:
            return -10.0 if right_side else 190.0  # 기본 각
        vx = float(elbow[0] - shoulder[0])
        vy = float(elbow[1] - shoulder[1])
        ang = math.degrees(math.atan2(vy, vx))
        # 날개가 팔과 완전히 평행하지 않도록  +/− 보정(디자인 취향)
        ang += -10.0 if right_side else +10.0
        return ang

    def _smooth_kps(self, kps: np.ndarray) -> np.ndarray:
        if self._kps_ema is None:
            self._kps_ema = kps.copy()
        else:
            self._kps_ema = self.ema * self._kps_ema + (1.0 - self.ema) * kps
        return self._kps_ema

    def update(self, kps: Optional[np.ndarray], vis: Optional[np.ndarray] = None):
        if kps is None or len(kps) == 0:
            return
        self._smooth_kps(kps[0])  # (K,2)

    def _draw_one(self, frame_bgr: np.ndarray, side: str):
        if self._kps_ema is None:
            return frame_bgr

        if side == 'L':
            shoulder = self._kps_ema[5] if len(self._kps_ema) > 5 else None
            elbow    = self._kps_ema[7] if len(self._kps_ema) > 7 else None
            img      = self.imgL if self.imgL is not None else self.imgR
            pivot    = self.pivotL
            right_side = False
        else:
            shoulder = self._kps_ema[6] if len(self._kps_ema) > 6 else None
            elbow    = self._kps_ema[8] if len(self._kps_ema) > 8 else None
            img      = self._get_imgR()
            # 오른쪽 피벗: 좌우 반전했으면 왼쪽 pivot을 미러링
            if img is not None:
                h, w = img.shape[:2]
                self.pivotR = (w - self.pivotL[0], self.pivotL[1])
            pivot   = self.pivotR
            right_side = True

        if img is None or shoulder is None:
            return frame_bgr

        # 펼침 정도 → 스케일
        raise_f = self._arm_raise_factor(shoulder, elbow)
        spread = 1.0 + (self.max_spread - 1.0) * raise_f
        scale = self.base_scale * spread

        # 회전 각
        angle = self._angle_from_arm(shoulder, elbow, right_side)

        # 이미지 회전/스케일
        warped, (nw, nh) = _rotate_scale_rgba(img, angle_deg=angle, scale=scale, pivot=pivot)

        # 합성 위치(어깨 기준 + 오프셋)
        ox, oy = self.offset
        # 오른쪽에선 x 오프셋 부호 반전(원하면)
        if right_side:
            ox = -ox
        x = int(round(shoulder[0] + ox - pivot[0]))
        y = int(round(shoulder[1] + oy - pivot[1]))

        return _composite_rgba(frame_bgr, warped, x, y, alpha_mul=self.alpha)

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_bgr = self._draw_one(frame_bgr, 'L')
        frame_bgr = self._draw_one(frame_bgr, 'R')
        return frame_bgr
