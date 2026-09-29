# module/effects/per_person_core.py
from __future__ import annotations
import math, time
from typing import List, Tuple, Dict, Any, Iterable, Callable, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import cv2
except Exception:
    cv2 = None


# ======================= Palette & Small Utils =======================

@torch.no_grad()
def parse_palette(palette_str: str, device=None, dtype=torch.float32) -> Tensor:
    """
    "r,g,b|r,g,b|..." 또는 "r,g,b;r,g,b;..." → Tensor [P,3], 값 0..1
    빈 문자열이면 기본 팔레트 반환.
    """
    if not isinstance(palette_str, str) or not palette_str.strip():
        palette = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0)]
    else:
        parts = []
        for sep in ("|", ";"):
            if sep in palette_str:
                parts = [p.strip() for p in palette_str.split(sep) if p.strip()]
                break
        if not parts:
            parts = [palette_str.strip()]
        palette = []
        for p in parts:
            try:
                r, g, b = [float(x) for x in p.split(",")]
            except Exception:
                r, g, b = 255.0, 0.0, 0.0
            palette.append((r, g, b))
    t = torch.tensor(palette, device=device, dtype=dtype)
    if t.numel() == 0:
        t = torch.tensor([[255.0, 0.0, 0.0]], device=device, dtype=dtype)
    if t.max().item() > 1.0:
        t = t / 255.0
    return t.clamp_(0.0, 1.0)  # [P,3]


# ======================= Connected Components (from alpha) =======================

@torch.no_grad()
def connected_instances_from_alpha(
    alpha_1chw: Tensor,              # [1,1,H,W] float 0..1
    thr: float = 0.5,
    min_area_px: int = 2000,
    max_instances: int = 4,
    sort_by: str = "area",           # "area" | "x" | "y"
) -> List[Tensor]:
    """
    alpha를 이진화 후 connected components로 인스턴스 분리.
    반환: list[Tensor] 각 항목 [1,H,W] float(0..1)
    cv2가 없으면 전체를 하나로 반환(폴백).
    """
    H = int(alpha_1chw.shape[2])
    W = int(alpha_1chw.shape[3])
    device = alpha_1chw.device
    dtype = alpha_1chw.dtype

    if cv2 is None:
        return [(alpha_1chw.squeeze(0) >= thr).to(dtype)]  # [1,H,W]

    a = alpha_1chw.squeeze(0).squeeze(0).detach().float().cpu().numpy()  # [H,W]
    mask = (a >= float(thr)).astype("uint8")

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return [(alpha_1chw.squeeze(0) >= thr).to(dtype)]

    idxs = [i for i in range(1, num) if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area_px)]
    if not idxs:
        return [(alpha_1chw.squeeze(0) >= thr).to(dtype)]

    if sort_by == "x":
        idxs.sort(key=lambda i: float(centroids[i][0]))  # x asc
    elif sort_by == "y":
        idxs.sort(key=lambda i: float(centroids[i][1]))  # y asc
    else:
        idxs.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)

    idxs = idxs[:max(1, int(max_instances))]

    out = []
    for i in idxs:
        m = (labels == i).astype("uint8")
        t = torch.from_numpy(m).to(device=device, dtype=dtype).unsqueeze(0)  # [1,H,W]
        out.append(t)

    if not out:
        out = [(alpha_1chw.squeeze(0) >= thr).to(dtype)]
    return out


# ======================= YOLO + CC 한방 유틸 =======================

@torch.no_grad()
def get_instance_masks(
    *,
    frame_bgr_np, H: int, W: int,
    alpha_1chw: Tensor,
    use_yolo: bool = False, yolo_model=None, yolo_img: int = 640, yolo_conf: float = 0.5,
    cc_thr: float = 0.5, cc_min_area: int = 2000, cc_max_instances: int = 4, cc_sort_by: str = "area",
    device="cuda", yolo_downscale: int = 2
) -> List[Tensor]:
    """
    우선 YOLO(seg)로 사람 마스크를 얻고, 실패 시 alpha CC 폴백.
    반환 마스크: 각 [1,H,W] 0..1 float
    """
    masks: List[Tensor] = []
    if use_yolo and (yolo_model is not None):
        try:
            res = yolo_model.predict(
                source=frame_bgr_np, imgsz=int(yolo_img), conf=float(yolo_conf),
                classes=[0], verbose=False, device=0 if (str(device).startswith("cuda")) else None
            )
            if res and len(res) > 0 and res[0].masks is not None:
                m = res[0].masks  # ultralytics.Masks
                mdata = m.data  # [N,h,w]
                if mdata.ndim == 3:
                    mdata = mdata.unsqueeze(1)  # [N,1,h,w]
                mfull = F.interpolate(
                    mdata.float(), size=(H, W), mode="bilinear", align_corners=False
                )  # [N,1,H,W]
                for i in range(mfull.shape[0]):
                    mi = (mfull[i] >= 0.5).float()
                    if mi.sum().item() >= 1000:
                        masks.append(mi.to(device=alpha_1chw.device, dtype=alpha_1chw.dtype))
        except Exception:
            masks = []

    if not masks:
        masks = connected_instances_from_alpha(
            alpha_1chw=alpha_1chw, thr=float(cc_thr),
            min_area_px=int(cc_min_area), max_instances=int(cc_max_instances), sort_by=str(cc_sort_by)
        )
    return masks


# ======================= Shell Lines (Reusable) =======================

@torch.no_grad()
def compute_shell_lines(
    inst_mask_1hw: Tensor,        # [1,H,W] 0..1
    *,
    alpha_1chw: Tensor,           # [1,1,H,W]
    dilate_fn,                    # callable([1,1,H,W], int)->[1,1,H,W]
    t_seconds: float,
    count: int,
    width_px: int,
    spacing_px: int,
    speed_px_s: float,
    start_offset_px: int,
    on_bg_only: bool = True,
) -> Tensor:
    """
    인스턴스 마스크 기준 이동하는 얇은 링들의 누적 마스크 [1,H,W] 계산.
    """
    H, W = int(inst_mask_1hw.shape[1]), int(inst_mask_1hw.shape[2])
    device = inst_mask_1hw.device
    dtype  = inst_mask_1hw.dtype

    lines = torch.zeros((1, H, W), device=device, dtype=dtype)
    base11 = inst_mask_1hw.clamp(0, 1).unsqueeze(0)  # [1,1,H,W]

    A = 0.5 * float(spacing_px)
    omega = float(speed_px_s) / max(A, 1e-3)

    for i in range(int(count)):
        r_base = float(start_offset_px + i * spacing_px)
        phi = float(i) * 0.8
        r_center = r_base + A * math.sin(omega * float(t_seconds) + phi)
        r0 = int(max(0.0, r_center))
        r1 = r0 + int(width_px)
        if r1 <= 0:
            continue

        d0 = base11 if r0 <= 0 else dilate_fn(base11, int(r0))
        d1 = dilate_fn(base11, int(max(1, r1)))
        shell = (d1 - d0).clamp(0, 1).squeeze(0)  # [1,H,W]

        if on_bg_only:
            bg_mask = (1.0 - alpha_1chw.squeeze(0)).clamp(0, 1)  # [1,H,W]
            shell = (shell * bg_mask).clamp(0, 1)

        lines = (lines + shell).clamp(0, 1)

    return lines


# ======================= Per-person Painter Reduce Loop =======================

@torch.no_grad()
def blend_apply(mode: str, base_rgb_3hw: Tensor, over_rgb_3hw: Tensor, alpha_1hw: Tensor) -> Tensor:
    """
    mode: 'screen' | 'add' | 'normal'
    over_rgb_3hw: 프리멀티플 가정 아님(일반 RGB), alpha_1hw는 0..1
    """
    m = str(mode).lower().strip()
    a = alpha_1hw.clamp(0, 1)
    if m == "add":
        y = torch.clamp(base_rgb_3hw + over_rgb_3hw * a, 0.0, 1.0)
    elif m == "screen":
        y = 1.0 - (1.0 - base_rgb_3hw) * (1.0 - over_rgb_3hw * a)
        y = y.clamp(0.0, 1.0)
    else:  # "normal"
        y = base_rgb_3hw * (1.0 - a) + over_rgb_3hw * a
        y = y.clamp(0.0, 1.0)
    return y


@torch.no_grad()
def per_person_paint_loop(
    *,
    comp_rgb_3hw: Tensor,                     # [3,H,W]
    instance_masks_1hw: List[Tensor],
    painter_for_one: Callable[[Tensor, Tensor, int], Tuple[Tensor, Tensor]],
    reduce_mode: str = "screen",
) -> Tensor:
    """
    painter_for_one(base_rgb_3hw, mask_1hw, idx) -> (over_rgb_3hw, alpha_1hw)
    여러 인스턴스의 결과를 reduce_mode로 합성하여 반환.
    """
    out = comp_rgb_3hw
    for i, m1hw in enumerate(instance_masks_1hw or []):
        over_rgb, a_1hw = painter_for_one(out, m1hw, i)
        out = blend_apply(reduce_mode, out, over_rgb, a_1hw)
    return out


# ======================= Tracker Helper (optional) =======================

@torch.no_grad()
def build_detections_from_masks(masks_1hw: List[Tensor]) -> List[Dict[str, Any]]:
    """
    마스크들로부터 bbox 검출 리스트 구축: [{'bbox':(x1,y1,x2,y2), 'conf':1.0, 'cls':0}, ...]
    """
    dets: List[Dict[str, Any]] = []
    for m in masks_1hw or []:
        try:
            idx = torch.nonzero((m[0] > 0.5), as_tuple=False)
            if idx.numel() == 0:
                continue
            ys = idx[:, 0].to(torch.float32); xs = idx[:, 1].to(torch.float32)
            x1 = int(xs.min().item()); y1 = int(ys.min().item())
            x2 = int(xs.max().item()); y2 = int(ys.max().item())
            dets.append({'bbox': (x1, y1, x2, y2), 'conf': 1.0, 'cls': 0})
        except Exception:
            continue
    return dets


# ======================= SIDE MAPPING (LEFT / RIGHT) =======================
# 이 블록은 좌/우 색 매핑 및 정렬을 다양한 효과 모듈에서 재사용하기 위한 공용 유틸입니다.

@torch.no_grad()
def centroid_x(mask_1hw: Tensor) -> float:
    """
    단일 마스크의 x 중심(열 평균) 반환. 없으면 +inf(오른쪽 끝보다 더 오른쪽 취급)
    """
    try:
        idx = torch.nonzero((mask_1hw[0] > 0.5), as_tuple=False)
        if idx.numel() == 0:
            return float("inf")
        xs = idx[:, 1].to(torch.float32)
        return float(xs.mean().item())
    except Exception:
        return float("inf")


@torch.no_grad()
def classify_side(mask_1hw: Tensor, W: int, center_x: Optional[float] = None) -> str:
    """
    화면 중심 기준 좌/우 분류. center_x가 없으면 W/2 사용.
    반환: 'L' 또는 'R'
    """
    cx = centroid_x(mask_1hw)
    mid = float(center_x) if (center_x is not None) else (0.5 * float(W))
    return 'L' if cx < mid else 'R'


@torch.no_grad()
def order_masks_by_side(
    masks_1hw: List[Tensor], W: int,
    mode: str = "center_out",  # "center_out" | "left_to_right" | "right_to_left"
    center_x: Optional[float] = None
) -> List[Tuple[Tensor, str, float]]:
    """
    마스크들을 지정한 정렬방식으로 정렬하여 [(mask, side, key), ...] 반환.
      - center_out: 중심에서 가까운 순으로(동일 거리면 x 작은 것 우선)
      - left_to_right: x 중심 오름차순
      - right_to_left: x 중심 내림차순
    side: 'L' or 'R'
    key: 정렬 기준값(디버깅용)
    """
    mid = float(center_x) if (center_x is not None) else (0.5 * float(W))
    cxs: List[Tuple[int, float]] = []  # (idx, cx)
    for i, m in enumerate(masks_1hw or []):
        cxs.append((i, centroid_x(m)))

    items: List[Tuple[Tensor, str, float]] = []
    if mode == "left_to_right":
        cxs.sort(key=lambda t: t[1])
        for i, cx in cxs:
            items.append((masks_1hw[i], 'L' if cx < mid else 'R', cx))
        return items
    if mode == "right_to_left":
        cxs.sort(key=lambda t: t[1], reverse=True)
        for i, cx in cxs:
            items.append((masks_1hw[i], 'L' if cx < mid else 'R', cx))
        return items

    # center_out
    by_dist = []
    for i, cx in cxs:
        dist = abs(cx - mid)
        by_dist.append((i, dist, cx))
    by_dist.sort(key=lambda t: (t[1], t[2]))  # (distance asc, cx asc)
    for i, dist, cx in by_dist:
        items.append((masks_1hw[i], 'L' if cx < mid else 'R', dist))
    return items


@torch.no_grad()
def assign_colors_by_side(
    ordered_masks: List[Tuple[Tensor, str, float]],
    *,
    palette_left_3c: Tensor,     # [PL,3]
    palette_right_3c: Tensor,    # [PR,3]
    palette_default_3c: Optional[Tensor] = None  # [P,3] (옵션)
) -> List[Tuple[Tensor, Tensor, str]]:
    """
    정렬된 (mask, side, key) 리스트에 좌/우 팔레트를 순환 적용하여
    [(mask, color_3hw, side), ...] 반환.
    palette_default_3c가 주어지면 side가 모호할 때(이상치) 기본 팔레트 사용.
    """
    out: List[Tuple[Tensor, Tensor, str]] = []
    li = 0; ri = 0
    PL = int(palette_left_3c.shape[0]) if palette_left_3c is not None else 0
    PR = int(palette_right_3c.shape[0]) if palette_right_3c is not None else 0

    def _col_from(pal: Tensor, idx: int) -> Tensor:
        return pal[idx % pal.shape[0]].view(3, 1, 1)

    for m, side, _ in (ordered_masks or []):
        if side == 'L' and PL > 0:
            col = _col_from(palette_left_3c, li); li += 1
        elif side == 'R' and PR > 0:
            col = _col_from(palette_right_3c, ri); ri += 1
        else:
            # 모호/이상치 → 기본 팔레트 사용
            if palette_default_3c is not None and palette_default_3c.numel() > 0:
                idx = (li + ri)
                col = _col_from(palette_default_3c, idx)
            else:
                # fallback: 노란색
                col = torch.tensor([1.0, 1.0, 0.0], device=(palette_left_3c.device if palette_left_3c is not None else 'cpu')).view(3,1,1)
        out.append((m, col, side))
    return out


@torch.no_grad()
def order_and_colorize_masks(
    *,
    masks_1hw: List[Tensor], W: int,
    side_mode: str = "center_out",      # "center_out" | "left_to_right" | "right_to_left"
    left_palette_str: str = "255,0,0|255,64,64|255,128,128",
    right_palette_str: str = "0,128,255|0,200,255|64,224,255",
    default_palette_str: Optional[str] = None,
    device=None, dtype=torch.float32, center_x: Optional[float] = None
) -> List[Tuple[Tensor, Tensor, str]]:
    """
    한 번에: 마스크 정렬 → 좌/우 팔레트로 색 배정.
    반환: [(mask_1hw, color_3hw, side('L'/'R')),...]  (정렬된 순서)
    """
    ordered = order_masks_by_side(masks_1hw, W=W, mode=side_mode, center_x=center_x)
    palL = parse_palette(left_palette_str, device=device, dtype=dtype)
    palR = parse_palette(right_palette_str, device=device, dtype=dtype)
    palD = parse_palette(default_palette_str, device=device, dtype=dtype) if (default_palette_str not in (None, "")) else None
    return assign_colors_by_side(ordered, palette_left_3c=palL, palette_right_3c=palR, palette_default_3c=palD)
