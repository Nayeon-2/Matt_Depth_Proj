# module/effects/energy_per_person.py
from __future__ import annotations
import torch
from torch import Tensor
from typing import List

from .per_person_core import (
    parse_palette, connected_instances_from_alpha, get_instance_masks,
    compute_shell_lines, per_person_paint_loop, blend_apply
)


# ========== 기존 호환용: 내부 helper 노출 ==========
# (run_loop 등에서 임포트하던 이름 유지)
_parse_palette = parse_palette
_connected_instances_from_alpha = connected_instances_from_alpha


@torch.no_grad()
def _energy_over_from_lines(
    *,
    base_rgb_3hw: Tensor,     # [3,H,W] (현재 합성 위)
    lines_1hw: Tensor,        # [1,H,W] (0..1)
    color_3hw: Tensor,        # [3,1,1] (0..1)
    gain: float,
    blend: str = "screen",
) -> tuple[Tensor, Tensor]:
    """
    라인 마스크와 색으로 over 이미지를 생성.
    반환: (over_rgb_3hw, alpha_1hw)  — 프리멀티 알파 가정
    """
    line_rgb = torch.clamp(color_3hw * lines_1hw * float(gain), 0.0, 1.0)  # [3,H,W]
    b = str(blend).lower().strip()
    if b == "add":
        over = torch.clamp(base_rgb_3hw + line_rgb, 0.0, 1.0)
    elif b == "screen":
        over = 1.0 - (1.0 - base_rgb_3hw) * (1.0 - line_rgb)
    else:  # normal
        over = line_rgb
    # 알파는 라인 강도 자체를 사용
    alpha = lines_1hw.clamp(0, 1)
    return over, alpha


@torch.no_grad()
def draw_energy_from_instance_masks(
    *,
    comp_rgb_3hw: Tensor,                     # [3,H,W] 0..1
    instance_masks_1hw: List[Tensor],         # 각 [1,H,W] float(0..1)
    alpha_1chw: Tensor,                       # [1,1,H,W] 0..1 (bg 제한용)
    dilate_fn,
    t_seconds: float,
    # 라인/합성
    count: int = 3,
    width_px: int = 2,
    spacing_px: int = 24,
    speed_px_s: float = 48.0,
    start_offset_px: int = 6,
    gain: float = 1.0,
    blend: str = "screen",
    on_bg_only: bool = True,
    # 팔레트
    palette_str: str = "255,0,0|0,255,0|0,128,255|255,255,0",
) -> Tensor:
    """
    (사람별) 이동 라인 에너지 렌더링. 기존 시그니처 유지.
    """
    device = comp_rgb_3hw.device
    dtype  = comp_rgb_3hw.dtype
    palette = parse_palette(palette_str, device=device, dtype=dtype)  # [P,3]

    H, W = comp_rgb_3hw.shape[1], comp_rgb_3hw.shape[2]
    painted = torch.zeros((1, H, W), device=device, dtype=dtype)      # 중첩 차단

    def _painter_for_one(base_3hw: Tensor, mask_1hw: Tensor, idx: int):
        nonlocal painted
        # 동심 링 생성
        lines = compute_shell_lines(
            mask_1hw, alpha_1chw=alpha_1chw, dilate_fn=dilate_fn, t_seconds=t_seconds,
            count=int(count), width_px=int(width_px), spacing_px=int(spacing_px),
            speed_px_s=float(speed_px_s), start_offset_px=int(start_offset_px),
            on_bg_only=bool(on_bg_only),
        )
        unique_lines = (lines * (1.0 - painted)).clamp(0, 1)
        col = palette[idx % palette.shape[0]].view(3, 1, 1)
        over_rgb, a_1hw = _energy_over_from_lines(
            base_rgb_3hw=base_3hw, lines_1hw=unique_lines, color_3hw=col, gain=gain, blend=blend
        )
        painted = torch.clamp(painted + unique_lines, 0.0, 1.0)
        return over_rgb, a_1hw

    out = per_person_paint_loop(
        comp_rgb_3hw=comp_rgb_3hw,
        instance_masks_1hw=instance_masks_1hw or [],
        painter_for_one=_painter_for_one,
        reduce_mode="screen" if blend == "screen" else ("add" if blend == "add" else "normal"),
    )
    return out


@torch.no_grad()
def draw_energy_per_person(
    *,
    comp_rgb_3hw: Tensor,             # [3,H,W] 0..1
    alpha_1chw: Tensor,               # [1,1,H,W] 0..1
    dilate_fn,
    t_seconds: float,
    # 파라미터 (run_loop 전달과 호환)
    count: int = 3,
    width_px: int = 2,
    spacing_px: int = 24,
    speed_px_s: float = 48.0,
    start_offset_px: int = 6,
    gain: float = 1.0,
    blend: str = "screen",
    on_bg_only: bool = True,
    per_person: bool = True,          # 시그니처 유지용
    palette_str: str = "255,0,0|0,255,0|0,128,255|255,255,0",
    # 인스턴스 검출 파라미터
    max_instances: int = 4,
    instance_thr: float = 0.5,
    min_area_px: int = 2000,
    sort_by: str = "area",
) -> Tensor:
    """
    YOLO 없이 alpha의 connected-components만 사용해 per-person 에너지를 그립니다.
    (기존 백워드 호환용)
    """
    masks = connected_instances_from_alpha(
        alpha_1chw=alpha_1chw, thr=float(instance_thr),
        min_area_px=int(min_area_px), max_instances=int(max_instances), sort_by=str(sort_by)
    )
    return draw_energy_from_instance_masks(
        comp_rgb_3hw=comp_rgb_3hw, instance_masks_1hw=masks, alpha_1chw=alpha_1chw,
        dilate_fn=dilate_fn, t_seconds=t_seconds, count=count, width_px=width_px,
        spacing_px=spacing_px, speed_px_s=speed_px_s, start_offset_px=start_offset_px,
        gain=gain, blend=blend, on_bg_only=on_bg_only, palette_str=palette_str
    )


# ========== (선택) 통합 인스턴스 파이프라인 래퍼 ==========
@torch.no_grad()
def draw_energy_per_person_auto(
    *,
    comp_rgb_3hw: Tensor,
    frame_bgr_np, H: int, W: int,
    alpha_1chw: Tensor,
    dilate_fn,
    t_seconds: float,
    # 효과 파라미터
    count: int = 3,
    width_px: int = 2,
    spacing_px: int = 24,
    speed_px_s: float = 48.0,
    start_offset_px: int = 6,
    gain: float = 1.0,
    blend: str = "screen",
    on_bg_only: bool = True,
    palette_str: str = "255,0,0|0,255,0|0,128,255|255,255,0",
    # 인스턴스 검출 파라미터
    use_yolo: bool = False, yolo_model=None, yolo_img: int = 640, yolo_conf: float = 0.5,
    max_instances: int = 4, instance_thr: float = 0.5, min_area_px: int = 2000, sort_by: str = "area",
    device="cuda",
) -> Tensor:
    """
    YOLO(가능 시) → CC 폴백으로 인스턴스를 구해 에너지 per-person을 그리는 원스톱 함수.
    run_loop에서 “사람별 효과가 1개라도 있으면” 이걸 호출해도 됨.
    """
    masks = get_instance_masks(
        frame_bgr_np=frame_bgr_np, H=H, W=W, alpha_1chw=alpha_1chw,
        use_yolo=use_yolo, yolo_model=yolo_model, yolo_img=yolo_img, yolo_conf=yolo_conf,
        cc_thr=instance_thr, cc_min_area=min_area_px, cc_max_instances=max_instances, cc_sort_by=sort_by,
        device=comp_rgb_3hw.device, yolo_downscale=2
    )
    return draw_energy_from_instance_masks(
        comp_rgb_3hw=comp_rgb_3hw, instance_masks_1hw=masks, alpha_1chw=alpha_1chw,
        dilate_fn=dilate_fn, t_seconds=t_seconds, count=count, width_px=width_px,
        spacing_px=spacing_px, speed_px_s=speed_px_s, start_offset_px=start_offset_px,
        gain=gain, blend=blend, on_bg_only=on_bg_only, palette_str=palette_str
    )
