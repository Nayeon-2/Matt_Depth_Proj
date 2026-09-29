# module/effects/ripple_lines_1.py
# -*- coding: utf-8 -*-
import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# 전역 캐시: (그리드, 가우시안 커널, AA 결과 등)
# ─────────────────────────────────────────────────────────────
_CACHE: Dict[Tuple, torch.Tensor] = {}

# ------------------------------------------------------------
# 3D/4D 안전 contiguous 헬퍼
# ------------------------------------------------------------
def _contig_safe(t: torch.Tensor) -> torch.Tensor:
    # 4D(NCHW)만 channels_last 의미가 있음
    if t.dim() == 4:
        return t.contiguous(memory_format=torch.channels_last)
    return t.contiguous()

# ------------------------------------------------------------
# 유틸: "r,g,b; r,g,b; ..." or "r,g,b" -> [[r,g,b], ...] (0..1)
# ------------------------------------------------------------
def _parse_color_list(s: str) -> List[List[float]]:
    s = (s or "").strip()
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        cols = []
        for p in parts:
            r, g, b = [float(x) for x in p.split(",")]
            cols.append([r / 255.0, g / 255.0, b / 255.0])
        return cols if cols else [[0.0, 1.0, 1.0]]
    if s:
        r, g, b = [float(x) for x in s.split(",")]
        return [[r / 255.0, g / 255.0, b / 255.0]]
    return [[0.0, 1.0, 1.0]]  # 기본: CYAN

# ------------------------------------------------------------
# 좌표 그리드 캐시 (픽셀 좌표)
# ------------------------------------------------------------
def _get_xy_1hw(H: int, W: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = ("xy", H, W, str(device), str(dtype))
    got = _CACHE.get(key)
    if got is not None:
        return got  # type: ignore
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    xx = xx.unsqueeze(0)  # [1,H,W]
    yy = yy.unsqueeze(0)
    _CACHE[key] = (xx, yy)
    return xx, yy

# ------------------------------------------------------------
# Sobel
# ------------------------------------------------------------
def _sobel_1hw(x_1hw: torch.Tensor):
    kkey = ("sobel", str(x_1hw.device), str(x_1hw.dtype))
    k = _CACHE.get(kkey)
    if k is None:
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                          device=x_1hw.device, dtype=x_1hw.dtype).view(1,1,3,3)/8.0
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                          device=x_1hw.device, dtype=x_1hw.dtype).view(1,1,3,3)/8.0
        _CACHE[kkey] = (kx, ky)
        k = _CACHE[kkey]
    kx, ky = k  # type: ignore
    x = x_1hw.unsqueeze(0)  # [1,1,H,W]
    gx = F.conv2d(x, kx, padding=1).squeeze(0)     # [1,H,W]
    gy = F.conv2d(x, ky, padding=1).squeeze(0)     # [1,H,W]
    return gx, gy

# ------------------------------------------------------------
# 가우시안 블러 (separable, 1채널 [1,H,W] 전용)
# ------------------------------------------------------------
def _gaussian_blur_1hw(x_1hw: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px is None or sigma_px <= 0.0:
        return x_1hw
    device = x_1hw.device
    dtype  = x_1hw.dtype
    radius = max(1, int(3.0 * float(sigma_px)))
    ks     = 2 * radius + 1

    kkey = ("g1", dtype, device, float(sigma_px), radius, ks)
    kernels = _CACHE.get(kkey)
    if kernels is None:
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k1 = torch.exp(-(x * x) / (2.0 * (sigma_px ** 2)))
        k1 = (k1 / k1.sum()).view(1, 1, 1, ks)         # [1,1,1,ks]
        k1T = k1.transpose(2, 3).contiguous()          # [1,1,ks,1]
        _CACHE[kkey] = (k1, k1T)
        kernels = _CACHE[kkey]
    k1, k1T = kernels  # type: ignore

    inp = x_1hw.unsqueeze(0)                           # [1,1,H,W]
    y = F.conv2d(inp, k1, padding=(0, radius))
    y = F.conv2d(y,   k1T, padding=(radius, 0))
    return y.squeeze(0)                                # [1,H,W]

# ------------------------------------------------------------
# 연속 도메인 지글 감지(구조텐서) + 에지 각도
# ------------------------------------------------------------
def _jiggle_from_alpha(alpha_1hw: torch.Tensor,
                       win_px: int = 24,
                       min_grad: float = 0.02,
                       coh_hi: float = 0.8):
    gx, gy = _sobel_1hw(alpha_1hw)                 # [1,H,W]
    mag = torch.sqrt(gx*gx + gy*gy).clamp_min(1e-8)
    sigma = max(1.0, float(win_px) / 6.0)

    J11 = _gaussian_blur_1hw(gx*gx, sigma)
    J22 = _gaussian_blur_1hw(gy*gy, sigma)
    J12 = _gaussian_blur_1hw(gx*gy, sigma)

    tr = J11 + J22
    dlt = torch.sqrt((J11 - J22)*(J11 - J22) + 4.0*J12*J12).clamp_min(1e-12)
    lam1 = 0.5*(tr + dlt)
    lam2 = 0.5*(tr - dlt)
    coherence = ((lam1 - lam2) / (lam1 + lam2 + 1e-9)).clamp(0, 1)

    theta = torch.atan2(gy, gx)                    # 법선 각도
    scale = max(1e-3, 0.5*min_grad)
    grad_gate = torch.sigmoid((mag - min_grad)/scale)
    jiggle = (1.0 - coherence) * grad_gate
    return jiggle.clamp(0,1), theta

# ------------------------------------------------------------
# 벡터화된 접선방향 1D 블러 (grid_sample 일괄 샘플)
# ------------------------------------------------------------
def _directional_blur_1hw(x_1hw: torch.Tensor,
                          theta_1hw: torch.Tensor,
                          radius_px: int = 3,
                          sigma_px: float = 2.0):
    """
    x_1hw: [1,H,W], theta_1hw: [1,H,W] (법선 각도)
    반환: [1,H,W]
    """
    H, W = x_1hw.shape[-2:]
    device, dtype = x_1hw.device, x_1hw.dtype

    # 접선 단위벡터 t = (-sin(theta), cos(theta))
    st = torch.sin(theta_1hw)
    ct = torch.cos(theta_1hw)
    tx = -st
    ty =  ct
    nrm = torch.sqrt(tx*tx + ty*ty).clamp_min(1e-8)
    tx, ty = tx/nrm, ty/nrm                           # [1,H,W]

    R = max(1, int(radius_px))
    K = 2*R + 1
    xs = torch.arange(-R, R+1, device=device, dtype=dtype).view(K,1,1)  # [K,1,1]

    # 베이스 좌표
    xx, yy = _get_xy_1hw(H, W, device, dtype)        # [1,H,W]
    ox = xx + xs * tx                                 # [K,H,W]
    oy = yy + xs * ty                                 # [K,H,W]

    # 정규화 그리드 [-1,1]
    gx = (ox/(W-1))*2 - 1
    gy = (oy/(H-1))*2 - 1
    grid = torch.stack([gx, gy], dim=-1)             # [K,H,W,2]

    # 입력 배치 확장 (4D → channels_last 가능)
    x4 = _contig_safe(x_1hw.unsqueeze(0).expand(K, -1, -1, -1))

    # 가우시안 가중
    w = torch.exp(-(xs.squeeze(-1).squeeze(-1) * xs.squeeze(-1).squeeze(-1)) / (2.0*(sigma_px**2)))
    w = (w / w.sum()).view(K,1,1,1)                  # [K,1,1,1]

    # 일괄 샘플
    samp = F.grid_sample(x4, grid, mode='bilinear', padding_mode='border', align_corners=True)
    y = (samp * w).sum(dim=0)                         # [1,H,W]
    return y

# ------------------------------------------------------------
# 다운스케일/업스케일 (AA 경량화)
# ------------------------------------------------------------
def _down_up_1hw(x_1hw: torch.Tensor, scale: float):
    if not (0.05 <= float(scale) < 1.0):
        return x_1hw, None  # no-op
    H, W = x_1hw.shape[-2:]
    h, w = max(1, int(round(H*scale))), max(1, int(round(W*scale)))
    # 다운
    try:
        xd = F.interpolate(x_1hw.unsqueeze(0), size=(h,w),
                           mode='bilinear', align_corners=False, antialias=True).squeeze(0)
    except TypeError:
        xd = F.interpolate(x_1hw.unsqueeze(0), size=(h,w),
                           mode='bilinear', align_corners=False).squeeze(0)
    # 업을 위한 클로저
    def up(z_1hw: torch.Tensor) -> torch.Tensor:
        return F.interpolate(z_1hw.unsqueeze(0), size=(H,W),
                             mode='bilinear', align_corners=False).squeeze(0)
    return xd, up

# ------------------------------------------------------------
# (참고) 바이너리 기반 지글 마스크 – 남겨두되 기본 경로는 미사용
# ------------------------------------------------------------
def _jaggy_mask_from_binary(binary_1hw: torch.Tensor,
                            window_px: int = 24,
                            transitions_thresh: float = 6.0) -> torch.Tensor:
    x = binary_1hw.float()
    dx = (x[:, :, 1:] - x[:, :, :-1]).abs()
    dy = (x[:, 1:, :] - x[:, :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    trans = dx + dy
    k = max(3, int(window_px) | 1)
    dens = F.avg_pool2d(trans.unsqueeze(0), kernel_size=k, stride=1, padding=k//2).squeeze(0)
    mask = (dens >= float(transitions_thresh)).float()
    return mask

# ------------------------------------------------------------
# 핵심: 이동하는 다중 외곽선 (연속 반경 보간 + 선택적 AA)
# ------------------------------------------------------------
@torch.no_grad()
def draw_moving_outlines(
    comp_rgb_3hw: torch.Tensor,   # [3,H,W] float(0..1)
    alpha_1chw: torch.Tensor,     # [1,1,H,W] float(0..1)
    *,
    dilate_fn,                    # callable(alpha_1chw, iterations:int)->alpha_1chw
    t_seconds: float,
    count: int = 3,
    width_px: int = 2,
    spacing_px: int = 24,
    speed_px_s: float = 48.0,
    start_offset_px: int = 6,
    color: str = "0,255,255",
    blend: str = "screen",        # "add" | "screen" | "normal"
    on_bg_only: bool = True,
    ripple_gain: float = 0.5,
    # --- AA 옵션 (고속화) ---
    aa_enable: bool = False,
    aa_window_px: int = 10,
    aa_transitions_thresh: float = 0.2,  # (보존: 미사용)
    aa_sigma_px: float = 3.0,
    aa_downscale: float = 0.5,
    aa_frame_stride: int = 3,            # 몇 프레임마다 AA 재계산할지
    frame_index: Optional[int] = None,   # 외부 프레임 번호(없으면 t_seconds로 근사)
) -> torch.Tensor:

    assert comp_rgb_3hw.dim() == 3 and comp_rgb_3hw.shape[0] == 3, "comp_rgb_3hw must be [3,H,W]"
    assert alpha_1chw.dim() == 4 and alpha_1chw.shape[0] == 1, "alpha_1chw must be [1,1,H,W]"

    # 메모리 포맷(경량 최적화)
    comp_rgb_3hw = _contig_safe(comp_rgb_3hw)

    _, H, W = comp_rgb_3hw.shape
    device = comp_rgb_3hw.device
    dtype  = comp_rgb_3hw.dtype

    # 바이너리/배경 마스크
    a  = (alpha_1chw.clamp(0, 1) > 0.5).to(dtype)    # [1,1,H,W]
    a1 = a.squeeze(0)                                # [1,H,W]
    bg_mask = (1.0 - a1).clamp(0, 1)                 # [1,H,W]

    # 누산 버퍼
    line_rgb_3hw = torch.zeros_like(comp_rgb_3hw).contiguous()  # [3,H,W]
    base = a1                                        # [1,H,W]

    # 팔레트 텐서 (한 번만 생성)
    pal = torch.tensor(_parse_color_list(color), device=device, dtype=dtype)  # [N,3]

    base_gain = float(ripple_gain)
    if base_gain <= 0.0 or int(count) <= 0:
        return comp_rgb_3hw  # gain=0 이면 효과 완전 비활성

    # 사인 왕복: r(t) = r0 + A sin(ωt+φ),  v_max = Aω = speed_px_s → ω = speed/A
    A = 0.5 * float(spacing_px)
    omega = float(speed_px_s) / max(A, 1e-3)

    # ── AA 전처리: aa_enable일 때만 수행(프레임 stride + 캐시)
    theta_1hw = None
    jiggle_1hw = None
    if aa_enable:
        fi = int(frame_index if frame_index is not None else int(t_seconds * 1000.0 // 33.0))  # ~30fps 근사
        aa_key = ("AA", H, W, int(aa_window_px), round(float(aa_sigma_px), 2), round(float(aa_downscale), 2))
        aa_slot = _CACHE.get(aa_key)
        need_recompute = True
        if aa_slot is not None:
            last_fi, cached_theta, cached_jiggle = aa_slot
            need_recompute = (fi - int(last_fi)) >= max(1, int(aa_frame_stride))
        if not need_recompute:
            theta_1hw, jiggle_1hw = cached_theta, cached_jiggle
        else:
            alpha_soft_1hw = alpha_1chw.squeeze(0).clamp(0,1)   # [1,H,W]
            if 0.05 <= float(aa_downscale) < 1.0:
                alpha_ds_1hw, up_fn = _down_up_1hw(alpha_soft_1hw, aa_downscale)
                jiggle_ds, theta_ds = _jiggle_from_alpha(
                    alpha_ds_1hw,
                    win_px=max(3, int(aa_window_px * aa_downscale)),
                    min_grad=0.02, coh_hi=0.8
                )
                # 각도 업샘플: sin/cos 보간 → atan2
                s_up = up_fn(torch.sin(theta_ds))
                c_up = up_fn(torch.cos(theta_ds))
                theta_1hw = torch.atan2(s_up, c_up)
                jiggle_1hw = up_fn(jiggle_ds).clamp(0,1)
            else:
                jiggle_1hw, theta_1hw = _jiggle_from_alpha(
                    alpha_soft_1hw, win_px=int(aa_window_px), min_grad=0.02, coh_hi=0.8
                )
            _CACHE[aa_key] = (fi, theta_1hw, jiggle_1hw)

    # ── 프레임 내 dilate 캐시
    dil_cache: Dict[int, torch.Tensor] = {}
    base_1chw = _contig_safe(base.unsqueeze(0))  # [1,1,H,W]

    def get_dilate(iterations: int) -> torch.Tensor:
        it = int(max(0, iterations))
        got = dil_cache.get(it)
        if got is not None:
            return got                                # [1,H,W]
        if it <= 0:
            out = base_1chw                           # [1,1,H,W]
        else:
            out = dilate_fn(base_1chw, it)           # [1,1,H,W]
        dil_cache[it] = out.squeeze(0)               # [1,H,W]
        return dil_cache[it]

    # ── 미리 필요한 dilation 반복 수 수집 → 일괄 캐시 채우기 (오버헤드 절감)
    need_iters = set()
    for i in range(int(count)):
        r_base = float(start_offset_px + i * spacing_px)
        phi    = float(i) * 0.8
        r      = r_base + A * math.sin(omega * float(t_seconds) + phi)
        rf     = math.floor(r)
        Wpx    = max(1, int(width_px))
        need_iters.update([max(0, rf), max(0, rf + Wpx), max(0, rf + 1), max(0, rf + 1 + Wpx)])
    for it in sorted(need_iters):
        _ = get_dilate(it)

    # ── 라인 루프
    for i in range(int(count)):
        r_base = float(start_offset_px + i * spacing_px)
        phi    = float(i) * 0.8
        r      = r_base + A * math.sin(omega * float(t_seconds) + phi)

        r_floor = math.floor(r)
        r_frac  = float(r - r_floor)
        Wpx     = max(1, int(width_px))

        d0  = get_dilate(r_floor)                    # [1,H,W]
        d1  = get_dilate(r_floor + Wpx)
        d0n = get_dilate(r_floor + 1)
        d1n = get_dilate(r_floor + 1 + Wpx)

        shell_floor = (d1 - d0).clamp(0, 1)          # [1,H,W]
        shell_next  = (d1n - d0n).clamp(0, 1)        # [1,H,W]

        if on_bg_only:
            shell_floor = (shell_floor * bg_mask).clamp(0, 1)
            shell_next  = (shell_next  * bg_mask).clamp(0, 1)

        shell_soft = (1.0 - r_frac) * shell_floor + (r_frac) * shell_next  # [1,H,W]

        if aa_enable and (theta_1hw is not None) and (jiggle_1hw is not None):
            # 다운스케일 AA는 이미 theta/jiggle에 반영됨 → 여기서는 접선 블러만 실행
            tangent_blur = _directional_blur_1hw(
                shell_soft, theta_1hw,
                radius_px=max(1, int(aa_window_px // 5)),   # 보수적으로 축소
                sigma_px=float(max(1.0, aa_sigma_px))
            )
            line_mask = (1.0 - jiggle_1hw) * shell_soft + jiggle_1hw * tangent_blur
        else:
            line_mask = shell_soft

        col = pal[i % pal.shape[0]].view(3, 1, 1)    # [3,1,1]
        line_rgb_3hw = line_rgb_3hw + col * line_mask

    # 효과 누산 clamp
    line_rgb_3hw = line_rgb_3hw.clamp(0.0, 1.0)

    # ── 블렌드 100% 결과(y_line)를 만든 뒤 gain으로 보간 → gain=0 확실히 무효화
    blend = (blend or "screen").lower()
    if blend == "add":
        y_line = (comp_rgb_3hw + line_rgb_3hw).clamp(0, 1)
    elif blend == "screen":
        y_line = (1.0 - (1.0 - comp_rgb_3hw) * (1.0 - line_rgb_3hw)).clamp(0, 1)
    else:  # "normal"
        mask = line_rgb_3hw.max(dim=0, keepdim=True).values.clamp(0, 1)  # [1,H,W]
        y_line = (comp_rgb_3hw * (1.0 - mask) + line_rgb_3hw * mask).clamp(0, 1)

    # 최종: comp + gain*(y_line - comp)  → gain=0이면 완전 원본
    y = comp_rgb_3hw + float(ripple_gain) * (y_line - comp_rgb_3hw)
    return y
