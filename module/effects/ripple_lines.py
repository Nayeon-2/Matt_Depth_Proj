# module/effects/ripple_lines.py
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def draw_moving_outlines(
    comp_rgb_3hw: torch.Tensor,   # [3,H,W] RGB 0..1 (CUDA 권장)
    alpha_1chw: torch.Tensor,     # [1,1,H,W] 0..1   (CUDA 권장)
    *,
    dilate_fn,                    # callable([1,1,H,W], k)->[1,1,H,W]
    t_seconds: float,
    count: int = 3,
    width_px: int = 2,
    spacing_px: int = 24,
    speed_px_s: float = 48.0,     # 사인 왕복 최대 반경 속도(px/s)
    start_offset_px: int = 6,
    color: str = "0,255,255",     # "R,G,B"
    blend: str = "screen",        # "add" | "screen" | "normal"
    on_bg_only: bool = True,
    ripple_gain: float = 1.0
) -> torch.Tensor:
    """
    - 라인 '움직임'은 사인 파형(왕복)으로 부드럽게.
    - 라인 스무딩은 가벼운 3x3 평균+약한 3x3 가우시안만 적용.
    - 분수 반경 보간/다중 샘플 제거로 성능 향상.
    - 큰 반경 dilate 시 1/2 해상도에서 연산 후 업샘플로 가속.
    """
    H, W = comp_rgb_3hw.shape[1:]

    # soft alpha (경계 부드럽게 활용)
    a1 = alpha_1chw.clamp(0, 1).squeeze(0)   # [1,H,W]
    bg_mask = (1.0 - a1).clamp(0, 1)         # [1,H,W]

    # 색 (RGB)
    cr, cg, cb = [float(x)/255.0 for x in color.split(",")]
    line_col = torch.tensor([cr, cg, cb],
                            device=comp_rgb_3hw.device,
                            dtype=comp_rgb_3hw.dtype).view(3,1,1)

    lines = torch.zeros((1, H, W), device=comp_rgb_3hw.device, dtype=comp_rgb_3hw.dtype)
    base  = a1  # [1,H,W]

    # ── 가속용: 큰 반경(k)이면 저해상도 경로 사용 ─────────────────────────────
    # max_pool2d의 커널이 매우 커지면 비용 급증 → 1/2 스케일에서 dilate 후 업샘플.
    def _dilate_int_fast(x_1hw: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x_1hw
        # 임계값은 경험적으로 45 전후에서 전환 (환경에 맞게 조절 가능)
        if k <= 45:
            return dilate_fn(x_1hw.unsqueeze(0), int(k)).squeeze(0)

        # 저해상도 경로
        scale = 0.5
        ks = max(1, int(round(k * scale)))
        xs = F.interpolate(x_1hw.unsqueeze(0), scale_factor=scale, mode="bilinear",
                           align_corners=False).squeeze(0)              # [1,h,w]
        ds = dilate_fn(xs.unsqueeze(0), ks).squeeze(0)                   # [1,h,w]
        # 업샘플(부드럽게)
        du = F.interpolate(ds.unsqueeze(0), size=(H, W), mode="bilinear",
                           align_corners=False).squeeze(0)               # [1,H,W]
        return du

    # ── 움직임: 사인 왕복 ────────────────────────────────────────────────────
    # r(t) = r_base + A * sin(ω t + φ),  v_max = A*ω ≈ speed_px_s → ω = speed_px_s/A
    A = 0.5 * float(spacing_px)                 # 왕복 진폭(간격의 절반 정도)
    omega = float(speed_px_s) / max(A, 1e-3)    # 최대 반경속도를 speed_px_s에 맞춤

    for i in range(int(count)):
        r_base = float(start_offset_px + i * spacing_px)
        phi    = float(i) * 0.9                 # 라인 간 위상차로 겹침 감소

        r_center = r_base + A * math.sin(omega * float(t_seconds) + phi)
        r0 = int(max(0.0, math.floor(r_center)))
        r1 = r0 + max(1, int(width_px))

        if r1 <= 0:
            continue

        # 정수 반경으로 한 번씩만 연산 → 빠름
        d0 = base if r0 <= 0 else _dilate_int_fast(base, r0)   # [1,H,W]
        d1 = _dilate_int_fast(base, r1)                        # [1,H,W]
        shell = (d1 - d0).clamp(0, 1)

        if on_bg_only:
            shell = (shell * bg_mask).clamp(0, 1)

        lines = (lines + shell).clamp(0, 1)

    # ── 가벼운 스무딩 (프레임 드랍 최소) ─────────────────────────────────────
    # 1) 3x3 평균풀링(한 번)
    lines = F.avg_pool2d(lines, kernel_size=3, stride=1, padding=1)

    # 2) 약한 3x3 분리형 가우시안(한 번) — 매우 저비용
    def _gblur3(x, sigma=0.85):
        ks = 3
        ar = torch.arange(ks, device=x.device, dtype=x.dtype) - ks//2
        w  = torch.exp(-(ar**2) / (2*sigma*sigma))
        w  = (w / w.sum()).to(dtype=x.dtype)
        kx = w.view(1,1,1,ks)   # [1,1,1,3]
        ky = w.view(1,1,ks,1)   # [1,1,3,1]
        y  = F.conv2d(x, kx, padding=(0, ks//2))
        y  = F.conv2d(y, ky, padding=(ks//2, 0))
        return y

    lines = _gblur3(lines).clamp(0, 1)


    # 3) 감마로 가장자리 살짝 조임(0.85~0.92 권장)
    lines = lines.pow(0.88)

    # ── 블렌딩 ───────────────────────────────────────────────────────────────
    if blend == "add":
        y = torch.clamp(comp_rgb_3hw + line_col * lines * float(ripple_gain), 0, 1)
    elif blend == "screen":
        # screen은 비선형이라 gain이 커질수록 금방 흰색으로 포화 → 과하면 효과 변화가 둔감하게 느껴집니다.
        line_img = torch.clamp(line_col * lines * float(ripple_gain), 0.0, 1.0)
        y = 1.0 - (1.0 - comp_rgb_3hw) * (1.0 - line_img)
        y = y.clamp(0, 1)
    else:  # normal
        y = comp_rgb_3hw * (1.0 - lines) + (line_col * lines * float(ripple_gain))

    return y
