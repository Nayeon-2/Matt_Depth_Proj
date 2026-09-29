# effects/energy.py
from __future__ import annotations
import math, time
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# 캐시 (디바이스·dtype·해상도별로 재사용)
# ─────────────────────────────────────────────────────────────
_CACHE = {
    # key: ("yy", device, dtype, H) -> 1xHx1
    # key: ("abs_mask", device, dtype, H, W, center, width, sigma) -> 1xHxW
    # key: ("bbox_state", device, dtype, Hs) -> dict(ema_top, ema_bot, alpha)
}

def _dilate_gpu(a_1chw: torch.Tensor, k: int):
    k = _odd(k)
    if k <= 1: return a_1chw
    pad = k // 2
    # 최대 풀링 = 팽창
    return F.max_pool2d(a_1chw, kernel_size=k, stride=1, padding=pad)

def _get_xy(H: int, W: int, device, dtype):
    """
    0..1 정규화 좌표 그리드 캐시
    반환: xx, yy (각각 [H,W])
    """
    key = ("xy", device, str(dtype), int(H), int(W))
    xy = _CACHE.get(key)
    if xy is None:
        yy = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(H, 1).expand(H, W)
        xx = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, W).expand(H, W)
        _CACHE[key] = (xx, yy)
    return _CACHE[key]

def _get_yy(H: int, device, dtype):
    key = ("yy", device, str(dtype), int(H))
    y = _CACHE.get(key)
    if y is None:
        # 0..100 스케일의 세로 좌표 벡터(1xHx1), 캐시
        if H <= 1:
            y = torch.zeros((1, H, 1), device=device, dtype=dtype)
        else:
            y = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
            y = (y / float(H - 1)) * 100.0
        _CACHE[key] = y
    return y

def _abs_band_mask(H: int, W: int, device, dtype, center: float, widthp: float, sigma: float):
    key = ("abs_mask", device, str(dtype), int(H), int(W), float(center), float(widthp), float(sigma))
    m = _CACHE.get(key)
    if m is not None:
        return m
    yy = _get_yy(H, device, dtype)  # 1xHx1
    if sigma > 0.0:
        band = torch.exp(-0.5 * ((yy - center) / sigma) ** 2)  # 1xHx1
        band = band.expand(1, H, W)                             # 1xHxW
    else:
        half = widthp * 0.5
        band = ((yy >= (center - half)) & (yy <= (center + half))).to(dtype)  # 1xHx1
        band = band.expand(1, H, W)
    _CACHE[key] = band
    return band

def _bbox_band_mask(alpha_1chw: torch.Tensor,
                    center: float,
                    width: float,
                    sigma: float,
                    down_h: int = 240) -> torch.Tensor:
    """
    bbox 정규화된 Y(0..100) 축에서 중심/폭/시그마로 밴드 마스크 생성.
    - alpha_1chw: [1,1,H,W] (0..1)
    반환: [1,H,W] (alpha와 동일 해상도, 값 0..1)
    """
    # 1) 준비
    a_full = alpha_1chw.squeeze(0).squeeze(0)  # [H,W]
    H, W = a_full.shape

    # 2) 다운스케일 후 bbox 탐지(성능↑)
    if down_h is not None and H > down_h:
        scale = float(down_h) / float(H)
        Wd = max(1, int(round(W * scale)))
        a_ds = F.interpolate(a_full.unsqueeze(0).unsqueeze(0),
                             size=(down_h, Wd), mode="nearest").squeeze()
    else:
        a_ds = a_full

    # 3) 활성 행(row) 인덱스 찾기 (bool → nonzero)
    rows_bool = (a_ds > 0.5).any(dim=1)  # [Hd] bool
    idx = torch.nonzero(rows_bool, as_tuple=False).flatten()

    # 4) bbox 없으면 전체 높이로 대체
    if idx.numel() == 0:
        y_top, y_bot = 0, H - 1
    else:
        y_top_ds = int(idx.min().item())
        y_bot_ds = int(idx.max().item())
        s = float(H) / float(a_ds.shape[0])  # 복원 스케일
        y_top = int(round(y_top_ds * s))
        y_bot = int(round(y_bot_ds * s))
        y_top = max(0, min(H - 1, y_top))
        y_bot = max(y_top + 1, min(H - 1, y_bot))  # 최소 1픽셀 이상 두께

    # 5) bbox 기준 0..100 정규화된 y좌표축
    yy = torch.arange(H, device=a_full.device, dtype=a_full.dtype).view(H, 1)
    denom = max(1.0, float(y_bot - y_top))
    y_norm = (yy - float(y_top)) / denom * 100.0  # [H,1]

    # 6) 가우시안/사각 밴드 생성 후 [1,H,W]로 확장
    if float(sigma) > 0.0:
        band_line = torch.exp(-0.5 * ((y_norm - float(center)) / float(sigma)) ** 2)  # [H,1]
        band_1hw = band_line.view(1, H, 1).expand(1, H, W)
    else:
        half = float(width) * 0.5
        band_line = ((y_norm >= (float(center) - half)) &
                     (y_norm <= (float(center) + half))).to(a_full.dtype)          # [H,1]
        band_1hw = band_line.view(1, H, 1).expand(1, H, W)

    return band_1hw


# --- util: color string -> tensor ---
def _parse_rgb(c, device=None, dtype=torch.float32, default=(0.0, 255.0, 255.0)) -> torch.Tensor:
    if isinstance(c, str):
        try:
            r, g, b = [float(x.strip()) for x in c.split(",")[:3]]
        except Exception:
            r, g, b = default
    elif isinstance(c, (tuple, list)):
        try:
            r, g, b = [float(c[0]), float(c[1]), float(c[2])]
        except Exception:
            r, g, b = default
    elif isinstance(c, torch.Tensor):
        t = c.flatten().to(torch.float32)
        if t.numel() < 3:
            t = torch.tensor(list(default), dtype=torch.float32)
        r, g, b = float(t[0].item()), float(t[1].item()), float(t[2].item())
    else:
        r, g, b = default

    # 2) 0..255 범위면 0..1로 정규화 (자동 감지)
    mx = max(abs(r), abs(g), abs(b))
    if mx > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0

    # 3) 텐서화 + clamp
    t = torch.tensor([r, g, b], dtype=dtype, device=device)
    return t.clamp(0.0, 1.0)

def _odd(k: int) -> int:
    k = int(k)
    if k < 1: return 1
    return k if (k % 2 == 1) else (k + 1)

def _gaussian_kernel_1d(sigma: float, dtype, device):
    radius = max(1, int(3.0 * float(sigma))); k = 2 * radius + 1
    x = torch.arange(k, device=device, dtype=torch.float32) - radius
    g = torch.exp(-0.5 * (x / float(sigma)) ** 2); g = g / g.sum()
    return g.to(dtype=dtype)

def _blur_alpha_gpu(alpha_1chw: torch.Tensor, sigma: float):
    if sigma <= 0: return alpha_1chw
    dtype = alpha_1chw.dtype; device = alpha_1chw.device
    g1d = _gaussian_kernel_1d(sigma, dtype=dtype, device=device)
    k = g1d.numel(); pad = k // 2
    kx = g1d.view(1,1,1,k); ky = g1d.view(1,1,k,1)
    x = F.pad(alpha_1chw, (pad, pad, 0, 0), mode="reflect"); x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect"); x = F.conv2d(x, ky)
    return x

def _erode_gpu(a_1chw: torch.Tensor, k: int):
    k = _odd(k)
    if k <= 1: return a_1chw
    pad = k // 2
    return 1.0 - F.max_pool2d(1.0 - a_1chw, kernel_size=k, stride=1, padding=pad)

@torch.no_grad()
def apply_energy(
    comp_rgb_3hw: torch.Tensor,     # 3xHxW (RGB 0..1)
    alpha_1chw: torch.Tensor,       # 1x1xHxW (0..1)
    *,
    color: str = "0,255,255",
    width: int = 3,
    blur: float = 12.0,
    gain: float = 0.85,
    on_bg: bool = True,
    pulse: bool = False,
    pulse_speed: float = 1.0,
    pulse_min: float = 0.7,
    pulse_max: float = 1.3,
    pulse_shape: str = "sine",
    # Y-band
    y_mode: str = "off",            # "off" | "abs" | "bbox"
    y_center: float = 50.0,
    y_width: float = 10.0,
    y_sigma: float = 0.0,
    y_bbox_down_h: int = 240,       # bbox 모드 계산 해상도

    aura: bool = False,
    aura_amp: float = 0.35,
    aura_freq: float = 1.6,
    aura_speed: float = 0.6,
    aura_twirl: float = 0.8,
    aura_rgb_phase: float = 0.0,
    aura_down: int = 1,
    aura_roi_thr: float = 0.0,

    **_,
) -> torch.Tensor:
    """
    comp_rgb_3hw + 윤곽선 glow를 합성해 반환.
    Y-band는 내부에서 캐시/저해상도 기반으로 계산하여 오버헤드 최소화.
    """
    _, H, W = comp_rgb_3hw.shape
    device, dtype = comp_rgb_3hw.device, comp_rgb_3hw.dtype

    # ── 에지(윤곽) 만들기
    w = max(1, int(width))
    edge_out_1chw = (_dilate_gpu(alpha_1chw, w) - alpha_1chw).clamp(0, 1)
    edge_soft = _blur_alpha_gpu(edge_out_1chw, float(max(0.0, blur))).squeeze(0)  # [1,H,W]
    edge_1hw  = edge_soft.squeeze(0)                                                # HxW

    # ── 게인(펄스)
    g = float(max(0.0, gain))
    if pulse:
        t_now = time.time()
        f = float(max(0.0, pulse_speed))
        if pulse_shape == "triangle":
            p = (t_now * f) % 1.0
            tri = 2.0 * (0.5 - abs(p - 0.5))      # 0..1..0
            g *= (pulse_min + (pulse_max - pulse_min) * tri)
        else:
            s = 0.5 * (math.sin(2.0 * math.pi * f * t_now) + 1.0)
            g *= (pulse_min + (pulse_max - pulse_min) * s)

    # ── 색 적용
    col = _parse_rgb(color, device=device, dtype=dtype).view(3,1,1)
    energy_rgb = (col * edge_1hw.unsqueeze(0)) * g         # 3xHxW

    # ── Aura modulation: 시간/공간 요동 + 소용돌이 위상
    if aura and aura_amp > 0.0:
        xx, yy = _get_xy(H, W, device, torch.float32)

        # 알파 가중 중심 (소용돌이 기준 및 마스크)
        a1 = alpha_1chw.squeeze(0).squeeze(0).to(torch.float32)     # [H,W]
        a_sum = a1.sum()
        if float(a_sum) > 1e-6:
            cx = (xx * a1).sum() / a_sum
            cy = (yy * a1).sum() / a_sum
        else:
            cx = torch.tensor(0.5, device=device, dtype=torch.float32)
            cy = torch.tensor(0.5, device=device, dtype=torch.float32)

        # 거리/각도 (링/소용돌이용)
        dx, dy = xx - cx, yy - cy
        r  = torch.sqrt(dx*dx + dy*dy + 1e-9)                      # [H,W]
        ang = torch.atan2(dy, dx)

        # 시간
        t = torch.tensor(time.time(), device=device, dtype=torch.float32)

        # ===== 모드 선택 =====
        # aura_twirl == 0 → "선형(물결) 모드"
        # aura_twirl != 0 → "원형 링 + 소용돌이" 모드
        if float(aura_twirl) == 0.0:
            s = xx  # 0..1, x축 진행파
            phase = 2.0 * math.pi * (float(aura_freq) * s - float(aura_speed) * t)
        else:
            phase = 2.0 * math.pi * (float(aura_freq) * r - float(aura_speed) * t) + float(aura_twirl) * ang

        # ===== 파형 생성 =====
        base = 1.0 + float(aura_amp) * torch.sin(phase)             # [H,W]
        if aura_rgb_phase and float(aura_rgb_phase) != 0.0:
            dph = 2.0 * math.pi * float(aura_rgb_phase)
            m0 = 1.0 + float(aura_amp) * torch.sin(phase + 0.0 * dph)
            m1 = 1.0 + float(aura_amp) * torch.sin(phase + 1.0 * dph)
            m2 = 1.0 + float(aura_amp) * torch.sin(phase + 2.0 * dph)
            mod = torch.stack([m0, m1, m2], dim=0)                  # [3,H,W]
        else:
            mod = base.unsqueeze(0).expand(3, -1, -1)               # [3,H,W]

        # 얇은 링/띠 옵션(선택): sin(phase)=0 부근만 살림
        use_thin_stripes = (int(aura_down) >= 2)
        if use_thin_stripes:
            sigma = 0.20 / float(max(1, int(aura_down)))           # 0.10~0.30 권장
            stripe = torch.exp(-0.5 * (torch.sin(phase) / sigma) ** 2)  # [H,W]
            stripe = stripe.unsqueeze(0).expand(3, -1, -1)          # [3,H,W]
            mod = 1.0 + (stripe - 0.5) * (2.0 * float(aura_amp))    # 0.5 기준 상하 진폭

        # 마스킹: 엣지 주변만
        edge_soft_local = edge_soft.clamp(0, 1)                     # [1,H,W]
        edge_mask = edge_soft_local
        mod = (mod * edge_mask).to(dtype)                           # dtype 정렬(fp16 등)

        # 색 적용 + 합성(곱셈 변조, 중간 clamp 금지)
        # k: 아우라 강도. 과발광 방지하면서 무빙 가시성 확보
        k = 0.6
        energy_rgb = energy_rgb * (1.0 + k * (mod - 1.0))

    # ── BG 제한
    if on_bg:
        alpha_1hw = alpha_1chw.squeeze(0)                  # 1xHxW
        energy_rgb = energy_rgb * (1.0 - alpha_1hw)

    # ── Y-band (고속/캐시)
    y_mode = (y_mode or "off").lower()
    if y_mode != "off":
        if y_mode == "abs":
            band = _abs_band_mask(H, W, device, dtype, float(y_center), float(y_width), float(y_sigma))  # 1xHxW
        elif y_mode == "bbox":
            band = _bbox_band_mask(alpha_1chw, float(y_center), float(y_width), float(y_sigma),
                                   down_h=int(y_bbox_down_h))                                           # 1xHxW
        else:
            band = None
        if band is not None:
            energy_rgb = energy_rgb * band

    return (comp_rgb_3hw + energy_rgb).clamp(0.0, 1.0)
