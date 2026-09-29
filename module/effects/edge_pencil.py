# -*- coding: utf-8 -*-
"""
색연필 아웃라인 이펙트 (고속화 버전, 최소패치: 업샘플 안전화)
- comp_rgb_3hw [3,H,W] (0..1), alpha_1hw [1,H,W] (0..1) → [3,H,W] (0..1)

개선 포인트
- 내부 마스크 연산 저해상도에서 수행 후 업샘플
- Sobel/대시 마스크 캐시
- FP16 + autocast + channels_last 친화
- --pencil-gap-len<0 → 알파 경계 바깥쪽 오프셋 띠 지원
- 무지개 팔레트(가로/세로 밴드) 옵션 추가
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple

# ── CPU 환경 대비: nullcontext ────────────────────────────────────────────────
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False

# ── 전역 캐시 ─────────────────────────────────────────────────────────────────
_CACHED = {
    "kx": None, "ky": None,  # Sobel 커널
    # dash 마스크 캐싱 키: ("dash", Ws, dash_len, gap_len) -> tensor
}

def _ensure_kernels(device, dtype):
    if _CACHED["kx"] is None or _CACHED["kx"].device != device:
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
        _CACHED["kx"], _CACHED["ky"] = kx, ky
    return _CACHED["kx"], _CACHED["ky"]

def _get_dash_mask(Ws: int, dash_len: int, gap_len: int, device, dtype):
    key = ("dash", Ws, dash_len, gap_len)
    if _CACHED.get(key) is None:
        period = max(2, int(dash_len) + max(1, int(gap_len)))
        xx = torch.arange(Ws, device=device, dtype=dtype).view(1,1,1,Ws)
        mask = ((xx % period) < int(dash_len)).float()
        _CACHED[key] = mask
    return _CACHED[key]

# ── 업샘플 안전 보조(핵심 최소패치) ───────────────────────────────────────────
def _upsample_2d_1ch(x: torch.Tensor, size_hw: tuple[int,int], mode="bilinear", align_corners=False):
    """
    입력 x: [H,W] or [1,H,W] → 출력: [1,H,W] (size_hw로 보간)
    """
    if x.dim() == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif x.dim() == 3 and x.size(0) == 1:
        x4 = x.unsqueeze(0)               # [1,1,H,W]
    else:
        raise ValueError(f"_upsample_2d_1ch expects [H,W] or [1,H,W], got {tuple(x.shape)}")
    y4 = F.interpolate(x4, size=size_hw, mode=mode, align_corners=align_corners)
    y = y4.squeeze(0)  # [1,H,W]
    return y

# ── 본 함수 ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def draw_pencil_outline(
    comp_rgb_3hw: torch.Tensor,
    alpha_1hw: torch.Tensor,
    *,
    # on/off
    pencil: int = 1,
    # 색상
    pencil_color: str = "255,128,0",
    pencil_color2: str | None = None,
    pencil_multicolor: int = 0,     # 0 단색, 1 노이즈로 color↔color2 혼합
    # ▼ 추가: 팔레트/밴드 옵션
    pencil_palette: str = "",       # "" | "rainbow"
    pencil_bands: int = 7,          # 무지개 구간 수(기본 7색)
    pencil_repeat: int = 1,         # 무지개 반복 횟수(가로/세로로 반복)
    pencil_axis: str = "x",         # "x"=세로줄(가로 진행), "y"=가로줄(세로 진행)
    # 형태/강도
    pencil_width: int = 3,          # 스트로크 굵기(px)
    pencil_soft: int = 1,           # 스트로크 가장자리 소프트닝(픽셀, 0=하드)
    pencil_gain: float = 1.0,       # 에지 강도 스케일
    pencil_opacity: float = 0.8,    # 최종 불투명도(0..1 권장)
    pencil_thresh: float = 0.08,    # 에지 감지 임계(낮을수록 민감)
    pencil_rough: float = 0.7,      # 질감 거칠기(0..1)
    pencil_jitter: float = 0.8,     # 선 굴곡/떨림(0..1)
    # 점선/오프셋
    pencil_dash: int = 0,           # 0 실선 / 1 점선
    pencil_dash_len: int = 12,      # 점선 길이(px)
    pencil_gap_len: int = 8,        # 점선 간격(px). 음수면 '알파 바깥쪽 띠 오프셋(px)'로 사용
    # 적용 영역
    pencil_on_bg: int = 1,          # 배경(=알파 밖) 쪽에 그림
    pencil_on_fg: int = 1,          # 전경 윤곽에도 약간 겹치기
    # 블렌딩
    pencil_blend: str = "alpha",    # alpha | add | screen | overlay

    # ── [추가] 해칭 온/오프 및 파라미터 ───────────────────────────────────
    pencil_hatch: int = 0,            # 0=off, 1=on
    pencil_hatch_mode: str = "normal",# global | tangent | normal
    pencil_hatch_angle: float = 30.0, # global일 때 각도
    pencil_hatch_period: float = 6.0, # px
    pencil_hatch_width: float = 2.0,  # px
    pencil_hatch_phase: float = 0.0,  # 0~1
    pencil_hatch_jitter: float = 0.30,

    pencil_sweep_prog: float | None = None,   # 0..1 진행도(None이면 비활성)
    pencil_sweep_soft_px: int = 24,           # 경계 소프트닝(px)
    pencil_sweep_axis: str = "y", 
    main_person_mask_1hw: torch.Tensor | None = None,
) -> torch.Tensor:

    if not pencil:
        return comp_rgb_3hw

    # 기본 체크/준비
    assert comp_rgb_3hw.dim() == 3 and comp_rgb_3hw.shape[0] == 3, "comp_rgb_3hw: [3,H,W]"
    assert alpha_1hw.dim() == 3 and alpha_1hw.shape[0] == 1, "alpha_1hw: [1,H,W]"
    C, H, W = comp_rgb_3hw.shape
    device = comp_rgb_3hw.device
    dtype = comp_rgb_3hw.dtype

    # 내부 마스크 연산 해상도 축소비 (자동)
    if H >= 2160 or W >= 3840:
        ds = 4
    elif H >= 1080 or W >= 1920:
        ds = 2
    else:
        ds = 1

    # 채널라스트 친화 유지(가능하면)
    try:
        comp_rgb_3hw = comp_rgb_3hw.contiguous(memory_format=torch.channels_last)
    except Exception:
        pass

    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x * x * (3.0 - 2.0 * x)

    def _make_sweep_mask(H: int, W: int, prog01: float, soft_px: int,
                     axis: str, device, dtype):
        prog = float(max(0.0, min(1.0, prog01)))
        soft_px = int(max(0, soft_px))

        if axis.lower() == "x":
            # 좌→우
            xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
            x_norm = xx / max(1.0, float(W))
            cut = prog
            if soft_px <= 0:
                m = (x_norm <= cut).float()             # [1,W]
            else:
                width = float(soft_px) / max(1.0, float(W))
                t = (cut - x_norm) / max(width, 1e-6)
                t = torch.clamp(t, 0.0, 1.0)
                m = t*t*(3.0 - 2.0*t)
            m = m.expand(H, W).unsqueeze(0)             # [1,H,W]
        else:
            # 기본: 위→아래(y)
            yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
            y_norm = yy / max(1.0, float(H))
            cut = prog
            if soft_px <= 0:
                m = (y_norm <= cut).float()             # [H,1]
            else:
                width = float(soft_px) / max(1.0, float(H))
                t = (cut - y_norm) / max(width, 1e-6)
                t = torch.clamp(t, 0.0, 1.0)
                m = t*t*(3.0 - 2.0*t)
            m = m.expand(H, W).unsqueeze(0)             # [1,H,W]

        return m.to(dtype=dtype)


    # ── 색 파서
    def _parse_color(s: str) -> torch.Tensor:
        try:
            r, g, b = [int(float(x.strip())) for x in s.split(",")[:3]]
        except Exception:
            r, g, b = 255, 128, 0
        r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
        return torch.tensor([r, g, b], device=device, dtype=dtype).view(3,1,1) / 255.0

    # ── 커널 캐시
    kx, ky = _ensure_kernels(device, dtype)

    # ── 저해상도 공간으로 다운스케일
    Hs, Ws = max(8, H//ds), max(8, W//ds)
    a_full = alpha_1hw.clamp(0, 1)
    a = F.interpolate(a_full.unsqueeze(0), size=(Hs, Ws), mode="bilinear", align_corners=False).squeeze(0)  # [1,Hs,Ws]

    # ── 에지(Sobel) @ low-res
    ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
    with ctx:
        gx = F.conv2d(a.unsqueeze(0), kx, padding=1)
        gy = F.conv2d(a.unsqueeze(0), ky, padding=1)
        grad = torch.sqrt(gx*gx + gy*gy).squeeze(0)  # [1,Hs,Ws]
        grad = grad / (grad.max().clamp_min(1e-6))
        edges = (grad * float(pencil_gain)).clamp(0, 1)  # [1,Hs,Ws]

        hard = (edges >= float(pencil_thresh)).float()
        soft = edges.clamp_min(float(pencil_thresh))
        soft = ((soft - float(pencil_thresh)) / (1.0 - float(pencil_thresh))).clamp(0, 1)
        edge_mask = (hard * 0.5 + soft * 0.5)  # [1,Hs,Ws]

        # 폭/소프트(저해상도에서 1~2회만)
        if int(pencil_width) > 1:
            k = int(max(1, pencil_width // ds))
            if k > 0:
                edge_mask = F.max_pool2d(edge_mask, kernel_size=2*k+1, stride=1, padding=k)
        if int(pencil_soft) > 0:
            edge_mask = F.avg_pool2d(edge_mask, kernel_size=3, stride=1, padding=1)

        # 노이즈 1회 + 지터/거칠기
        noise = torch.rand((1,1,Hs,Ws), device=device, dtype=dtype)
        noise = F.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
        jitter_amp = (0.15 + 0.85 * float(pencil_jitter))
        rough_amp  = (0.25 + 0.75 * float(pencil_rough))
        jitter = 1.0 + (noise * 2.0 - 1.0) * 0.6 * jitter_amp
        rough  = (noise * 0.8 + 0.2) * rough_amp

        stroke = (edge_mask * jitter).clamp(0, 1)
        stroke = (stroke * rough).clamp(0, 1)  # [1,Hs,Ws]

        # 점선(옵션)
        if int(pencil_dash):
            dash = _get_dash_mask(
                Ws,
                max(1, int(pencil_dash_len) // ds),
                max(1, int(pencil_gap_len) // ds) if int(pencil_gap_len) > 0 else 1,
                device, dtype
            )
            stroke = (stroke * dash).clamp(0, 1)

        # 오프셋 링(알파에서 바깥/안쪽으로 띠 이동)
        bin_a = (a >= 0.5).float()
        width_scaled = max(1, int(pencil_width // ds))
        gap = int(pencil_gap_len)
        if gap < 0:
            # 바깥쪽으로 |gap|
            off = int((-gap) // ds)
            outer0 = F.max_pool2d(bin_a, kernel_size=2*off+1, stride=1, padding=off)
            outer1 = F.max_pool2d(bin_a, kernel_size=2*(off+width_scaled)+1, stride=1, padding=(off+width_scaled))
            ring_mask = (outer1 - outer0).clamp(0, 1)
        elif gap == 0:
            # 경계 바로 바깥쪽 띠
            ring_mask = (F.max_pool2d(bin_a, kernel_size=2*width_scaled+1, stride=1, padding=width_scaled) - bin_a).clamp(0, 1)
        else:
            # 양수 gap은 '점선 간격' 의미 → 오프셋 제한 없음
            ring_mask = torch.ones_like(bin_a)
        stroke = (stroke * ring_mask).clamp(0, 1)  # [1,Hs,Ws]

        # ── [추가] 법선/접선/글로벌 해칭(저해상도에서 생성) ─────────────────
        if int(pencil_hatch):
            # 저해상도 스케일 보정(내부는 Hs×Ws라서 px도 ds로 나눔)
            period_s = max(1.0, float(pencil_hatch_period) / float(ds))
            width_s  = max(1.0, float(pencil_hatch_width)  / float(ds))

            # gx, gy는 이미 저해상도에서 구해둠
            hatch_lo = _make_stripes(
                H=Hs, W=Ws, device=device, dtype=dtype,
                mode=str(pencil_hatch_mode).lower(),
                angle_deg=float(pencil_hatch_angle),
                period_px=period_s, line_width_px=width_s,
                phase=float(pencil_hatch_phase),
                gx=gx, gy=gy,
                jitter=float(pencil_hatch_jitter)
            )  # [1,1,Hs,Ws]

            # 링/점선으로 만든 stroke에 해칭을 곱해 결 방향을 부여
            stroke = (stroke * hatch_lo).clamp(0, 1)

    # 고해상도로 업샘플 (안전보조 사용)
    stroke_hi = _upsample_2d_1ch(stroke.squeeze(0), (H, W), mode="bilinear", align_corners=False)  # [1,H,W]

    # 적용 영역(고해상도에서)
    a_hi = a_full  # [1,H,W]
    bg_mask = (1.0 - a_hi).clamp(0, 1) if int(pencil_on_bg) else torch.zeros_like(a_hi)

    # 전경 윤곽 감쇠(저해상도 edges를 업샘플: 안전보조 사용)
    fg_edge = _upsample_2d_1ch(edges.squeeze(0), (H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    fg_mask = (fg_edge * 0.6) if int(pencil_on_fg) else torch.zeros_like(a_hi)

    region = (bg_mask + fg_mask).clamp(0, 1)
    stroke_hi = (stroke_hi * region).clamp(0, 1)  # [1,H,W]

    # ★ 추가: main_person_mask_1hw가 있으면 그 영역에만 크레파스 적용
    if main_person_mask_1hw is not None:
        try:
            m = main_person_mask_1hw
            # [1,1,H,W] 또는 [1,H,W] 모두 허용
            if m.dim() == 4:
                m = m.squeeze(0)        # [1,H,W]
            if m.dim() == 2:
                m = m.unsqueeze(0)      # [1,H,W]
            m = m.to(device=stroke_hi.device, dtype=stroke_hi.dtype)
            stroke_hi = (stroke_hi * m).clamp(0, 1)
        except Exception:
            # 마스크 이상하면 전체 적용
            pass

    # ── 색상/멀티컬러/팔레트 ─────────────────────────────────────────────────
    colA = _parse_color(pencil_color)  # [3,1,1]

    use_rainbow = (str(pencil_palette).lower() == "rainbow")
    if use_rainbow:
        # ① 팔레트 정의
        palette = torch.tensor([
            [246,118,190],  # 0 hot pink (start)
            [240, 90,210],  # 1 magenta-pink
            [226,140,248],  # 2 light magenta
            [172,130,250],  # 3 violet
            [160,185,255],  # 4 periwinkle pastel
            [188,220,255],  # 5 baby-blue pastel
            [208,160,245],  # 6 lavender
            [224,172,196],  # 7 mauve/rose-beige (lavender→peach bridge)
            [240,186,140],  # 8 peach→amber bridge
            [248,150,160],  # 9 peach-pink (end, loops back to 0 hot pink)
        ], device=device, dtype=dtype) / 255.0

        PALETTE_RAINBOW = [
            [246,118,190],  # 0 hot pink (start)
            [240, 90,210],  # 1 magenta-pink
            [226,140,248],  # 2 light magenta
            [172,130,250],  # 3 violet
            [160,185,255],  # 4 periwinkle pastel
            [188,220,255],  # 5 baby-blue pastel
            [208,160,245],  # 6 lavender
            [224,172,196],  # 7 mauve/rose-beige (lavender→peach bridge)
            [240,186,140],  # 8 peach→amber bridge
            [248,150,160],  # 9 peach-pink (end, loops back to 0 hot pink)
        ]

        PALETTE_PINK_HEAVY = [
            [255,212,224],  # 0 baby pink (start)
            [255,196,210],  # 1 blush
            [255,178,196],  # 2 light peach-pink
            [252,160,186],  # 3 salmon pink
            [246,140,180],  # 4 rose pink
            [242,128,184],  # 5 bubblegum pink
            [246, 90,190],  # 6 hot pink (짧게 스침)
            [238, 72,192],  # 7 vivid fuchsia → 금방 내려옴
            [228,100,208],  # 8 orchid
            [220,120,220],  # 9 pink-violet
            [212,150,230],  # 10 lavender-pink (부드럽게 완화)
            [236,170,200],  # 11 dusty rose (복귀 브릿지)
            [255,208,222],  # 12 soft baby pink (loop to 0)
        ]

        # 머리 꼭대기색 지정
        target_idx = 1   # (예) 핫핑크가 palette[6]일 때
        #rainbow 일 때 1 / pink heavy 일 때 6

        nb = palette.size(0)
        reps = max(1, int(pencil_repeat))
        total_segs = nb * reps

        import math
        two_pi = 2.0 * math.pi

        band_blend_frac = 1  # 경계 블렌드 폭(0=하드, 0.15~0.35 권장)

        axis = (pencil_axis or "x").lower()

        if axis == "around":
            with torch.no_grad():
                a_bin = (a_full >= 0.5).float()
                if a_bin.sum() < 1:
                    a_bin = (a_hi >= 0.5).float()

                yy = torch.arange(H, device=device, dtype=torch.float32)
                xx = torch.arange(W, device=device, dtype=torch.float32)
                Y, X = torch.meshgrid(yy, xx, indexing="ij")
                Y = Y.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
                X = X.unsqueeze(0).unsqueeze(0)

                # mass = a_bin.squeeze(0)                            # [H,W]
                # msum = mass.sum().clamp_min(1.0)
                # yc = (Y * mass).sum() / msum
                # xc = (X * mass).sum() / msum
                yc = torch.tensor(float(H //2 ), device=device, dtype=torch.float32)  # 바닥 줄
                xc = torch.tensor(float(W) / 2.0, device=device, dtype=torch.float32)
                y_top = torch.tensor(float(H//2), device=device, dtype=torch.float32)
                x_top = torch.tensor(float(W//2), device=device, dtype=torch.float32)

                # top_row = (mass > 0).any(dim=1).nonzero(as_tuple=False)
                # if top_row.numel() > 0:
                #     y_top = int(top_row[0].item())
                #     xs_in_top = (mass[y_top] > 0).nonzero(as_tuple=False)
                #     x_top = int(xs_in_top[:,0].float().mean().item()) if xs_in_top.numel() > 0 else int(xc.item())
                # else:
                #     y_top = int(max(0, yc.item()))
                #     x_top = int(max(0, xc.item()))

                dy = (Y - yc)
                dx = (X - xc)
                phi = -torch.atan2(dy, dx)                         # (-π,π]
                dy0  = (y_top - yc)
                dx0  = (x_top - xc)
                phi0 = -torch.atan2(dy0, dx0)

                two_pi   = 2.0 * math.pi
                phi_shift= (phi - phi0) % two_pi

                seg_size = two_pi / float(total_segs)
                t = (phi_shift / max(1e-6, seg_size)).to(dtype)  # 이후 보간용 연속 좌표


        elif axis == "y":
            coord = torch.arange(H, device=device, dtype=dtype).view(H, 1)     # [H,1]
            t = coord / max(1.0, (H / float(total_segs)))     # [1,1,H,1]
            t = t.expand(1,1,H,W)                                          # [H,W]

        else:  # axis == "x"
            coord = torch.arange(W, device=device, dtype=dtype).view(1, W)     # [1,W]
            t = coord / max(1.0, (W / float(total_segs)))     # [1,1,1,W]
            t = t.expand(1,1,H,W)                                          # [H,W]

        k = torch.floor(t).long()                  # 현재 밴드 인덱스
        frac = (t - k.float()).clamp(0, 1)         # [1,1,H,W], 0..1

        

        # 12시 좌표(화면 맨 위 중앙)의 현재 팔레트 인덱스 구하기
        y12, x12 = 0, W // 2
        k_top = int(torch.floor(t[..., y12, x12]) % nb)  # 현재 12시가 가리키는 인덱스

        shift = (target_idx - k_top) % nb

        # k에만 쉬프트 주면 팔레트 배열은 그대로, 색만 회전
        k = k + shift

        # smoothstep으로 경계 부드럽게 (band_blend_frac만큼만 섞이게)
        a = 1.0 - float(band_blend_frac)
        s = ((frac - a) / max(1e-6, (1.0 - a))).clamp(0, 1)   # [1,1,H,W]
        s = s * s * (3.0 - 2.0 * s)                           # smoothstep

        # 팔레트에서 k, k+1 색 뽑기
        k0 = torch.remainder(k, nb)            # [1,1,H,W]
        k1 = torch.remainder(k + 1, nb)

        # 인덱싱 편의상 [H,W]로 맞추기
        k0_ = k0.squeeze(0).squeeze(0)         # [H,W]
        k1_ = k1.squeeze(0).squeeze(0)         # [H,W]

        col0_hw3 = palette[k0_]                # [H,W,3]
        col1_hw3 = palette[k1_]                # [H,W,3]

        # 보간해서 최종 색 만들기 → [3,H,W]
        s3 = s.squeeze(0)                      # [1,H,W]
        col_hw3 = col0_hw3 * (1.0 - s3.permute(1,2,0)) + col1_hw3 * s3.permute(1,2,0)  # [H,W,3]
        col = col_hw3.permute(2,0,1).contiguous()  # [3,H,W]

        # ③ 크레파스 질감에 어울리게 아주 약간의 노이즈 섞기(선택)
        mix_noise = _upsample_2d_1ch(noise.squeeze(0), (H, W), mode="bilinear", align_corners=False)  # [1,H,W]
        col = (col * (1.0 - 0.08 * mix_noise) + col.roll(shifts=1, dims=2) * (0.08 * mix_noise)).clamp(0, 1)

    else:
        if int(pencil_multicolor) and (pencil_color2 not in (None, "", "None")):
            colB = _parse_color(str(pencil_color2))  # [3,1,1]
            mix = _upsample_2d_1ch(noise.squeeze(0), (H, W), mode="bilinear", align_corners=False)  # [1,H,W]
            col = colA * (1.0 - mix) + colB * mix    # [3,1,H,W]
            col = col.squeeze(1)                      # → [3,H,W]
        else:
            col = colA  # [3,1,1] (브로드캐스트 OK)

    # ── 블렌딩 ────────────────────────────────────────────────────────────────
    base = comp_rgb_3hw.clamp(0, 1)  # [3,H,W]
    sA = (stroke_hi * float(pencil_opacity)).clamp(0, 1)  # [1,H,W]

    if col.dim() == 3 and col.shape[1] == 1 and col.shape[2] == 1:
        paint = (col.expand_as(base)) * sA  # [3,H,W]
    else:
        # col이 [3,H,W]인 경우(팔레트/멀티컬러 분기)
        paint = col * sA  # [3,H,W]

    mode = (pencil_blend or "alpha").lower()
    if mode == "add":
        out = (base + paint).clamp(0, 1)
    elif mode == "screen":
        out = 1.0 - (1.0 - base) * (1.0 - paint)
        out = out.clamp(0, 1)
    elif mode == "overlay":
        mask = (base <= 0.5).float()
        out = base * (mask * (1 - 2*(1 - paint)*(1 - base)) + (1 - mask) * (2*paint))
        out = out.clamp(0, 1)
    else:
        out = base * (1.0 - sA) + paint

    # ── 스윕 적용(선택) : m=1이면 '연필 적용', 0이면 '원본 유지'
    if (pencil_sweep_prog is not None) and (0.0 <= float(pencil_sweep_prog) <= 1.0):
        m = _make_sweep_mask(
            H, W,
            prog01=float(pencil_sweep_prog),
            soft_px=int(pencil_sweep_soft_px),
            axis=str(pencil_sweep_axis),
            device=out.device,
            dtype=out.dtype,
        )  # [1,H,W]
        base = comp_rgb_3hw.clamp(0, 1)       # 스윕 밖은 원본 유지
        out  = out * m + base * (1.0 - m)     # 브로드캐스팅으로 [3,H,W]에 적용

    return out.clamp(0, 1)



# ── 해칭 스트라이프 생성(저해상도에서 만들고 업샘플) ────────────────────────
def _make_stripes(
    H: int, W: int, device, dtype,
    *, mode: str = "global",        # global | tangent | normal
    angle_deg: float = 30.0,        # mode=global일 때만 사용
    period_px: float = 6.0,
    line_width_px: float = 2.0,
    phase: float = 0.0,
    gx: torch.Tensor | None = None, # [1,1,H,W] (저해상도)
    gy: torch.Tensor | None = None, # [1,1,H,W]
    jitter: float = 0.25
) -> torch.Tensor:
    """
    return: [1,1,H,W] 0..1 (밝은 선)
    주의: '스트라이프의 진행 방향'은 벡터(cx,sy)에 수직. (평면파 φ= X*cx + Y*sy)
         → mode='normal'일 때 '법선에 평행한 선'을 얻으려면 '접선 벡터'로 진행시켜야 함.
           그래서 아래 구현에서는:
             - mode='normal'  → 진행벡터=tangent  (선은 법선과 평행)
             - mode='tangent' → 진행벡터=normal   (선은 접선과 평행)
    """
    yy = torch.arange(H, device=device, dtype=dtype)   # 1D
    xx = torch.arange(W, device=device, dtype=dtype)   # 1D
    Y, X = torch.meshgrid(yy, xx, indexing="ij")       # [H,W]
    X = X.unsqueeze(0).unsqueeze(0)                    # [1,1,H,W]
    Y = Y.unsqueeze(0).unsqueeze(0)

    if mode == "global":
        import math
        th = math.radians(float(angle_deg))
        cx, sy = math.cos(th), math.sin(th)
        s = (X*cx + Y*sy) / max(1e-6, float(period_px)) + float(phase)
    else:
        assert gx is not None and gy is not None, "gx/gy required for tangent/normal"
        mag = (gx*gx + gy*gy + 1e-6).sqrt()
        nx, ny = gx/mag, gy/mag  # 법선 단위벡터

        if mode == "normal":
            # 선이 '법선과 평행'해 보이도록 → 진행벡터는 '접선'을 사용
            cx, sy = -ny, nx
        else: # mode == "tangent"
            # 선이 '접선과 평행'해 보이도록 → 진행벡터는 '법선'을 사용
            cx, sy = nx, ny

        s = (X*cx + Y*sy) / max(1e-6, float(period_px)) + float(phase)

    frac = torch.remainder(s, 1.0)
    dist = torch.minimum(frac, 1.0-frac)                # 0..0.5
    dpx  = dist * max(1e-6, float(period_px))

    # 가우시안 라인(안티앨리어싱 느낌)
    import math
    sigma = max(0.5, float(line_width_px)) / 2.355
    line = torch.exp(-(dpx*dpx) / (2*sigma*sigma))      # [1,1,H,W]

    if jitter > 0:
        n = torch.rand_like(line)
        n = F.avg_pool2d(n, 3, 1, 1)
        line = (line * (0.8 + jitter*n)).clamp(0,1)

    return line



