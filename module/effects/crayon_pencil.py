# # -*- coding: utf-8 -*-
# """
# Crayon(벅벅 칠) + Rainbow + Pencil Outline (Torch, GPU 친화, 단일 파일)
# - 입력
#   comp_rgb_3hw: [3,H,W] (0..1)
#   alpha_1hw   : [1,H,W] (0..1)
# - 출력
#   [3,H,W] (0..1)

# 핵심:
# - 크레파스: 경계 띠를 픽셀별 두께(노이즈)로 흔들어 '모양' 자체를 들쑥날쑥화 + 스탬프 질감
# - 무지개: 두 효과 모두 축/구간/반복을 지정해 [3,H,W] 컬러 필드로 브로드캐스트
# """

# from __future__ import annotations
# from typing import Tuple, Optional
# import math
# import torch
# import torch.nn.functional as F

# # ───────────────────────── 공통 유틸/캐시 ─────────────────────────
# _CACHED = {"kx": None, "ky": None}

# class nullcontext:
#     def __enter__(self): return None
#     def __exit__(self, *exc): return False

# def _ensure_kernels(device, dtype):
#     if _CACHED["kx"] is None or _CACHED["kx"].device != device:
#         kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
#                           device=device, dtype=dtype).view(1,1,3,3)
#         ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
#                           device=device, dtype=dtype).view(1,1,3,3)
#         _CACHED["kx"], _CACHED["ky"] = kx, ky
#     return _CACHED["kx"], _CACHED["ky"]

# def _upsample_2d_1ch(x: torch.Tensor, size_hw: Tuple[int,int], mode="bilinear", align_corners=False):
#     if x.dim() == 2:
#         x4 = x[None,None,...]
#     elif x.dim() == 3 and x.size(0) == 1:
#         x4 = x[None,...]
#     else:
#         raise ValueError(f"_upsample_2d_1ch expects [H,W] or [1,H,W], got {tuple(x.shape)}")
#     y = F.interpolate(x4, size=size_hw, mode=mode, align_corners=align_corners).squeeze(0)
#     return y  # [1,H,W]

# def _parse_rgb(s: str, device, dtype):
#     try:
#         r,g,b = [float(v) for v in str(s).split(",")[:3]]
#     except Exception:
#         r,g,b = 255.0,128.0,0.0
#     t = torch.tensor([r,g,b], device=device, dtype=dtype).view(3,1,1)
#     if t.max() > 1: t = t/255.0
#     return t.clamp(0,1)

# # ───────────────────────── 간단 형태학/블러 ─────────────────────────
# def _dilate(x_1chw: torch.Tensor, k: int) -> torch.Tensor:
#     if k <= 0: return x_1chw
#     pad = k
#     return F.max_pool2d(x_1chw, kernel_size=2*k+1, stride=1, padding=pad)

# def _erode(x_1chw: torch.Tensor, k: int) -> torch.Tensor:
#     if k <= 0: return x_1chw
#     return 1.0 - _dilate(1.0 - x_1chw, k)

# def _gauss1d(sigma: float, dtype, device):
#     sigma = max(0.0, float(sigma))
#     if sigma == 0: return torch.tensor([1.0], device=device, dtype=dtype)
#     r = max(1, int(3.0*sigma))
#     xs = torch.arange(-r, r+1, device=device, dtype=dtype)
#     ker = torch.exp(-(xs*xs)/(2*sigma*sigma))
#     ker = ker/ker.sum()
#     return ker

# def _blur_1ch(x_1chw: torch.Tensor, sigma: float) -> torch.Tensor:
#     if sigma <= 0: return x_1chw
#     c = x_1chw
#     device, dtype = c.device, c.dtype
#     k = _gauss1d(sigma, dtype, device)
#     kx = k.view(1,1,1,-1)
#     ky = k.view(1,1,-1,1)
#     c = F.conv2d(c, kx, padding=(0, kx.size(-1)//2))
#     c = F.conv2d(c, ky, padding=(ky.size(-2)//2, 0))
#     return c

# # ───────────────────────── Rainbow 필드 생성 ─────────────────────────
# def _rainbow_field(H:int, W:int, device, dtype,
#                    bands:int=7, repeat:int=1, axis:str="x") -> torch.Tensor:
#     # 부드러운 핑크-스카이-블루-보라 톤(요청 스타일) 7~12 톤 중 선택
#     base = torch.tensor([
#         [255,165,100],[255,145,160],[245,130,205],[220,120,235],
#         [150,190,255],[120,180,255],[150,140,250],[185,130,240],
#         [220,125,230],[245,140,205],[255,165,170],[255,190,120],
#     ], device=device, dtype=dtype)/255.0
#     nb = int(max(2, min(bands, base.size(0))))
#     reps = int(max(1, repeat))
#     total = nb*reps

#     if (axis or "x").lower()=="y":
#         coord = torch.arange(H, device=device, dtype=dtype).view(H,1)
#         seg = (coord/ max(1.0, H/float(total))).floor().long()%nb
#         seg = seg.expand(H,W)
#     else:
#         coord = torch.arange(W, device=device, dtype=dtype).view(1,W)
#         seg = (coord/ max(1.0, W/float(total))).floor().long()%nb
#         seg = seg.expand(H,W)

#     col_hw3 = base[:nb][seg]           # [H,W,3]
#     return col_hw3.permute(2,0,1).contiguous()  # [3,H,W]

# # ───────────────────────── 크레파스(벅벅 칠) ─────────────────────────
# def _rand_smooth_noise(h, w, device, dtype, k:int=13):
#     n = torch.rand((1,1,h,w), device=device, dtype=dtype)
#     if k>1:
#         n = F.avg_pool2d(n, k, stride=1, padding=k//2)
#     return n.clamp(0,1)

# def _ragged_band(alpha_1chw: torch.Tensor, width:int, jitter:float) -> torch.Tensor:
#     """경계 띠를 픽셀별 두께로 흔들어 외형을 거칠게 만든다. [1,1,H,W]"""
#     dev, dt = alpha_1chw.device, alpha_1chw.dtype
#     e = (alpha_1chw - _erode(alpha_1chw, 1)).clamp(0,1)               # 얇은 경계
#     H, W = e.shape[-2:]
#     noise = _rand_smooth_noise(H, W, dev, dt, k=13)                    # 0..1
#     W0 = float(max(1, width))
#     rad = W0 + float(jitter)*(noise*2-1)                               # 픽셀별 허용 두께

#     K = max(2, int(W0 + abs(jitter)))
#     layers = []
#     cur = e
#     for _ in range(K):
#         cur = _dilate(cur, 1)
#         layers.append(cur)
#     stack = torch.stack(layers, dim=0)             # [K,1,1,H,W]
#     idx = (rad.long().clamp(1, K)-1).view(1,1,H,W)
#     gather = torch.gather(stack.squeeze(1), 0, idx)                    # [1,H,W]
#     rag = (gather - e.squeeze(1)).clamp(0,1).unsqueeze(1)              # [1,1,H,W]
#     return rag

# def draw_crayon_fill(
#     comp_rgb_3hw: torch.Tensor,      # [3,H,W] 0..1
#     alpha_1hw: torch.Tensor,         # [1,H,W] 0..1
#     *,
#     enable: bool = True,
#     width: int = 14,                 # 띠 두께(px)
#     jitter: float = 6.0,             # 외형 들쑥 강도(px)
#     density: float = 0.75,           # 스탬프 밀도
#     stamp: int = 8,                  # 스탬프 반경(페더 σ에 반영)
#     hardness: float = 0.6,           # 0=부드러움, 1=딱딱함
#     grain: float = 0.6,              # 질감 대비
#     on_bg: bool = True,              # 전경 외부에만
#     blend: str = "overlay",          # add | screen | multiply | overlay | alpha
#     # 색상
#     color: str = "255,120,255",
#     palette: str = "rainbow",        # "rainbow" | ""(단색)
#     bands: int = 7, repeat: int = 1, axis: str = "x",
# ) -> torch.Tensor:
#     if not enable: return comp_rgb_3hw

#     base = comp_rgb_3hw.clamp(0,1)
#     dev, dt = base.device, base.dtype
#     H, W = base.shape[-2:]

#     band = _ragged_band(alpha_1hw, int(width), float(jitter))          # [1,1,H,W]

#     # 스탬프·질감
#     dmask = (torch.rand((1,1,H,W), device=dev, dtype=dt) < float(density)).float()
#     texture = _rand_smooth_noise(H, W, dev, dt, k=9).pow(0.5)
#     paint_mask = (band * dmask * (0.5 + float(grain)*texture)).clamp(0,1)  # [1,1,H,W]
#     if stamp>1:
#         sigma = max(0.01, stamp*(1.0-float(hardness)+0.01))
#         paint_mask = _blur_1ch(paint_mask, sigma)

#     if on_bg:
#         paint_mask = paint_mask * (1.0 - alpha_1hw)

#     # 색상: 무지개 or 단일
#     if str(palette).lower()=="rainbow":
#         col = _rainbow_field(H, W, dev, dt, bands=int(bands), repeat=int(repeat), axis=str(axis))
#         # 약간의 색 노이즈 블렌드
#         n = _rand_smooth_noise(H, W, dev, dt, k=7)
#         col = (col*(1-0.08*n) + col.roll(1, dims=2)*(0.08*n)).clamp(0,1)
#     else:
#         col = _parse_rgb(color, dev, dt)  # [3,1,1]

#     over = (col * paint_mask).clamp(0,1)                               # [3,H,W]
#     b = str(blend).lower()
#     if b=="add":
#         out = (base + over).clamp(0,1)
#     elif b=="screen":
#         out = 1 - (1-base)*(1-over); out = out.clamp(0,1)
#     elif b=="multiply":
#         out = base * (0.5 + over); out = out.clamp(0,1)
#     elif b=="alpha":
#         out = base*(1-paint_mask) + over
#     else:  # overlay
#         out = torch.where(base<0.5, 2*base*over, 1-2*(1-base)*(1-over)).clamp(0,1)
#     return out

# # ───────────────────────── 색연필 아웃라인(무지개 지원) ─────────────────────────
# @torch.no_grad()
# def draw_pencil_outline(
#     comp_rgb_3hw: torch.Tensor,
#     alpha_1hw: torch.Tensor,
#     *,
#     pencil: int = 1,
#     # 색/무지개
#     pencil_color: str = "255,128,0",
#     pencil_color2: Optional[str] = None,
#     pencil_multicolor: int = 0,
#     pencil_palette: str = "rainbow",   # ← 기본을 무지개로
#     pencil_bands: int = 7,
#     pencil_repeat: int = 1,
#     pencil_axis: str = "x",
#     # 형태/강도
#     pencil_width: int = 3,
#     pencil_soft: int = 1,
#     pencil_gain: float = 1.0,
#     pencil_opacity: float = 0.85,
#     pencil_thresh: float = 0.08,
#     pencil_rough: float = 0.7,
#     pencil_jitter: float = 0.8,
#     # 적용 영역
#     pencil_on_bg: int = 1,
#     pencil_on_fg: int = 1,
#     # 블렌드
#     pencil_blend: str = "alpha",
# ) -> torch.Tensor:
#     if not pencil: return comp_rgb_3hw
#     img = comp_rgb_3hw.clamp(0,1)
#     C,H,W = img.shape; dev, dt = img.device, img.dtype
#     kx, ky = _ensure_kernels(dev, dt)

#     # 다운스케일
#     ds = 2 if (H>=1080 or W>=1920) else 1
#     Hs, Ws = max(8,H//ds), max(8,W//ds)
#     a_full = alpha_1hw.clamp(0,1)
#     a = F.interpolate(a_full[None], size=(Hs,Ws), mode="bilinear", align_corners=False).squeeze(0)

#     ctx = torch.amp.autocast("cuda", dtype=torch.float16) if (torch.cuda.is_available() and img.is_cuda) else nullcontext()
#     with ctx:
#         gx = F.conv2d(a[None], kx, padding=1); gy = F.conv2d(a[None], ky, padding=1)
#         g  = torch.sqrt(gx*gx + gy*gy).squeeze(0)
#         g  = (g/g.max().clamp_min(1e-6))*float(pencil_gain)
#         hard = (g >= float(pencil_thresh)).float()
#         soft = ((g - float(pencil_thresh))/(1.0-float(pencil_thresh))).clamp(0,1)
#         edge = (0.5*hard + 0.5*soft)
#         if pencil_width>1:
#             k = max(1, pencil_width//ds)
#             edge = F.max_pool2d(edge, 2*k+1, 1, k)
#         if pencil_soft>0:
#             edge = F.avg_pool2d(edge, 3, 1, 1)

#         noise = F.avg_pool2d(torch.rand((1,1,Hs,Ws), device=dev, dtype=dt), 3, 1, 1)
#         jitter = 1.0 + (noise*2-1)*0.6*(0.15+0.85*float(pencil_jitter))
#         rough  = (noise*0.8+0.2)*(0.25+0.75*float(pencil_rough))
#         stroke = (edge*jitter).clamp(0,1)
#         stroke = (stroke*rough).clamp(0,1)

#     stroke_hi = _upsample_2d_1ch(stroke.squeeze(0), (H,W))
#     fg_edge   = _upsample_2d_1ch(g.squeeze(0), (H,W)).clamp(0,1)

#     bg_mask = (1.0 - a_full) if int(pencil_on_bg) else torch.zeros_like(a_full)
#     fg_mask = (fg_edge * 0.6) if int(pencil_on_fg) else torch.zeros_like(a_full)
#     region  = (bg_mask + fg_mask).clamp(0,1)
#     stroke_hi = (stroke_hi*region).clamp(0,1)  # [1,H,W]
    

#     # 색상(무지개/단색)
#     if str(pencil_palette).lower()=="rainbow":
#         col = _rainbow_field(H,W,dev,dt,bands=int(pencil_bands),
#                              repeat=int(pencil_repeat), axis=str(pencil_axis))
#         n = _rand_smooth_noise(H,W,dev,dt,k=7)
#         col = (col*(1-0.06*n) + col.roll(1,2)*(0.06*n)).clamp(0,1)
#     else:
#         colA = _parse_rgb(pencil_color, dev, dt)
#         if int(pencil_multicolor) and (pencil_color2 not in (None,"","None")):
#             colB = _parse_rgb(str(pencil_color2), dev, dt)
#             mix  = _upsample_2d_1ch(noise.squeeze(0), (H,W))
#             col  = colA*(1-mix) + colB*mix
#         else:
#             col  = colA  # [3,1,1]

#     sA   = (stroke_hi*float(pencil_opacity)).clamp(0,1)
#     paint= (col*sA).clamp(0,1)

#     mode = (pencil_blend or "alpha").lower()
#     if mode=="add":
#         out = (img + paint).clamp(0,1)
#     elif mode=="screen":
#         out = 1-(1-img)*(1-paint); out = out.clamp(0,1)
#     elif mode=="overlay":
#         out = torch.where(img<0.5, 2*img*paint, 1-2*(1-img)*(1-paint)).clamp(0,1)
#     else:
#         out = img*(1-sA) + paint
#     return out.clamp(0,1)

# # ───────────────────────── 편의: 둘 다 한번에 ─────────────────────────
# def draw_pencil_and_crayon(
#     comp_rgb_3hw: torch.Tensor, alpha_1hw: torch.Tensor, *,
#     # Crayon
#     crayon_enabled=True, crayon_width=14, crayon_jitter=6.0,
#     crayon_density=0.75, crayon_stamp=8, crayon_hardness=0.6, crayon_grain=0.6,
#     crayon_on_bg=True, crayon_blend="overlay",
#     crayon_color="255,120,255",
#     crayon_palette="rainbow", crayon_bands=7, crayon_repeat=1, crayon_axis="x",
#     # Pencil
#     pencil_enabled=True, **pencil_kwargs
# ) -> torch.Tensor:
#     out = comp_rgb_3hw
#     if crayon_enabled:
#         out = draw_crayon_fill(
#             out, alpha_1hw,
#             enable=True, width=int(crayon_width), jitter=float(crayon_jitter),
#             density=float(crayon_density), stamp=int(crayon_stamp),
#             hardness=float(crayon_hardness), grain=float(crayon_grain),
#             on_bg=bool(crayon_on_bg), blend=str(crayon_blend),
#             color=str(crayon_color), palette=str(crayon_palette),
#             bands=int(crayon_bands), repeat=int(crayon_repeat), axis=str(crayon_axis)
#         )
#     if pencil_enabled:
#         out = draw_pencil_outline(out, alpha_1hw, pencil=1, **pencil_kwargs)
#     return out
