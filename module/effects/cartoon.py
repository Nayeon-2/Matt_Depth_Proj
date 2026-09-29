# effects/cartoon.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]

@dataclass
class CartoonConfig:
    enabled: bool = False
    mode: str = "all"           # "all" | "alpha" | "bg"
    alpha_thr: int = 127
    scale: float = 0.5
    levels: int = 8
    smooth: str = "bilateral"   # "bilateral" | "epf" | "gaussian"
    smooth_iter: int = 1
    sigma: float = 1.2
    bilateral_d: int = 7
    bilateral_sc: float = 75.0
    bilateral_ss: float = 75.0
    edge: str = "adaptive"      # "adaptive" | "canny" (canny는 sobel 유사 대체)
    edge_ks: int = 7
    edge_th1: int = 80
    edge_th2: int = 150
    edge_thicken: int = 1
    edge_blend: float = 1.0
    keep_brightness: float = 0.0

# ---------- 유틸: 타입 변환 ----------
def _to_torch_chw_u01(img: ArrayLike, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, bool]:
    """
    입력: numpy [H,W,3](u8/float) 또는 torch [3,H,W]/[H,W,3]
    출력: (torch [3,H,W] float[0..1], was_numpy)
    """
    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        elif not (t.ndim == 3 and t.shape[0] == 3):
            raise ValueError("expected CHW or HWC 3ch")
        t = t.to(device=(device or t.device), dtype=dtype)
        if t.max().item() > 1.5:
            t = t / 255.0
        return t, False
    # numpy
    if not (img.ndim == 3 and img.shape[-1] == 3):
        raise ValueError("expected numpy HWC 3ch")
    t = torch.from_numpy(img)
    if t.dtype == torch.uint8:
        t = t.to(torch.float32) / 255.0
    else:
        t = t.to(torch.float32)
        if t.max().item() > 1.5:
            t = t / 255.0
    if device is not None:
        t = t.to(device)
    return t.permute(2, 0, 1), True

def _from_torch_like(t: torch.Tensor, to_numpy: bool, like: ArrayLike) -> ArrayLike:
    t = t.clamp(0, 1)
    if to_numpy:
        return (t.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    if isinstance(like, torch.Tensor):
        return t.to(dtype=like.dtype, device=like.device)
    return t

# ---------- 블러 커널 ----------
def _gaussian_kernel1d(ks: int, sigma: float, device, dtype):
    if ks % 2 == 0: ks += 1
    ar = torch.arange(ks, device=device, dtype=dtype) - ks // 2
    w = torch.exp(-(ar ** 2) / (2 * sigma * sigma))
    return (w / w.sum()).to(dtype=dtype)

def _gaussian_blur_3hw(x3: torch.Tensor, ks: int, sigma: float) -> torch.Tensor:
    c, h, w = x3.shape
    k = _gaussian_kernel1d(ks, sigma, x3.device, x3.dtype)
    kx = k.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
    y = F.conv2d(x3.unsqueeze(0), kx, padding=(0, ks//2), groups=c)
    y = F.conv2d(y, ky, padding=(ks//2, 0), groups=c).squeeze(0)
    return y

# ---------- 박스필터(EPF/Guided Filter용) ----------
def _box_blur_1hw(x1: torch.Tensor, r: int) -> torch.Tensor:
    # x1: [1,H,W]
    ks = r * 2 + 1
    k = torch.ones((1, 1, ks, ks), device=x1.device, dtype=x1.dtype) / (ks * ks)
    return F.conv2d(x1.unsqueeze(0), k, padding=r).squeeze(0)

def _guided_filter_color(I3: torch.Tensor, p3: torch.Tensor, r: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """
    I3: guidance [3,H,W] (0..1), p3: input [3,H,W]
    간단/가벼운 색상 guided filter (edge-preserving). 모든 연산 GPU.
    """
    C, H, W = I3.shape
    I = I3
    p = p3
    ones = torch.ones((1, H, W), device=I.device, dtype=I.dtype)

    mean_I_r = _box_blur_1hw(I[0:1], r)
    mean_I_g = _box_blur_1hw(I[1:2], r)
    mean_I_b = _box_blur_1hw(I[2:3], r)

    mean_p_r = _box_blur_1hw(p[0:1], r)
    mean_p_g = _box_blur_1hw(p[1:2], r)
    mean_p_b = _box_blur_1hw(p[2:3], r)

    # covariance of I
    var_I_rr = _box_blur_1hw(I[0:1]*I[0:1], r) - mean_I_r*mean_I_r + eps
    var_I_rg = _box_blur_1hw(I[0:1]*I[1:1+1], r) - mean_I_r*mean_I_g
    var_I_rb = _box_blur_1hw(I[0:1]*I[2:2+1], r) - mean_I_r*mean_I_b
    var_I_gg = _box_blur_1hw(I[1:2]*I[1:2], r) - mean_I_g*mean_I_g + eps
    var_I_gb = _box_blur_1hw(I[1:2]*I[2:3], r) - mean_I_g*mean_I_b
    var_I_bb = _box_blur_1hw(I[2:3]*I[2:3], r) - mean_I_b*mean_I_b + eps

    cov_Ip_r = _box_blur_1hw(I[0:1]*p[0:1], r) - mean_I_r*mean_p_r
    cov_Ip_g = _box_blur_1hw(I[1:2]*p[1:2], r) - mean_I_g*mean_p_g
    cov_Ip_b = _box_blur_1hw(I[2:3]*p[2:3], r) - mean_I_b*mean_p_b

    # solve 3x3 per-pixel
    # build matrices
    a_r = (cov_Ip_r * var_I_gg * var_I_bb - cov_Ip_r * var_I_gb * var_I_gb
           - cov_Ip_g * var_I_rg * var_I_bb + cov_Ip_g * var_I_rb * var_I_gb
           + cov_Ip_b * var_I_rg * var_I_gb - cov_Ip_b * var_I_rb * var_I_gg)
    a_g = (- cov_Ip_r * var_I_rg * var_I_bb + cov_Ip_r * var_I_rb * var_I_gb
           + cov_Ip_g * var_I_rr * var_I_bb - cov_Ip_g * var_I_rb * var_I_rb
           - cov_Ip_b * var_I_rr * var_I_gb + cov_Ip_b * var_I_rb * var_I_rg)
    a_b = (cov_Ip_r * var_I_rg * var_I_gb - cov_Ip_r * var_I_rb * var_I_gg
           - cov_Ip_g * var_I_rr * var_I_gb + cov_Ip_g * var_I_rb * var_I_rg
           + cov_Ip_b * var_I_rr * var_I_gg - cov_Ip_b * var_I_rg * var_I_rg)
    det = (var_I_rr * var_I_gg * var_I_bb
           + 2*var_I_rg * var_I_rb * var_I_gb
           - var_I_rr * var_I_gb * var_I_gb
           - var_I_gg * var_I_rb * var_I_rb
           - var_I_bb * var_I_rg * var_I_rg)
    det = torch.clamp(det, min=1e-6)
    a_r = a_r / det
    a_g = a_g / det
    a_b = a_b / det

    b = (mean_p_r - a_r*mean_I_r) + (mean_p_g - a_g*mean_I_g) + (mean_p_b - a_b*mean_I_b)
    mean_a_r = _box_blur_1hw(a_r, r)
    mean_a_g = _box_blur_1hw(a_g, r)
    mean_a_b = _box_blur_1hw(a_b, r)
    mean_b   = _box_blur_1hw(b,   r)

    q = mean_a_r*I[0:1] + mean_a_g*I[1:2] + mean_a_b*I[2:3] + mean_b
    return torch.cat([q, q, q], dim=0).clamp(0, 1)  # 채널별 동일 적용

# ---------- 엣지 ----------
def _rgb_to_gray_1hw(x3: torch.Tensor) -> torch.Tensor:
    # x3: [3,H,W] float
    r, g, b = x3[0], x3[1], x3[2]
    return (0.299*r + 0.587*g + 0.114*b).unsqueeze(0)

def _edge_mask_torch(x3_small: torch.Tensor, edge_ks: int, th1: float, th2: float, mode: str = "adaptive") -> torch.Tensor:
    """
    반환: [1,H,W] float 0..1 (edges=1)
    """
    g1 = _rgb_to_gray_1hw(x3_small)
    # 부드럽게
    ks = max(3, (edge_ks|1))
    g1s = _gaussian_blur_3hw(g1, ks=ks, sigma=max(1.0, ks/6.0))

    # Sobel
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x3_small.device, dtype=x3_small.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x3_small.device, dtype=x3_small.dtype).view(1,1,3,3)
    gx = F.conv2d(g1s.unsqueeze(0), kx, padding=1).squeeze(0)
    gy = F.conv2d(g1s.unsqueeze(0), ky, padding=1).squeeze(0)
    mag = torch.sqrt(gx*gx + gy*gy)  # [1,H,W]

    if mode == "adaptive":
        # 지역 평균 기반(적응형) 임계
        r = max(1, edge_ks//2)
        mean_mag = _box_blur_1hw(mag, r)
        edges = (mag > mean_mag).to(x3_small.dtype)
    else:
        # 간단한 이중 임계 흉내 (th2만 사용)
        edges = (mag > (th2/255.0)).to(x3_small.dtype)

    # 두껍게
    if (edge_ks or 0) > 0:
        it = max(0, int(round(edge_ks/7)))  # config의 edge_thicken과 유사 스케일
        if it > 0:
            k = torch.ones((1,1,3,3), device=x3_small.device, dtype=x3_small.dtype)
            for _ in range(it):
                edges = torch.clamp(F.conv2d(edges.unsqueeze(0), k, padding=1).squeeze(0), 0, 1)
                edges = (edges > 0).to(x3_small.dtype)
    return edges.clamp(0,1)

# ---------- 본체 ----------
class CartoonEffect:
    def __init__(self, cfg: CartoonConfig):
        assert cfg.mode in ("all","alpha","bg")
        self.cfg = cfg

    def _down(self, x3: torch.Tensor) -> Tuple[torch.Tensor, float]:
        s = float(max(0.3, min(1.0, self.cfg.scale)))
        if s >= 0.999: return x3, 1.0
        _, H, W = x3.shape
        h2 = max(1, int(H * s))
        w2 = max(1, int(W * s))
        y = F.interpolate(x3.unsqueeze(0), size=(h2, w2), mode="bilinear", align_corners=False).squeeze(0)
        return y, s

    def _up(self, x3_small: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return F.interpolate(x3_small.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

    def _smooth_torch(self, x3: torch.Tensor) -> torch.Tensor:
        # gaussian: 완전 GPU
        # bilateral/epf: 가벼운 guided filter로 대체(EPF 유사), 반복 적용 가능
        if self.cfg.smooth == "gaussian":
            ks = max(3, int(round(2*self.cfg.sigma))*2+1)
            y = x3
            for _ in range(max(1, self.cfg.smooth_iter)):
                y = _gaussian_blur_3hw(y, ks=ks, sigma=max(0.6, self.cfg.sigma))
            return y.clamp(0,1)
        else:
            # guided filter 반복: guidance=원본(gray) / 입력=x3
            y = x3
            for _ in range(max(1, self.cfg.smooth_iter)):
                guide = x3
                y = _guided_filter_color(guide, y, r=max(2, int(self.cfg.sigma*4)), eps=1e-3)
            return y.clamp(0,1)

    def _quantize_torch(self, x3: torch.Tensor) -> torch.Tensor:
        # 색 단계 줄이기 (0..1 범위에서 근사)
        L = max(2, int(self.cfg.levels))
        div = 1.0 / L
        return torch.floor(x3 / div) * div

    def _apply_cartoon_torch(self, bgr3: torch.Tensor) -> torch.Tensor:
        # bgr3: [3,H,W] float 0..1
        small, _ = self._down(bgr3)
        sm = self._smooth_torch(small)
        qt = self._quantize_torch(sm)

        edges = _edge_mask_torch(small, edge_ks=int(self.cfg.edge_ks),
                                 th1=float(self.cfg.edge_th1),
                                 th2=float(self.cfg.edge_th2),
                                 mode=("adaptive" if self.cfg.edge != "canny" else "canny"))
        # edge blend: 윤곽선(=1)을 어둡게
        if self.cfg.edge_blend > 0:
            a = float(max(0.0, min(1.0, self.cfg.edge_blend)))
            # edges: 1이면 더 어둡게. 3ch로 확장
            e3 = edges.repeat(3,1,1)
            qt = torch.clamp(qt - e3 * a * 0.5, 0, 1)

        out = self._up(qt, bgr3.shape[1], bgr3.shape[2])

        if self.cfg.keep_brightness > 1e-6:
            # HSV에서 V만 원본과 섞기 (간단 버전: RGB에서 밝기 근사)
            k = float(max(0.0, min(1.0, self.cfg.keep_brightness)))
            # 밝기 근사: max(R,G,B)
            v_src = torch.max(bgr3, dim=0, keepdim=True).values
            v_out = torch.max(out,  dim=0, keepdim=True).values
            scale = torch.clamp((v_src * k + v_out * (1-k)) / (v_out + 1e-6), 0.0, 4.0)
            out = torch.clamp(out * scale, 0, 1)
        return out

    def apply(self, frame_bgr: ArrayLike, alpha_u8: Optional[ArrayLike] = None) -> ArrayLike:
        """
        frame_bgr:
          - numpy [H,W,3] u8/float  → GPU로 올려 처리 후 numpy로 반환
          - torch [3,H,W] float(0..1) → 같은 device/dtype로 반환
        alpha_u8: (선택) [H,W] u8 또는 torch [H,W]/[1,H,W] float(0..1)
        """
        if not self.cfg.enabled:
            return frame_bgr

        dev = None
        if isinstance(frame_bgr, torch.Tensor):
            dev = frame_bgr.device
        x3, was_numpy = _to_torch_chw_u01(frame_bgr, device=dev, dtype=torch.float32)

        # 전체 처리 or 마스크 처리
        if self.cfg.mode == "all" or alpha_u8 is None:
            y3 = self._apply_cartoon_torch(x3)
        else:
            # 마스크 생성 (torch 1xHxW, {0,1})
            if isinstance(alpha_u8, torch.Tensor):
                a = alpha_u8
                if a.ndim == 2:
                    a = a.unsqueeze(0)
                thr = (float(self.cfg.alpha_thr) / 255.0) if a.dtype.is_floating_point else self.cfg.alpha_thr
                m = (a > thr).to(torch.float32)
                m = m.to(device=x3.device)
            else:
                a = np.asarray(alpha_u8)
                if a.ndim != 2:
                    raise ValueError("alpha_u8 must be [H,W]")
                if a.dtype != np.uint8:
                    thr_f = float(self.cfg.alpha_thr) / 255.0
                    a = (a > thr_f).astype(np.uint8) * 255
                m = torch.from_numpy((a > self.cfg.alpha_thr).astype(np.uint8)).to(torch.float32).unsqueeze(0).to(x3.device)

            if self.cfg.mode == "bg":
                m = 1.0 - m

            styled = self._apply_cartoon_torch(x3)
            y3 = styled * m + x3 * (1.0 - m)

        return _from_torch_like(y3, was_numpy, frame_bgr)
