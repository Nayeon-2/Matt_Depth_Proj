# alphaops/alpha.py
from __future__ import annotations
import time
import torch
import torch.nn.functional as F

__all__ = [
    "gaussian_kernel_1d",
    "blur_alpha_gpu",
    "unsharp_mask_alpha",
    "dilate_gpu",
    "erode_gpu",
    "close_gpu",
    "ema_smooth_alpha",
]

def gaussian_kernel_1d(sigma: float, dtype, device):
    """가우시안 1D 커널 (합=1)"""
    if sigma <= 0:
        sigma = 1.0
    radius = max(1, int(3.0 * float(sigma)))
    k = 2 * radius + 1
    x = torch.arange(k, device=device, dtype=torch.float32) - radius
    g = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    g = g / g.sum()
    return g.to(dtype=dtype)

@torch.inference_mode()
def blur_alpha_gpu(alpha_1chw: torch.Tensor, sigma: float):
    """채널 1짜리 알파(1x1xH xW)에 가우시안 블러(분리합성)."""
    if sigma <= 0:
        return alpha_1chw
    dtype = alpha_1chw.dtype
    device = alpha_1chw.device
    g1d = gaussian_kernel_1d(sigma, dtype=dtype, device=device)
    k = g1d.numel()
    pad = k // 2
    kx = g1d.view(1, 1, 1, k)
    ky = g1d.view(1, 1, k, 1)
    x = F.pad(alpha_1chw, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, ky)
    return x

@torch.inference_mode()
def unsharp_mask_alpha(alpha_1chw: torch.Tensor, amount: float = 0.3, radius: float = 1.0):
    """언샤프 마스크로 알파 선명도 보정."""
    if amount <= 0 or radius <= 0:
        return alpha_1chw
    blur = blur_alpha_gpu(alpha_1chw, radius)
    sharp = (alpha_1chw + amount * (alpha_1chw - blur)).clamp(0, 1)
    return sharp

def _odd(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)

@torch.inference_mode()
def dilate_gpu(a_1chw: torch.Tensor, k: int):
    """팽창 (MaxPool)"""
    k = _odd(k)
    if k <= 1:
        return a_1chw
    pad = k // 2
    return F.max_pool2d(a_1chw, kernel_size=k, stride=1, padding=pad)

@torch.inference_mode()
def erode_gpu(a_1chw: torch.Tensor, k: int):
    """침식 (MinPool=1-MaxPool(1-x))"""
    k = _odd(k)
    if k <= 1:
        return a_1chw
    pad = k // 2
    return 1.0 - F.max_pool2d(1.0 - a_1chw, kernel_size=k, stride=1, padding=pad)

@torch.inference_mode()
def close_gpu(a_1chw: torch.Tensor, k: int):
    """클로징 (팽창 후 침식)"""
    k = _odd(k)
    if k <= 1:
        return a_1chw
    return erode_gpu(dilate_gpu(a_1chw, k), k)

@torch.inference_mode()
def ema_smooth_alpha(alpha: torch.Tensor,
                     alpha_prev: torch.Tensor | None,
                     base_ema: float = 0.8,
                     adaptive: bool = True,
                     loop_start_ts: float | None = None,
                     tau: float = 0.060,
                     motion_thresh: float = 0.06,
                     fast_ema: float = 0.55):
    """
    알파 EMA(시간 감쇠 + 적응형 가중)
    - alpha: 현재 알파(1x1xH xW)
    - alpha_prev: 이전 EMA 결과(같은 shape) 또는 None
    반환: (smoothed_alpha, new_alpha_prev, motion_mag)
    """
    if alpha_prev is None:
        return alpha.detach(), alpha.detach(), None

    motion_mag = (alpha - alpha_prev).abs().mean().item()
    use_ema = base_ema if not (motion_mag is not None and motion_mag > motion_thresh) else fast_ema

    if loop_start_ts is None:
        dt_decay = 0.0
    else:
        dt_decay = max(1e-3, time.time() - loop_start_ts)

    time_decay = max(0.0, min(0.95, use_ema ** (dt_decay / tau)))

    if adaptive:
        diff = (alpha - alpha_prev).abs().clamp(0, 1)
        ema_scale = torch.clamp(1.0 - 3.0 * diff, 0.2, 1.0)
        new_prev = (ema_scale * (time_decay * alpha_prev) + (1.0 - ema_scale) * alpha).detach()
    else:
        new_prev = (time_decay * alpha_prev + (1.0 - time_decay) * alpha).detach()

    return new_prev, new_prev, motion_mag
