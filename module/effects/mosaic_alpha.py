# effects/mosaic_alpha.py
from __future__ import annotations
import math
from typing import Tuple, Optional, Literal, Union

import torch
import torch.nn.functional as F
import numpy as np

ArrayLike = Union[np.ndarray, torch.Tensor]
MosaicMode = Literal["off", "alpha", "bg", "all"]
MosaicType = Literal["pixelate", "blur"]


def _to_torch_chw(img: ArrayLike, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, bool]:
    """
    입력:
      - HxWx3 (numpy uint8/float) 또는 CHW (torch)
    출력:
      - img_3hw: torch.Tensor (C,H,W) float [0..1] 범위
      - was_numpy: True면 호출자가 numpy였음(나중에 원래 타입으로 반환하기 위함)
    """
    if isinstance(img, torch.Tensor):
        # CHW 또는 HWC를 모두 허용
        if img.ndim == 3 and img.shape[0] == 3:
            t = img
        elif img.ndim == 3 and img.shape[-1] == 3:
            t = img.permute(2, 0, 1)
        else:
            raise ValueError("Expected torch tensor with shape [3,H,W] or [H,W,3]")
        t = t.to(dtype=dtype, device=(device or img.device))
        # 값 범위가 0..255 가능 → 0..1로 정규화
        if t.dtype.is_floating_point and t.max().item() > 1.5:
            t = t / 255.0
        return t, False

    # numpy
    if not (img.ndim == 3 and img.shape[-1] == 3):
        raise ValueError("Expected numpy array with shape [H,W,3]")
    t = torch.from_numpy(img)
    if t.dtype == torch.uint8:
        t = t.to(torch.float32) / 255.0
    else:
        t = t.to(torch.float32)
        if t.max().item() > 1.5:
            t = t / 255.0
    if device is not None:
        t = t.to(device)
    t = t.permute(2, 0, 1)  # HWC -> CHW
    return t, True


def _from_torch_chw(t: torch.Tensor, to_numpy: bool, like: ArrayLike) -> ArrayLike:
    """
    torch CHW float[0..1] → 호출자 타입으로 반환
    - numpy로 들어왔으면 numpy HWC uint8로 돌려준다.
    - torch로 들어왔으면 CHW float[0..1] 유지(호출자 텐서와 같은 device/dtype로 반환)
    """
    t = t.clamp(0, 1)
    if to_numpy:
        out = (t.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
        return out
    # torch로 들어온 경우: 호출자 dtype/device 최대한 유지
    # (대개 float[0..1], device='cuda') 형태일 것
    if isinstance(like, torch.Tensor):
        out = t.to(dtype=like.dtype, device=like.device)
        return out
    return t


def _compute_kernel_from_block(block: int) -> int:
    # block≈18일 때 k≈(block/2)*2+1 권장
    k = max(3, (max(3, int(round(block / 2))) * 2 + 1))
    if k % 2 == 0: k += 1
    return k


def _pixelate_torch(img_3hw: torch.Tensor, block: int) -> torch.Tensor:
    """
    간단 픽셀화: down → up (nearest)
    img_3hw: [3,H,W] float[0..1]
    """
    _, H, W = img_3hw.shape
    h2 = max(1, H // block)
    w2 = max(1, W // block)
    x = F.interpolate(img_3hw.unsqueeze(0), size=(h2, w2), mode="nearest")
    x = F.interpolate(x, size=(H, W), mode="nearest").squeeze(0)
    return x.clamp(0, 1)


def _gaussian_blur_torch(img_3hw: torch.Tensor, sigma: Optional[float], block: int) -> torch.Tensor:
    """
    Separable Gaussian blur (X, Y)
    - sigma가 주어지면 ksize는 sigma로 자동결정(홀수 최소 3)
    - sigma가 없으면 block 기반으로 ksize 추정, sigma는 ksize에서 유도
    """
    _, H, W = img_3hw.shape
    dtype = img_3hw.dtype
    device = img_3hw.device

    if sigma is None or sigma <= 0:
        k = _compute_kernel_from_block(block)
        # k에 대응하는 sigma 근사(일반적 경험치)
        sigma = max(0.1, (k - 1) / 6.0)
    else:
        # sigma 기반으로 k 자동결정 (대략 3 sigma 범위 커버)
        k = int(max(3, int(math.ceil(sigma * 6)) | 1))  # 홀수 보장

    # 1D 가우시안 커널
    ar = torch.arange(k, device=device, dtype=dtype) - k // 2
    w = torch.exp(-(ar ** 2) / (2 * sigma * sigma))
    w = (w / w.sum()).to(dtype=dtype, device=device)

    kx = w.view(1, 1, 1, k).repeat(3, 1, 1, 1)  # [3,1,1,k]
    ky = w.view(1, 1, k, 1).repeat(3, 1, 1, 1)  # [3,1,k,1]

    x = F.conv2d(img_3hw.unsqueeze(0), kx, padding=(0, k // 2), groups=3)
    x = F.conv2d(x, ky, padding=(k // 2, 0), groups=3).squeeze(0)
    return x.clamp(0, 1)


def _apply_style_torch(img_3hw: torch.Tensor, style: MosaicType, block: int, sigma: Optional[float]) -> torch.Tensor:
    if style == "pixelate":
        return _pixelate_torch(img_3hw, block)
    elif style == "blur":
        return _gaussian_blur_torch(img_3hw, sigma, block)
    else:
        # 안전장치: 모르는 스타일이면 원본
        return img_3hw


def _alpha_to_mask_1hw(alpha_like: ArrayLike,
                       alpha_thr: int,
                       dilate_px: int,
                       device=None,
                       dtype=torch.float32) -> torch.Tensor:
    """
    alpha_like:
      - numpy: [H,W] u8 (0..255) 또는 [H,W] float
      - torch: [H,W] or [1,H,W] float/uint8

    반환:
      - mask_1hw: torch float [1,H,W], {0.0,1.0}
    """
    if isinstance(alpha_like, torch.Tensor):
        a = alpha_like
        if a.ndim == 2:
            a = a.unsqueeze(0)  # [1,H,W]
        elif a.ndim == 3 and a.shape[0] != 1:
            raise ValueError("alpha tensor must be [H,W] or [1,H,W]")
        a = a.to(device=(device or a.device))
        if a.dtype == torch.uint8:
            thr = torch.tensor(alpha_thr, device=a.device, dtype=torch.uint8)
            m = (a > thr).to(torch.float32)
        else:
            # a가 0..1 범위로 가정
            thr_f = float(alpha_thr) / 255.0
            m = (a > thr_f).to(torch.float32)
    else:
        a = np.asarray(alpha_like)
        if a.ndim != 2:
            raise ValueError("alpha_u8 numpy must be [H,W]")
        if a.dtype != np.uint8:
            # float로 가정 (0..1)
            thr_f = float(alpha_thr) / 255.0
            a = (a > thr_f).astype(np.uint8) * 255
        m = torch.from_numpy((a > alpha_thr).astype(np.uint8)).to(torch.float32)
        m = m.unsqueeze(0)
        if device is not None:
            m = m.to(device)

    # 팽창(dilate): binary mask의 max pooling과 동일
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        pad = dilate_px
        # stride=1, same padding
        m = F.max_pool2d(m.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)
        m = (m > 0.5).to(torch.float32)

    return m  # [1,H,W] float


class AlphaMosaicEffect:
    """
    모드:
      - 'alpha' : 전경(RVM 알파>thr)만 처리
      - 'bg'    : 배경(알파 역영역)만 처리
      - 'all'   : 프레임 전체 처리
      - 'off'   : 비활성
    스타일:
      - mosaic_type: 'pixelate' | 'blur'
      - block: 픽셀화 블록 크기(또는 블러 커널 추정의 기준)
      - sigma: 가우시안 블러 강도(>0이면 우선)
    * 입력으로 torch 텐서를 주면 GPU에서 처리. numpy를 주면 내부에서 torch로 변환 후 처리.
    """
    def __init__(
        self,
        mode: MosaicMode = "off",
        mosaic_type: MosaicType = "pixelate",
        block: int = 18,
        dilate_px: int = 2,
        alpha_thr: int = 127,
        sigma: Optional[float] = None,
    ):
        assert mode in ("off", "alpha", "bg", "all")
        assert mosaic_type in ("pixelate", "blur")
        self.mode = mode
        self.mosaic_type = mosaic_type
        self.block = int(block)
        self.dilate_px = int(dilate_px)
        self.alpha_thr = int(alpha_thr)
        self.sigma = None if sigma is None else float(sigma)

    # ---------------- public API ----------------

    def apply_on_fg(self, frame_fg: ArrayLike, alpha_u8: ArrayLike) -> ArrayLike:
        """
        전경 레이어만 모자이크/블러 적용 (mode == 'alpha'일 때만 의미 있음).
        frame_fg: [H,W,3] numpy 또는 [3,H,W] torch float[0..1]
        alpha_u8: [H,W] numpy uint8 (0..255) or torch [H,W]/[1,H,W]
        """
        if self.mode != "alpha":
            return frame_fg

        # 같은 device로 처리
        dev = (frame_fg.device if isinstance(frame_fg, torch.Tensor) else None)
        img, was_numpy = _to_torch_chw(frame_fg, device=dev, dtype=torch.float32)
        mask_1hw = _alpha_to_mask_1hw(alpha_u8, self.alpha_thr, self.dilate_px, device=img.device)

        stylized = _apply_style_torch(img, self.mosaic_type, self.block, self.sigma)
        out = stylized * mask_1hw + img * (1.0 - mask_1hw)
        return _from_torch_chw(out, was_numpy, frame_fg)

    def apply_on_out(self, frame_out: ArrayLike, alpha_u8: Optional[ArrayLike] = None) -> ArrayLike:
        """
        frame_out: [H,W,3] numpy 또는 [3,H,W] torch float[0..1]
        alpha_u8 : [H,W] numpy uint8 or torch [H,W]/[1,H,W] (mode=='all'이면 무시 가능)
        """
        if self.mode == "off":
            return frame_out

        dev = (frame_out.device if isinstance(frame_out, torch.Tensor) else None)
        img, was_numpy = _to_torch_chw(frame_out, device=dev, dtype=torch.float32)

        if self.mode == "all":
            out = _apply_style_torch(img, self.mosaic_type, self.block, self.sigma)
            return _from_torch_chw(out, was_numpy, frame_out)

        if alpha_u8 is None:
            # 마스크가 필요하지만 없으면 원본 반환
            return frame_out

        mask_1hw = _alpha_to_mask_1hw(alpha_u8, self.alpha_thr, self.dilate_px, device=img.device)
        if self.mode == "bg":
            mask_1hw = 1.0 - mask_1hw  # 배경만

        stylized = _apply_style_torch(img, self.mosaic_type, self.block, self.sigma)
        out = stylized * mask_1hw + img * (1.0 - mask_1hw)
        return _from_torch_chw(out, was_numpy, frame_out)
