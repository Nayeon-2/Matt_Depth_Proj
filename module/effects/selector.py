# alphaops/selector.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2

__all__ = ["select_component"]

@torch.inference_mode()
def select_component(alpha_2d: torch.Tensor,
                     depth_np: Optional[np.ndarray],
                     mode: str,
                     center_xy: Tuple[int, int]) -> torch.Tensor:
    """
    알파 마스크에서 사람/객체 컴포넌트 선택.
    - alpha_2d: [H,W] torch.float (0..1)
    - depth_np: [H,W] float 또는 None
    - mode: "all" | "largest" | "center" | "nearest"
    - center_xy: (cx,cy)
    반환: keep_mask (alpha_2d와 같은 dtype/device, 0/1)
    """
    if mode == "all":
        return torch.ones_like(alpha_2d)

    H, W = alpha_2d.shape
    short = 480
    scale = min(1.0, short / max(H, W))
    if scale < 1.0:
        a_small = F.interpolate(alpha_2d.unsqueeze(0).unsqueeze(0),
                                scale_factor=scale, mode="bilinear",
                                align_corners=False).squeeze().cpu().numpy()
        d_small = None if depth_np is None else cv2.resize(depth_np, (a_small.shape[1], a_small.shape[0]),
                                                           interpolation=cv2.INTER_AREA)
    else:
        a_small = alpha_2d.detach().cpu().numpy()
        d_small = depth_np

    m = (a_small > 0.4).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return torch.zeros_like(alpha_2d)

    comps = []
    cx, cy = int(center_xy[0] * scale), int(center_xy[1] * scale)
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        if xs.size < 200:
            continue
        area = xs.size
        cxs, cys = xs.mean(), ys.mean()
        center_dist = ((cxs - cx) ** 2 + (cys - cy) ** 2) ** 0.5
        median_depth = -1.0
        if d_small is not None:
            median_depth = float(np.median(d_small[ys, xs]))
        comps.append(dict(lab=lab, area=area, center_dist=center_dist, median_depth=median_depth))

    if not comps:
        return torch.zeros_like(alpha_2d)

    if mode == "largest":
        sel = max(comps, key=lambda c: c["area"])
    elif mode == "center":
        sel = min(comps, key=lambda c: c["center_dist"])
    elif mode == "nearest" and d_small is not None:
        sel = max(comps, key=lambda c: c["median_depth"])
    else:
        sel = max(comps, key=lambda c: c["area"])

    keep_small = (labels == sel["lab"]).astype(np.uint8) * 255
    if scale < 1.0:
        keep = cv2.resize(keep_small, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        keep = keep_small

    keep_t = torch.from_numpy((keep > 127).astype(np.float32)).to(device=alpha_2d.device)
    return keep_t
