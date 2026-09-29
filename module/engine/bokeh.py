# effects/bokeh.py
from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def apply_depth_bokeh(
    base_bg_3hw: torch.Tensor,   # 3xHxW (RGB)
    alpha_1hw: torch.Tensor,     # 1xHxW
    depth_np: "np.ndarray | None",
    *,
    invert_depth: bool = True,
    blur_weak: float = 1.2,
    blur_strong: float = 4.0
) -> torch.Tensor:
    """
    깊이/알파 기반 배경 블러 합성된 배경을 반환 (피사체는 제외)
    """
    if depth_np is None:
        return base_bg_3hw

    # depth 0..1 텐서
    d_np = depth_np if invert_depth else (1.0 - depth_np)
    depth_t = torch.as_tensor(d_np, device=base_bg_3hw.device, dtype=torch.float32).clamp(0,1)

    def _blur(x, s):
        if s <= 0: return x
        k = max(1, int(3*s))*2+1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)

    weak   = _blur(base_bg_3hw.unsqueeze(0), float(blur_weak)).squeeze(0)
    strong = _blur(base_bg_3hw.unsqueeze(0), float(blur_strong)).squeeze(0)

    w_bg   = (1.0 - alpha_1hw).squeeze(0)  # HxW
    w_far  = depth_t                       # HxW
    mix_w  = (w_bg * w_far).clamp(0,1).unsqueeze(0)  # 1xHxW

    return weak*(1.0 - mix_w) + strong*mix_w
