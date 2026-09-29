# module/engine/gpu_pack.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def rgb01_to_uyvy422_packed_cuda(rgb01_3hw: torch.Tensor, standard: str = "bt709") -> torch.Tensor:
    """
    rgb01_3hw: [3,H,W] float32/float16 on CUDA, range [0,1]
    Returns: packed UYVY bytes as torch.uint8 tensor of shape [H, W*2] on CUDA
             (each 2-pixel group → 4 bytes: U, Y0, V, Y1)
    Notes:
      - 4:2:2 수평 서브샘플링 (짝수-홀수 픽셀 쌍)
      - 색공간: 기본 BT.709 (HD). 필요시 'bt601' 선택 가능
      - 리미티드 레인지 (Y:16..235, Cb/Cr:16..240)로 패킹
    """
    assert rgb01_3hw.is_cuda, "rgb01_to_uyvy422_packed_cuda: input must be on CUDA"
    C, H, W = rgb01_3hw.shape
    if C != 3:
        raise ValueError("rgb01_to_uyvy422_packed_cuda expects [3,H,W]")

    # 보정: W가 홀수면 마지막 1열 drop (UYVY는 2픽셀 단위)
    if (W & 1) == 1:
        rgb01_3hw = rgb01_3hw[:, :, :W-1]
        W = W - 1

    x = rgb01_3hw.float()
    r = x[0]; g = x[1]; b = x[2]

    # BT.709 (HD) / BT.601 (SD) 계수
    if standard.lower() == "bt601":
        kr, kg, kb = 0.299, 0.587, 0.114
    else:
        kr, kg, kb = 0.2126, 0.7152, 0.0722

    # Full-range RGB[0..1] → Limited Y'CbCr
    # Y  = 16   + 219*(kr*R + kg*G + kb*B)
    # Cb = 128  + 224*( (B-Y')/(2*(1-kb)) )
    # Cr = 128  + 224*( (R-Y')/(2*(1-kr)) )
    Yf  = kr*r + kg*g + kb*b
    Y   = 16.0  + 219.0 * Yf
    Cb  = 128.0 + 224.0 * ((b - Yf) / (2.0*(1.0 - kb) + 1e-8))
    Cr  = 128.0 + 224.0 * ((r - Yf) / (2.0*(1.0 - kr) + 1e-8))

    # 수평 2픽셀당 Cb/Cr 평균 (4:2:2)
    # shape 유지: [H, W] → [H, W//2]
    Cb_pair = (Cb[:, 0::2] + Cb[:, 1::2]) * 0.5
    Cr_pair = (Cr[:, 0::2] + Cr[:, 1::2]) * 0.5

    # 클램프 & 양자화
    def q8(t): return t.clamp(0.0, 255.0).round().to(torch.uint8)
    Y0 = q8(Y[:, 0::2])
    Y1 = q8(Y[:, 1::2])
    U  = q8(Cb_pair)
    V  = q8(Cr_pair)

    # UYVY 패킹: 2픽셀 그룹당 [U, Y0, V, Y1]
    # 각 행에 대해 interleave
    # U/V: [H, W//2], Y0/Y1: [H, W//2] → cat along last dim
    # 최종: [H, W//2, 4] → reshape [H, W*2]
    row = torch.stack([U, Y0, V, Y1], dim=2)           # [H, W//2, 4]
    packed = row.reshape(H, (W//2)*4)                  # [H, W*2]
    return packed
