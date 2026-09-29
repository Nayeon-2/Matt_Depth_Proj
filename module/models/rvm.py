# models/rvm.py
from __future__ import annotations
import os, sys, types, torch
from typing import List, Tuple

# --- torchvision 구버전 호환(shim) ---
from torch.hub import load_state_dict_from_url as _load_state_dict_from_url
_tv_utils = types.ModuleType("torchvision.models.utils")
_tv_utils.load_state_dict_from_url = _load_state_dict_from_url
sys.modules["torchvision.models.utils"] = _tv_utils

# 경로 설정: RobustVideoMatting 루트만 추가 (model 하위 X)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RVM_SRC_DIR = os.path.join(ROOT, "RobustVideoMatting")
if not os.path.isdir(RVM_SRC_DIR):
    raise RuntimeError(f"[ERR] RVM 소스 폴더 없음: {RVM_SRC_DIR}")
if RVM_SRC_DIR not in sys.path:
    sys.path.insert(0, RVM_SRC_DIR)

# 패키지 임포트 (상대임포트가 동작하도록 부모 패키지로 로드)
from RobustVideoMatting.model import MattingNetwork  # type: ignore

def _strip_module_prefix(sd: dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def compute_ds_ratio(h: int, w: int, target_long: int) -> float:
    return float(min(target_long / max(h, w), 1.0))

class RVM:
    """
    Robust Video Matting 추론 래퍼
    - rec(c1..c4) 상태 내부 보존
    - 입력: RGB 0..1, BCHW(1,3,H,W)
    - infer() -> (fgr_bchw, pha_bchw)
    """
    def __init__(
        self,
        weights_path: str,
        backbone: str = "mobilenetv3",
        device: str = "cuda",
        fp16: bool = True,
        use_channels_last: bool = True,
    ):
        self.device = device
        self.fp16 = bool(fp16)
        self.use_channels_last = bool(use_channels_last)
        self.net = self._load_rvm(weights_path, backbone, device)
        self.rec: List[torch.Tensor | None] = [None, None, None, None]

    def _load_rvm(self, weights_path: str, backbone: str, device: str):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"RVM 가중치 파일 없음: {weights_path}")

        # 1) TorchScript 우선
        try:
            m = torch.jit.load(weights_path, map_location=device).to(device).eval()
            if self.fp16 and device.startswith("cuda"):
                m = m.half()
            if self.use_channels_last:
                try: m = m.to(memory_format=torch.channels_last)
                except Exception: pass
            return m
        except Exception:
            pass

        # 2) state_dict 로드
        try:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = _strip_module_prefix(sd)

        net = MattingNetwork(backbone).to(device)
        net.load_state_dict(sd, strict=False)
        net.eval()
        if self.fp16 and device.startswith("cuda"):
            net = net.half()
        if self.use_channels_last:
            try: net = net.to(memory_format=torch.channels_last)
            except Exception: pass
        return net

    @torch.inference_mode()
    def infer(self, rgb01_bchw: torch.Tensor, target_long: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb01_bchw.ndim != 4 or rgb01_bchw.shape[0] != 1 or rgb01_bchw.shape[1] != 3:
            raise ValueError("입력은 1x3xHxW 이어야 합니다.")
        H, W = rgb01_bchw.shape[2], rgb01_bchw.shape[3]
        ds = compute_ds_ratio(H, W, int(target_long))

        if self.fp16 and self.device.startswith("cuda"):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                fgr, pha, *self.rec = self.net(rgb01_bchw, *self.rec, downsample_ratio=ds)
        else:
            fgr, pha, *self.rec = self.net(rgb01_bchw, *self.rec, downsample_ratio=ds)
        return fgr, pha

    def reset(self):
        self.rec = [None, None, None, None]

# 예전 스크립트 호환용
def load_rvm(weights_path: str, backbone: str, device: str = "cuda", fp16: bool = True):
    r = RVM(weights_path=weights_path, backbone=backbone, device=device, fp16=fp16)
    return r.net
