#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rvm_spout_Depth_fin.py
Alpha + Depth Spout (GPU-first, UYVY→RGB CUDA, CUDA-GL PBO interop, OWL Spout)

Spout Streams
- RT_RVM_ALPHA  : RGBA8 (RGB=0, A=alpha)
- RT_DEPTH_RG16 : RGBA8 (R=depth_hi8, G=depth_lo8, B=frameID, A=255)

Live console commands:
  delay / delayms / ema / sharp / sigma / long / cap / stats / now / flush / panic
  edge N | feather X | contrast X | gamma X | clip L H
  d.en 0/1 | d.stride N | d.size N | d.ema X | d.minmax L H

Notes:
- CPU 최소화, GPU 경로 고정 (CUDA → PBO → OpenGL → Spout)
- Spout는 Off World Live Live-Streaming Tools(OWL) 기준
"""

import os, sys, time, subprocess, shlex, ctypes, numpy as np, threading, queue, collections, math
import importlib as _importlib
import importlib.util as _importlib_util
import inspect, re
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

# ---- Torch perf ----
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
except Exception:
    pass

# ---- OpenGL / Spout ----
from OpenGL import GL
import glfw
from SpoutGL import SpoutSender
from SpoutGL.enums import GL_RGBA

# ---- CUDA Runtime interop (PBO) ----
def _load_cudart():
    for n in ("cudart64_12.dll","cudart64_120.dll","cudart64_110.dll"):
        try:
            return ctypes.windll.LoadLibrary(n)
        except Exception:
            continue
    return None

_cudart = _load_cudart()
cudaSuccess = 0
cudaMemcpyDeviceToDevice = 2
cudaGraphicsRegisterFlagsWriteDiscard = 2
if _cudart:
    _cudart.cudaGraphicsGLRegisterBuffer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint]
    _cudart.cudaGraphicsGLRegisterBuffer.restype  = ctypes.c_int
    _cudart.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
    _cudart.cudaGraphicsUnregisterResource.restype  = ctypes.c_int
    _cudart.cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    _cudart.cudaGraphicsMapResources.restype  = ctypes.c_int
    _cudart.cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    _cudart.cudaGraphicsUnmapResources.restype  = ctypes.c_int
    _cudart.cudaGraphicsResourceGetMappedPointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p
    ]
    _cudart.cudaGraphicsResourceGetMappedPointer.restype = ctypes.c_int
    _cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _cudart.cudaMemcpy.restype  = ctypes.c_int

def _ck(code, msg=""):
    if code != cudaSuccess:
        raise RuntimeError(f"CUDA error({code}) {msg}")

class CudaGLPBO:
    def __init__(self, pbo, size_bytes):
        if not _cudart:
            raise RuntimeError("CUDA runtime DLL(cudart) not found. Put CUDA 12.x DLLs in PATH.")
        self.res = ctypes.c_void_p()
        _ck(_cudart.cudaGraphicsGLRegisterBuffer(ctypes.byref(self.res), ctypes.c_uint(pbo),
                                                 cudaGraphicsRegisterFlagsWriteDiscard))
        self.size = int(size_bytes)
    def upload_from_torch_u8(self, torch_u8_hwc):
        _ck(_cudart.cudaGraphicsMapResources(1, ctypes.byref(self.res), None))
        try:
            dev_ptr  = ctypes.c_void_p()
            dev_size = ctypes.c_size_t()
            _ck(_cudart.cudaGraphicsResourceGetMappedPointer(ctypes.byref(dev_ptr), ctypes.byref(dev_size), self.res))
            _ck(_cudart.cudaMemcpy(dev_ptr,
                                   ctypes.c_void_p(int(torch_u8_hwc.data_ptr())),
                                   ctypes.c_size_t(torch_u8_hwc.numel()),
                                   cudaMemcpyDeviceToDevice))
        finally:
            _ck(_cudart.cudaGraphicsUnmapResources(1, ctypes.byref(self.res), None))
    def release(self):
        try: _ck(_cudart.cudaGraphicsUnregisterResource(self.res))
        except Exception: pass
        self.res = ctypes.c_void_p()

# ---- GL helpers ----
def gl_init_hidden(w=32, h=32):
    if not glfw.init(): raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(w, h, "hidden", None, None)
    glfw.make_context_current(win)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    return win

def make_tex_rgba8(w, h):
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    return tex

def make_pbo(size):
    pbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, size, None, GL.GL_STREAM_DRAW)
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
    return pbo

# ---- Utils ----
def parse_fps(value):
    if value is None: return 0.0
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)

# ---- torchvision shim (RVM 구버전 호환) ----
import types as _types
try:
    import torchvision as _tv
    from torch.hub import load_state_dict_from_url as _lsu
    _tv_models = _importlib.import_module("torchvision.models")
    if "torchvision.models.utils" not in sys.modules:
        _utils_mod = _types.ModuleType("torchvision.models.utils")
        _utils_mod.load_state_dict_from_url = _lsu
        sys.modules["torchvision.models.utils"] = _utils_mod
        setattr(_tv_models, "utils", _utils_mod)
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "torchvision 미설치. CUDA 버전에 맞춰 설치:\n"
        "CUDA 12.1: pip install torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "CUDA 11.8: pip install torchvision --index-url https://download.pytorch.org/whl/cu118\n"
        "CPU-only:  pip install torchvision --index-url https://download.pytorch.org/whl/cpu"
    )

# ---- Robust Video Matting ----
RVM_DIR = os.path.join(os.path.dirname(__file__), "RobustVideoMatting")
sys.path.insert(0, RVM_DIR)
from model import MattingNetwork

# =========================================================
# DepthAnything V2: 정석 경로 우선 + 이름/시グ니처 필터 스캐너
# =========================================================
_NAME_ALLOW_RE = re.compile(r"(depth.*anything|anything.*depth|depthanything|^dav?2$|^da[v_]?2$)", re.I)
_BAD_NAME_RE   = re.compile(r"(attention|mlp|block|layer|conv|norm|act|residual|head|neck|backbone)$", re.I)

def _add_paths(root_dir: Optional[str]):
    if not root_dir: return
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    src_dir = os.path.join(root_dir, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def _safe_import(modname: str):
    try:
        return _importlib.import_module(modname)
    except Exception:
        return None

def _scan_modules_by_path(root: str):
    if not root or not os.path.isdir(root):
        return
    for base in (root, os.path.join(root, "src")):
        if not os.path.isdir(base): continue
        for r, _, files in os.walk(base):
            rp = r.replace("\\","/").lower()
            if "depth_anything" not in rp and "depth-anything" not in rp:
                continue
            for fn in files:
                if not fn.endswith(".py"): continue
                path = os.path.join(r, fn)
                try:
                    mname = f"_dav2_{abs(hash(path))}"
                    spec = _importlib_util.spec_from_file_location(mname, path)
                    if spec and spec.loader:
                        m = _importlib_util.module_from_spec(spec)
                        spec.loader.exec_module(m)  # type: ignore
                        yield m
                except Exception:
                    continue

def _pick_model_symbol(mod) -> tuple[str, object] | None:
    best_class = None
    best_factory = None
    def _prefer_class(cand):
        nonlocal best_class
        name = cand.__name__
        if not _NAME_ALLOW_RE.search(name):  # 이름 필터
            return
        if _BAD_NAME_RE.search(name):        # 블록/레이어 제외
            return
        score = 0
        sig = inspect.signature(getattr(cand, "__init__", lambda: None))
        params = {p.name for p in sig.parameters.values()}
        if {"variant","model_variant","backbone","encoder"} & params: score += 2
        if "pretrained" in params or "ckpt" in params: score += 1
        if best_class is None or getattr(best_class, "_pick_score", -1) < score:
            cand._pick_score = score  # type: ignore
            best_class = cand

    def _prefer_factory(func):
        nonlocal best_factory
        fname = func.__name__.lower()
        if not any(k in fname for k in ("build","create","get","make")): return
        if "model" not in fname: return
        sig = inspect.signature(func)
        params = {p.name for p in sig.parameters.values()}
        score = 0
        if {"variant","model_variant","backbone","encoder"} & params: score += 2
        if "pretrained" in params or "ckpt" in params: score += 1
        if best_factory is None or getattr(best_factory, "_pick_score", -1) < score:
            func._pick_score = score  # type: ignore
            best_factory = func

    for k, v in vars(mod).items():
        if inspect.isclass(v) and issubclass(v, nn.Module):
            _prefer_class(v)
        elif callable(v):
            _prefer_factory(v)

    if best_class is not None:
        return ("class", best_class)
    if best_factory is not None:
        return ("factory", best_factory)
    return None

def resolve_depthanything_entry(root_dir: Optional[str]) -> Tuple[str, object, list]:
    """
    순서:
      1) depth_anything_v2.dav2:DepthAnythingV2 (정석)
      2) depth_anything_v2:DepthAnythingV2 / 팩토리
      3) depth_anything:... (구버전 대비)
      4) 파일 전수 스캔 → 이름/시그니처 필터
    """
    tried = []
    _add_paths(root_dir)

    # 1) 정석 경로
    m = _safe_import("depth_anything_v2.dav2")
    if m and hasattr(m, "DepthAnythingV2"):
        return "class", getattr(m, "DepthAnythingV2"), tried
    if m: tried.append("depth_anything_v2.dav2: DepthAnythingV2 missing")

    # 2) 패키지 루트에서 찾기
    for pkg in ("depth_anything_v2", "depth_anything"):
        m = _safe_import(pkg)
        if not m:
            tried.append(f"{pkg}: import fail"); continue
        if hasattr(m, "DepthAnythingV2"):
            return "class", getattr(m, "DepthAnythingV2"), tried
        picked = _pick_model_symbol(m)
        if picked: return picked[0], picked[1], tried
        tried.append(f"{pkg}: no suitable symbol")

    # 3) 파일 전수 스캔
    for sm in _scan_modules_by_path(root_dir or ""):
        picked = _pick_model_symbol(sm)
        if picked: return picked[0], picked[1], tried

    raise ImportError("Depth-Anything-V2 import 실패.\n  tried:\n  " + "\n  ".join(tried))

def build_depth_model(root_dir: Optional[str], variant: str, device: str, fp16: bool, ckpt_path: str):
    kind, sym, tried = resolve_depthanything_entry(root_dir)

    # 인스턴스 생성 (클래스/팩토리 모두 다양한 시그니처 대응)
    model = None
    tried_calls = []
    if kind == "class":
        call_kwargs_list = [
            {"variant": variant, "max_depth": None},
            {"model_variant": variant},
            {"backbone": variant},
            {"encoder": variant},
            {},
        ]
        for kw in call_kwargs_list:
            try:
                model = sym(**kw); break
            except TypeError as e:
                tried_calls.append(f"class {getattr(sym,'__name__','?')}{kw}: {e}")
    else:  # factory
        call_kwargs_list = [
            {"variant": variant},
            {"model_variant": variant},
            {"backbone": variant},
            {"encoder": variant},
            {},
        ]
        for kw in call_kwargs_list:
            try:
                model = sym(**kw); break
            except TypeError as e:
                tried_calls.append(f"factory {getattr(sym,'__name__','?')}{kw}: {e}")
    if model is None:
        raise TypeError("DepthAnything 모델/팩토리 호출 실패:\n  " + "\n  ".join(tried_calls))

    # 체크포인트 로드
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # state_dict 적용 (inner fallback 포함)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        inner = getattr(model, "model", None)
        if inner is not None:
            inner.load_state_dict(sd, strict=False)
        else:
            raise

    model = model.to(device).eval()
    if fp16:
        try: model = model.half()
        except Exception: pass
    try: model = model.to(memory_format=torch.channels_last)
    except Exception: pass
    return model
# ==================== /DepthAnything loader ====================

# ---- Gaussian / Morph ----
def _gaussian_1d(sigma: float):
    if sigma <= 0.0:
        k = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        return k / k.sum()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius+1, device='cuda', dtype=torch.float32)
    k = torch.exp(-(x*x) / (2.0*sigma*sigma))
    k = k / k.sum()
    return k

def gaussian_blur_alpha(alpha01, sigma, cache):
    if sigma <= 0.0: return alpha01
    if cache.get('sigma') != sigma:
        k = _gaussian_1d(sigma)
        cache['sigma'] = sigma
        cache['kx'] = k.view(1,1,1,-1)
        cache['ky'] = k.view(1,1,-1,1)
    kx, ky = cache['kx'], cache['ky']
    x = alpha01.view(1,1,alpha01.shape[0], alpha01.shape[1])
    x = F.conv2d(x, kx, padding=(0, kx.shape[-1]//2))
    x = F.conv2d(x, ky, padding=(ky.shape[-2]//2, 0))
    return x[0,0]

def morph_alpha(alpha01: torch.Tensor, px: int) -> torch.Tensor:
    if px == 0: return alpha01
    k = abs(int(px)); ks = 2*k+1
    x = alpha01[None, None, ...]
    if px > 0:  x = torch.nn.functional.max_pool2d(x, kernel_size=ks, stride=1, padding=k)
    else:       x = -torch.nn.functional.max_pool2d(-x, kernel_size=ks, stride=1, padding=k)
    return x[0,0].clamp(0.0, 1.0)

# ---- UYVY422 → RGB (CUDA) ----
def uyvy422_to_rgb_u8_cuda(u8_uyvy, H, W, device):
    U  = u8_uyvy[..., 0].to(torch.float32)
    Y0 = u8_uyvy[..., 1].to(torch.float32)
    V  = u8_uyvy[..., 2].to(torch.float32)
    Y1 = u8_uyvy[..., 3].to(torch.float32)
    Y = torch.stack([Y0, Y1], dim=-1).reshape(H, W)
    U = (U - 128.0).repeat_interleave(2, dim=-1)
    V = (V - 128.0).repeat_interleave(2, dim=-1)
    R = Y + 1.4020 * V
    G = Y - 0.344136 * U - 0.714136 * V
    B = Y + 1.7720 * U
    return torch.stack([R, G, B], dim=-1).clamp(0, 255).to(torch.uint8)

# ---- Depth packing ----
def pack_depth_to_rg16a(depth01_hw: torch.Tensor, frame_id_byte: int):
    d16 = (depth01_hw.clamp(0,1) * 65535.0 + 0.5).to(torch.uint16)
    lo = (d16 & 0xFF).to(torch.uint8)
    hi = ((d16 >> 8) & 0xFF).to(torch.uint8)
    B  = torch.full_like(lo, int(frame_id_byte) & 255)
    A  = torch.full_like(lo, 255)
    return torch.stack([hi, lo, B, A], dim=-1).contiguous()

# ---- FFmpeg DeckLink ----
def open_ffmpeg_pipe(ffmpeg_bin, device, size, pixfmt="bgra", threads=6,
                     decklink_format_code=None, decklink_connection=None,
                     force_output_fps=None, thread_queue_size=2048):
    W, H = [int(x) for x in size.lower().split("x")]
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-probesize", "32", "-analyzeduration", "0",
        "-thread_queue_size", str(int(thread_queue_size)),
        "-threads", str(int(threads)),
        "-f", "decklink"
    ]
    if decklink_format_code: cmd += ["-format_code", str(decklink_format_code)]
    if decklink_connection:  cmd += ["-video_input", str(decklink_connection)]
    cmd += ["-i", device, "-pix_fmt", pixfmt]
    if force_output_fps:     cmd += ["-r", str(force_output_fps)]
    cmd += ["-f", "rawvideo", "pipe:1"]
    print("[FFmpeg]", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    if   pixfmt.lower() in ("bgra","rgba"): bpp = 4
    elif pixfmt.lower() in ("bgr24","rgb24"): bpp = 3
    elif pixfmt.lower() in ("uyvy422",):     bpp = 2
    else: raise ValueError(f"Unsupported pixfmt for bpp: {pixfmt}")
    return proc, W*H*bpp, W, H, pixfmt

# ---- RVM loader ----
def load_rvm(weights, device="cuda", backbone="mobilenetv3", fp16=True):
    try:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net = MattingNetwork("mobilenetv3" if backbone=="mobilenetv3" else "resnet50")
    net.load_state_dict(sd, strict=False)
    net = net.to(device).eval()
    if fp16: net = net.half()
    try: net = net.to(memory_format=torch.channels_last)
    except Exception: pass
    return net

# ---- DepthAnything loader ----
def load_depth_anything_v2(root_dir, ckpt_path, variant="vits", device="cuda", fp16=True):
    return build_depth_model(root_dir, variant, device, fp16, ckpt_path)

# ---- Main ----
def run(ffmpeg_bin, decklink_device, force_size, raw_format,
        weights, cuda_index=0, long_side=640,
        decklink_format_code=None, decklink_connection=None, force_output_fps=None,
        spout_name_alpha="RT_RVM_ALPHA",
        spout_name_depth="RT_DEPTH_RG16",
        tx_delay_frames=0, tx_delay_max=120, tx_delay_ms=0,
        fps_cap=0.0, stats=1, spout_framesync=True, ffmpeg_threads=6, enable_cmd=1,
        alpha_ema=0.60, alpha_unsharp=0.25, alpha_sigma=0.75,
        # DepthAnything V2
        depth_enable=True,
        depth_root=None, depth_ckpt=None, depth_variant="vits",
        depth_in_size=320, depth_stride=2, depth_ema=0.85,
        depth_min_m=0.0, depth_max_m=0.0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cuda_index))
    torch.cuda.set_device(int(cuda_index))
    device = "cuda"

    # GL
    win = gl_init_hidden()

    # FFmpeg
    proc, frame_bytes, W, H, fmt = open_ffmpeg_pipe(
        ffmpeg_bin, decklink_device, force_size, pixfmt=raw_format,
        decklink_format_code=decklink_format_code,
        decklink_connection=decklink_connection,
        force_output_fps=force_output_fps,
        threads=ffmpeg_threads
    )

    # Spout senders
    s_alpha = SpoutSender(); s_alpha.setSenderName(spout_name_alpha)
    s_depth = SpoutSender(); s_depth.setSenderName(spout_name_depth)
    if spout_framesync:
        try: s_alpha.setFrameSync(True)
        except Exception: pass
        try: s_depth.setFrameSync(True)
        except Exception: pass

    # GL textures + PBOs
    tex_a = make_tex_rgba8(W, H); pbo_a = make_pbo(W*H*4); inter_a = CudaGLPBO(pbo_a, W*H*4)
    tex_d = make_tex_rgba8(W, H); pbo_d = make_pbo(W*H*4); inter_d = CudaGLPBO(pbo_d, W*H*4)
    print(f"[Spout] alpha='{spout_name_alpha}', depth='{spout_name_depth}'  ({W}x{H})")

    # Models
    rvm = load_rvm(weights, device=device, backbone="mobilenetv3", fp16=True)
    rec = [None, None, None, None]
    downsample = float(long_side) / float(max(W, H))

    depth_model = None
    if depth_enable:
        if not (depth_root and depth_ckpt):
            print("[Depth] disabled: ckpt/root not provided"); depth_enable = False
        else:
            depth_model = load_depth_anything_v2(depth_root, depth_ckpt, depth_variant, device=device, fp16=True)

    # Delay ring (timestamp,tensor)
    ring = collections.deque(maxlen=int(tx_delay_max))
    delay_frames = int(tx_delay_frames)
    delay_ms = max(0, int(tx_delay_ms))

    # Alpha post
    alpha_prev = None
    g_cache = {}
    edge_px = 0
    feather_sigma = 0.0
    edge_contrast = 1.0
    matte_gamma = 1.0
    clip_lo, clip_hi = 0.0, 1.0

    # Depth state
    depth_prev = None
    frame_id = 0
    _depth_frame_skip = 0
    _last_depth_rgba = None

    # FPS tracking
    last_print = 0.0
    fps_cap_val = parse_fps(fps_cap) if fps_cap else 0.0
    fps_ema = fps_cap_val if fps_cap_val>0 else 0.0
    fps_alpha = 0.10

    # stdin
    cmd_queue = queue.Queue()
    def start_cmd_thread(cmd_queue):
        print(
            "commands:\n"
            " delay N|+1|-1 | delayms N | ema X | sharp X | sigma X | long N | cap FPS | stats 0/1\n"
            " now/! | flush | panic | edge N | feather X | contrast X | gamma X | clip L H\n"
            " d.en 0/1 | d.stride N | d.size N | d.ema X | d.minmax L H | quit\n", flush=True)
        def _worker():
            while True:
                try: line = sys.stdin.readline()
                except Exception: break
                if not line:
                    time.sleep(0.05); continue
                cmd_queue.put(line.strip())
        threading.Thread(target=_worker, daemon=True).start()
    if enable_cmd:
        start_cmd_thread(cmd_queue)

    def frames_to_ms(frames, ref_fps): return int(round(frames * 1000.0 / (ref_fps if ref_fps>0 else 60.0)))
    def ms_to_frames(ms, ref_fps):     return ms * (ref_fps if ref_fps>0 else 60.0) / 1000.0
    frame_interval = (1.0/fps_cap_val) if fps_cap_val>0 else 0.0
    catch_up_once = False
    startup_pending_frames_to_ms = (delay_ms == 0 and delay_frames > 0)

    def _tick(dt):
        nonlocal last_print, fps_ema
        if dt <= 0: return
        dt = max(dt, 1.0/240.0)
        inst = 1.0/dt
        fps_ema = inst if fps_ema<=0 else (1.0-fps_alpha)*fps_ema + fps_alpha*inst
        now = time.time()
        if stats and (now - last_print) >= 0.3:
            ref_fps = fps_cap_val if fps_cap_val>0 else fps_ema
            dtxt = f"{delay_ms}ms (~{ms_to_frames(delay_ms, ref_fps):.1f}f @ {ref_fps:.2f}fps)" if delay_ms>0 else f"{delay_frames}f (legacy)"
            print(f"[FPS] {inst:5.1f} | refFPS={ref_fps:.2f} | delay={dtxt} | "
                  f"ema={alpha_ema:.2f} | sharp={alpha_unsharp:.2f} | sigma={alpha_sigma:.2f} | "
                  f"edge={edge_px} | feather={feather_sigma:.2f} | contrast={edge_contrast:.2f} | "
                  f"gamma={matte_gamma:.2f} | clip=({clip_lo:.2f},{clip_hi:.2f}) | long={int(long_side)} | "
                  f"DEPTH[en={int(depth_enable)} stride={int(depth_stride)} size={int(depth_in_size)} ema={depth_ema:.2f}]")
            last_print = now

    def select_time_delayed(now_s: float, target_ms: int):
        if not ring: return None
        if target_ms <= 0: return ring[-1][1]
        target_s = target_ms / 1000.0
        for t, f in reversed(ring):
            if (now_s - t) >= target_s: return f
        return ring[0][1]

    def handle_cmd(line: str):
        nonlocal delay_frames, delay_ms, fps_cap_val, frame_interval, fps_ema
        nonlocal stats, alpha_ema, alpha_unsharp, alpha_sigma, long_side, downsample
        nonlocal catch_up_once, ring
        nonlocal edge_px, feather_sigma, edge_contrast, matte_gamma, clip_lo, clip_hi
        nonlocal depth_enable, depth_stride, depth_in_size, depth_ema, depth_min_m, depth_max_m
        parts = line.strip().split()
        if not parts: return
        cmd = parts[0].lower()
        try:
            if cmd == "delayms" and len(parts)==2:
                delay_ms = max(0, int(float(parts[1]))); print(f"[CMD] delay_ms={delay_ms}")
            elif cmd == "delay":
                ref_fps = fps_cap_val if fps_cap_val>0 else (fps_ema if fps_ema>0 else 60.0)
                if len(parts)==2 and parts[1].startswith(("+","-")):
                    delay_frames = max(0, delay_frames + int(parts[1]))
                elif len(parts)==2:
                    delay_frames = max(0, int(float(parts[1])))
                delay_ms = frames_to_ms(delay_frames, ref_fps)
                print(f"[CMD] delay_frames={delay_frames} → delay_ms={delay_ms} (@{ref_fps:.2f}fps)")
            elif cmd == "ema" and len(parts)==2:
                alpha_ema = max(0.0, min(1.0, float(parts[1]))); print(f"[CMD] alpha_ema={alpha_ema}")
            elif cmd == "sharp" and len(parts)==2:
                alpha_unsharp = max(0.0, float(parts[1])); print(f"[CMD] alpha_unsharp={alpha_unsharp}")
            elif cmd == "sigma" and len(parts)==2:
                alpha_sigma = max(0.0, float(parts[1])); print(f"[CMD] alpha_sigma={alpha_sigma}")
            elif cmd == "long" and len(parts)==2:
                long_side = max(64, int(float(parts[1]))); downsample = float(long_side) / float(max(W, H))
                print(f"[CMD] long_side={long_side} (downsample_ratio={downsample:.4f})")
            elif cmd == "cap" and len(parts)==2:
                fps_cap_val = parse_fps(parts[1]); frame_interval = (1.0/fps_cap_val) if fps_cap_val>0 else 0.0
                print(f"[CMD] fps_cap={fps_cap_val}")
            elif cmd == "stats" and len(parts)==2:
                stats = int(float(parts[1])) != 0; print(f"[CMD] stats={int(stats)}")
            elif cmd in ("now","!"):
                catch_up_once = True; print("[CMD] one-shot catch-up")
            elif cmd == "flush":
                ring.clear(); print("[CMD] ring flushed")
            elif cmd == "panic":
                delay_ms = 0; delay_frames = 0; print("[CMD] PANIC delay=0")
            elif cmd == "edge" and len(parts)==2:
                edge_px = int(float(parts[1])); print(f"[CMD] edge_px={edge_px}")
            elif cmd == "feather" and len(parts)==2:
                feather_sigma = max(0.0, float(parts[1])); print(f"[CMD] feather_sigma={feather_sigma}")
            elif cmd == "contrast" and len(parts)==2:
                edge_contrast = max(0.01, float(parts[1])); print(f"[CMD] edge_contrast={edge_contrast}")
            elif cmd == "gamma" and len(parts)==2:
                matte_gamma = max(0.01, float(parts[1])); print(f"[CMD] matte_gamma={matte_gamma}")
            elif cmd == "clip" and len(parts)==3:
                lo = float(parts[1]); hi = float(parts[2])
                lo = max(0.0, min(1.0, lo)); hi = max(0.0, min(1.0, hi))
                if hi <= lo: lo, hi = 0.0, 1.0
                clip_lo, clip_hi = lo, hi; print(f"[CMD] clip=({clip_lo:.3f},{clip_hi:.3f})")
            # depth
            elif cmd == "d.en" and len(parts)==2:
                depth_enable = bool(int(float(parts[1]))); print(f"[CMD] depth_enable={int(depth_enable)}")
            elif cmd == "d.stride" and len(parts)==2:
                depth_stride = max(1, int(float(parts[1]))); print(f"[CMD] depth_stride={depth_stride}")
            elif cmd == "d.size" and len(parts)==2:
                depth_in_size = max(64, int(float(parts[1]))); print(f"[CMD] depth_in_size={depth_in_size}")
            elif cmd == "d.ema" and len(parts)==2:
                depth_ema = max(0.0, min(1.0, float(parts[1]))); print(f"[CMD] depth_ema={depth_ema}")
            elif cmd == "d.minmax" and len(parts)==3:
                depth_min_m = max(0.0, float(parts[1])); depth_max_m = max(0.0, float(parts[2]))
                if depth_max_m>0 and depth_min_m>=depth_max_m: depth_min_m, depth_max_m = 0.0, 0.0
                print(f"[CMD] depth_min_m={depth_min_m:.2f}, depth_max_m={depth_max_m:.2f} (0=auto)")
            elif cmd == "quit":
                print("[CMD] quit"); raise KeyboardInterrupt
            else:
                print(f"[CMD] unknown: {line}")
        except Exception as e:
            print(f"[CMD] error: {e}")

    buf = bytearray(frame_bytes)
    mv = memoryview(buf)
    t_prev = time.time()

    try:
        while True:
            try:
                while True:
                    handle_cmd(cmd_queue.get_nowait())
            except queue.Empty:
                pass

            # read one frame
            got = 0
            while got < frame_bytes:
                n = proc.stdout.readinto(mv[got:frame_bytes])
                if not n: raise RuntimeError("FFmpeg pipe closed")
                got += n

            t_now = time.time()
            _tick(t_now - t_prev); t_prev = t_now

            if startup_pending_frames_to_ms and (fps_cap_val > 0 or fps_ema > 0):
                ref_fps = fps_cap_val if fps_cap_val>0 else fps_ema
                delay_ms = int(round(delay_frames * 1000.0 / ref_fps))
                print(f"[Init] delay_frames={delay_frames} → delay_ms={delay_ms} (@{ref_fps:.2f}fps)")
                startup_pending_frames_to_ms = False

            # CPU→GPU upload
            if fmt.lower() == "uyvy422":
                arr = np.frombuffer(mv, dtype=np.uint8, count=frame_bytes).reshape(H, W//2, 4)
                t_u8 = torch.from_numpy(arr)
                try: t_u8 = t_u8.pin_memory()
                except Exception: pass
                uyvy = t_u8.to(device, non_blocking=True)
                rgb_u8 = uyvy422_to_rgb_u8_cuda(uyvy, H, W, device)
            else:
                ch = 4 if fmt.lower() in ("bgra","rgba") else 3
                arr = np.frombuffer(mv, dtype=np.uint8, count=frame_bytes).reshape(H, W, ch)
                t_u8 = torch.from_numpy(arr)
                try: t_u8 = t_u8.pin_memory()
                except Exception: pass
                if fmt.lower() == "bgra":   rgb_u8 = t_u8[..., [2,1,0]].to(device, non_blocking=True)
                elif fmt.lower() == "rgba": rgb_u8 = t_u8[..., [0,1,2]].to(device, non_blocking=True)
                elif fmt.lower() == "bgr24":rgb_u8 = t_u8[..., [2,1,0]].to(device, non_blocking=True)
                else:                        rgb_u8 = t_u8.to(device, non_blocking=True)

            # RVM inference
            x = (rgb_u8.permute(2,0,1).contiguous().to(torch.float16)*(1/255.0)).unsqueeze(0).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                fgr, pha, *rec = rvm(x, *rec, downsample_ratio=float(long_side)/float(max(W,H)))
            alpha = pha[0,0].float().clamp(0,1)

            # Alpha post
            alpha_prev = alpha if alpha_prev is None else alpha_ema*alpha_prev + (1.0-alpha_ema)*alpha
            alpha_smooth = alpha_prev
            if alpha_unsharp>0.0:
                blurred = gaussian_blur_alpha(alpha_smooth, float(alpha_sigma), g_cache)
                alpha_smooth = torch.clamp(alpha_smooth + alpha_unsharp*(alpha_smooth - blurred), 0.0, 1.0)
            if edge_px!=0:        alpha_smooth = morph_alpha(alpha_smooth, edge_px)
            if feather_sigma>0.0: alpha_smooth = gaussian_blur_alpha(alpha_smooth, float(feather_sigma), g_cache)
            if abs(edge_contrast-1.0)>1e-3: alpha_smooth = torch.sigmoid((alpha_smooth-0.5)*8.0*float(edge_contrast))
            if abs(matte_gamma-1.0)>1e-3:   alpha_smooth = torch.clamp(alpha_smooth,0,1)**float(matte_gamma)
            if (clip_lo>0.0) or (clip_hi<1.0):
                lo, hi = clip_lo, clip_hi
                alpha_smooth = torch.clamp(alpha_smooth, lo, hi); alpha_smooth = (alpha_smooth-lo)/max(1e-6,(hi-lo))

            a_u8 = (alpha_smooth*255.0 + 0.5).to(torch.uint8)
            rgba_alpha = torch.cat([torch.zeros((H,W,3), dtype=torch.uint8, device=device), a_u8.unsqueeze(-1)], dim=2).contiguous()

            ring.append((t_now, rgba_alpha))
            out_alpha = ring[-1][1] if (delay_ms<=0 or catch_up_once) else (select_time_delayed(time.time(), delay_ms) or ring[-1][1])
            catch_up_once = False

            # Depth
            depth_rgba = None
            if depth_enable and depth_model is not None:
                do_depth = (_depth_frame_skip == 0)
                _depth_frame_skip = (_depth_frame_skip + 1) % max(1, int(depth_stride))
                if do_depth:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        d_in = (rgb_u8.to(torch.float16)*(1/255.0)).permute(2,0,1).unsqueeze(0)
                        Hf, Wf = float(H), float(W)
                        if Hf < Wf:
                            new_h = depth_in_size; new_w = int(round(Wf*(depth_in_size/Hf)))
                        else:
                            new_w = depth_in_size; new_h = int(round(Hf*(depth_in_size/Wf)))
                        d_in = F.interpolate(d_in, size=(new_h,new_w), mode="bilinear", align_corners=False)
                        top  = max(0,(new_h-depth_in_size)//2); left = max(0,(new_w-depth_in_size)//2)
                        d_in = d_in[..., top:top+depth_in_size, left:left+depth_in_size]
                        pred = depth_model(d_in)
                        if isinstance(pred, dict):
                            dep = next((v for k,v in pred.items() if isinstance(v, torch.Tensor)), None)
                            dep = dep[0,0].float() if dep is not None and dep.ndim==4 else pred[list(pred.keys())[0]][0,0].float()
                        elif isinstance(pred, (list,tuple)):
                            dep = pred[0][0,0].float()
                        else:
                            dep = pred[0,0].float()

                    if depth_min_m>0.0 and depth_max_m>0.0:
                        dep01 = torch.clamp((dep - dep.min())/(dep.max()-dep.min()+1e-6),0,1)
                    else:
                        flat = dep.view(-1)
                        lo = torch.quantile(flat, 0.02); hi = torch.quantile(flat, 0.98)
                        dep01 = torch.clamp((dep - lo) / (hi - lo + 1e-6), 0, 1)

                    depth_prev = dep01 if depth_prev is None else depth_ema*depth_prev + (1.0-depth_ema)*dep01
                    dep_hw = F.interpolate(depth_prev[None,None,...], size=(H,W), mode="bilinear", align_corners=False)[0,0]
                    depth_rgba = pack_depth_to_rg16a(dep_hw, frame_id & 255)

            frame_id = (frame_id + 1) & 0xFFFFFFFF

            # Spout send: ALPHA
            inter_a.upload_from_torch_u8(out_alpha)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_a)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_a)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            SpoutSender.sendTexture(s_alpha, int(tex_a), int(GL.GL_TEXTURE_2D), W, H, False, int(GL_RGBA))
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

            # Spout send: DEPTH (hold last)
            if depth_enable and depth_model is not None:
                if depth_rgba is not None: _last_depth_rgba = depth_rgba
                if _last_depth_rgba is not None:
                    inter_d.upload_from_torch_u8(_last_depth_rgba)
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_d)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_d)
                    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
                    SpoutSender.sendTexture(s_depth, int(tex_d), int(GL.GL_TEXTURE_2D), W, H, False, int(GL_RGBA))
                    GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

            # optional cap
            if frame_interval > 0.0:
                remain = frame_interval - (time.time() - t_now)
                if remain > 0: time.sleep(remain)

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt")
    finally:
        try:
            if proc and proc.poll() is None: proc.terminate()
        except Exception: pass
        for s in (s_alpha, s_depth):
            try: s.release()
            except Exception: pass
        for t in (tex_a, tex_d):
            try: GL.glDeleteTextures([t])
            except Exception: pass
        for p in (pbo_a, pbo_d):
            try: GL.glDeleteBuffers(1, [p])
            except Exception: pass
        try: glfw.terminate()
        except Exception: pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    # 입력
    p.add_argument("--ffmpeg-bin", type=str, required=True)
    p.add_argument("--decklink-device", type=str, required=True)
    p.add_argument("--force-size", type=str, default="1920x1080")
    p.add_argument("--raw-format", type=str, default="bgra", choices=["bgra","rgba","bgr24","rgb24","uyvy422"])
    p.add_argument("--decklink-format-code", type=str, default=None)
    p.add_argument("--decklink-connection", type=str, default=None)
    p.add_argument("--force-output-fps", type=str, default=None)
    p.add_argument("--ffmpeg-threads", type=int, default=6)
    # RVM
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--cuda-index", type=int, default=0)
    p.add_argument("--long", type=int, default=640)
    # Spout
    p.add_argument("--spout-name-alpha", type=str, default="RT_RVM_ALPHA")
    p.add_argument("--spout-name-depth", type=str, default="RT_DEPTH_RG16")
    # Delay
    p.add_argument("--tx-delay-frames", type=int, default=0)
    p.add_argument("--tx-delay-max", type=int, default=120)
    p.add_argument("--tx-delay-ms", type=int, default=0)
    # 기타
    p.add_argument("--fps-cap", type=str, default="0")
    p.add_argument("--stats", type=int, default=1)
    p.add_argument("--spout-framesync", type=int, default=1)
    p.add_argument("--enable-cmd", type=int, default=1)
    # Alpha tuning
    p.add_argument("--alpha-ema", type=float, default=0.60)
    p.add_argument("--alpha-unsharp", type=float, default=0.25)
    p.add_argument("--alpha-sigma", type=float, default=0.75)
    # Depth
    p.add_argument("--depth-enable", type=int, default=1)
    p.add_argument("--depth-anything-root", type=str, default=None)
    p.add_argument("--depth-anything-ckpt", type=str, default=None)
    p.add_argument("--depth-anything-variant", type=str, default="vits",
                   choices=["vits","vitb","vitl","swinv2l","repvit"])
    p.add_argument("--depth-in-size", type=int, default=320)
    p.add_argument("--depth-stride", type=int, default=2)
    p.add_argument("--depth-ema", type=float, default=0.85)
    p.add_argument("--depth-min-m", type=float, default=0.0)
    p.add_argument("--depth-max-m", type=float, default=0.0)

    args = p.parse_args()

    run(
        ffmpeg_bin=args.ffmpeg_bin,
        decklink_device=args.decklink_device,
        force_size=args.force_size,
        raw_format=args.raw_format,
        weights=args.weights,
        cuda_index=args.cuda_index,
        long_side=args.long,
        decklink_format_code=args.decklink_format_code,
        decklink_connection=args.decklink_connection,
        force_output_fps=args.force_output_fps,
        spout_name_alpha=args.spout_name_alpha,
        spout_name_depth=args.spout_name_depth,
        tx_delay_frames=args.tx_delay_frames,
        tx_delay_max=args.tx_delay_max,
        tx_delay_ms=args.tx_delay_ms,
        fps_cap=args.fps_cap,
        stats=args.stats,
        spout_framesync=bool(args.spout_framesync),
        ffmpeg_threads=args.ffmpeg_threads,
        enable_cmd=bool(args.enable_cmd),
        alpha_ema=args.alpha_ema,
        alpha_unsharp=args.alpha_unsharp,
        alpha_sigma=args.alpha_sigma,
        depth_enable=bool(args.depth_enable),
        depth_root=args.depth_anything_root,
        depth_ckpt=args.depth_anything_ckpt,
        depth_variant=args.depth_anything_variant,
        depth_in_size=args.depth_in_size,
        depth_stride=args.depth_stride,
        depth_ema=args.depth_ema,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
    )
