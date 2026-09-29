#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rvm_spout_fin.py  (Alpha + Video, GPU-first, UYVY→RGB on CUDA, SpoutGL,
Time-based Delay, Frame→ms Auto-Convert, one-shot catch-up/flush/panic,
Matte edge controls: erosion/dilation, feather, contrast, gamma, clip,
Preset save/load via 'save' command)

- Spout streams:
    1) spout_name_alpha : RGBA8 where RGB=0, A=alpha 0..255
    2) spout_name_video : RGBA8 of camera RGB with A=255

- Depth path 없음. CPU 사용 최소화, GPU 경로 최대화.
"""

import os, sys, time, subprocess, shlex, ctypes, numpy as np, threading, queue, collections, math
import torch
import torch.nn.functional as F

# ---- Preset file path ----
PRESET_FILE = os.path.join(os.path.dirname(__file__), "rvm_spout_preset.txt")

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
            raise RuntimeError("Could not load CUDA runtime (cudart). Ensure CUDA 12.x runtime DLLs are in PATH.")
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
        try:
            _ck(_cudart.cudaGraphicsUnregisterResource(self.res))
        except Exception:
            pass
        self.res = ctypes.c_void_p()

# ---- OpenGL helpers ----
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

# ---- TorchVision compat shim ----
import sys as _sys, types as _types
try:
    import torchvision as _tv
    from torch.hub import load_state_dict_from_url as _lsu
    import importlib as _importlib
    try:
        _tv_models = _importlib.import_module("torchvision.models")
    except Exception as _e:
        raise ModuleNotFoundError(f"torchvision 설치는 되어 있으나 'torchvision.models' 로드 실패: {_e!r}")
    if "torchvision.models.utils" not in _sys.modules:
        _utils_mod = _types.ModuleType("torchvision.models.utils")
        _utils_mod.load_state_dict_from_url = _lsu
        _sys.modules["torchvision.models.utils"] = _utils_mod
        setattr(_tv_models, "utils", _utils_mod)
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "torchvision이 없습니다. torch/CUDA와 같은 계열로 설치하세요.\n"
        "CUDA 12.1: pip install torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "CUDA 11.8: pip install torchvision --index-url https://download.pytorch.org/whl/cu118\n"
        "CPU-only:  pip install torchvision --index-url https://download.pytorch.org/whl/cpu"
    )

# ---- Robust Video Matting ----
RVM_DIR = os.path.join(os.path.dirname(__file__), "RobustVideoMatting")
_sys.path.insert(0, RVM_DIR)
from model import MattingNetwork

# ---- stdin command thread ----
def start_cmd_thread(cmd_queue):
    help_txt = (
        "commands:\n"
        "  delay N|+1|-1       (frame-based input; auto→time ms)\n"
        "  delayms N           (time-based delay in ms)\n"
        "  ema X.Y             (temporal EMA [0..1])\n"
        "  sharp X.Y           (unsharp amount)\n"
        "  sigma X.Y           (gaussian sigma for unsharp)\n"
        "  long N              (RVM long-side resolution)\n"
        "  cap FPS             (fps cap, 0=off)\n"
        "  stats 0/1           (toggle fps print)\n"
        "  now / !             (one-shot delay bypass)\n"
        "  flush               (clear buffer)\n"
        "  panic               (disable delay)\n"
        "  edge N              (+:dilate, -:erode)\n"
        "  feather X.Y         (extra feather sigma)\n"
        "  contrast X.Y        (>1 harder, <1 softer)\n"
        "  gamma X.Y           (alpha gamma)\n"
        "  clip L H            (clip & remap)\n"
        "  save                (save current settings to preset file)\n"
        "  quit                (exit)\n"
    )
    print(help_txt, flush=True)
    def _worker():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                time.sleep(0.05)
                continue
            cmd_queue.put(line.strip())
    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th

# ---- FFmpeg input (DeckLink / DirectShow / File) ----
def open_ffmpeg_pipe(ffmpeg_bin, device_or_path, size, pixfmt="bgra", threads=6,
                     decklink_format_code=None, decklink_connection=None,
                     force_output_fps=None, thread_queue_size=2048,
                     input_type="decklink"):
    """
    input_type:
      - "decklink"   : device_or_path = DeckLink device name
      - "directshow" : device_or_path = DirectShow device name (e.g. 'OBS Virtual Camera')
      - "file"       : device_or_path = media file path
    """
    W, H = [int(x) for x in size.lower().split("x")]
    pix = pixfmt.lower()

    if input_type == "directshow":
        # OBS Virtual Camera 등 DirectShow 입력
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-thread_queue_size", str(int(thread_queue_size)),
            "-threads", str(int(threads)),
            "-f", "dshow",
            "-video_size", size,
            "-i", f"video={device_or_path}",
            "-pix_fmt", pixfmt,
        ]
        if force_output_fps:
            cmd += ["-r", str(force_output_fps)]
        cmd += ["-f", "rawvideo", "pipe:1"]

    elif input_type == "file":
        # 파일 입력 (테스트용)
        # uyvy422는 의미 없으니 파일 입력일 때는 bgra/rgba/bgr24/rgb24 사용 권장
        vf = f"scale={W}:{H}"
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", str(int(threads)),
            "-i", device_or_path,
            "-vf", vf,
            "-pix_fmt", pixfmt,
            "-f", "rawvideo", "pipe:1"
        ]
        if force_output_fps:
            cmd.insert(-4, "-r")
            cmd.insert(-4, str(force_output_fps))

    else:
        # DeckLink 입력 (기존)
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-thread_queue_size", str(int(thread_queue_size)),
            "-threads", str(int(threads)),
            "-f", "decklink"
        ]
        if decklink_format_code:
            cmd += ["-format_code", str(decklink_format_code)]
        if decklink_connection:
            cmd += ["-video_input", str(decklink_connection)]
        cmd += ["-i", device_or_path, "-pix_fmt", pixfmt]
        if force_output_fps:
            cmd += ["-r", str(force_output_fps)]
        cmd += ["-f", "rawvideo", "pipe:1"]

    print("[FFmpeg]", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    if pix == "bgra" or pix == "rgba":
        bpp = 4
    elif pix in ("bgr24", "rgb24"):
        bpp = 3
    elif pix == "uyvy422":
        bpp = 2
    else:
        raise ValueError(f"Unsupported pixfmt for bpp: {pixfmt}")

    return proc, W*H*bpp, W, H, pixfmt

# ---- YUV(UYVY422) → RGB (CUDA) ----
def uyvy422_to_rgb_u8_cuda(u8_uyvy, H, W, device):
    # u8_uyvy: [H, W/2, 4]  U, Y0, V, Y1
    U  = u8_uyvy[..., 0].to(torch.float32)
    Y0 = u8_uyvy[..., 1].to(torch.float32)
    V  = u8_uyvy[..., 2].to(torch.float32)
    Y1 = u8_uyvy[..., 3].to(torch.float32)

    Y = torch.stack([Y0, Y1], dim=-1).reshape(H, W)
    U = (U - 128.0).repeat_interleave(2, dim=-1)
    V = (V - 128.0).repeat_interleave(2, dim=-1)

    # BT.601
    R = Y + 1.4020 * V
    G = Y - 0.344136 * U - 0.714136 * V
    B = Y + 1.7720 * U

    RGB = torch.stack([R, G, B], dim=-1).clamp(0, 255).to(torch.uint8)
    return RGB

# ---- Gaussian kernel helpers (for unsharp) ----
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
    if sigma <= 0.0:
        return alpha01
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

# ---- Matte edge morph (dilate/erode) ----
def morph_alpha(alpha01: torch.Tensor, px: int) -> torch.Tensor:
    if px == 0:
        return alpha01
    k = abs(int(px))
    ks = 2 * k + 1
    x = alpha01[None, None, ...]
    if px > 0:
        x = torch.nn.functional.max_pool2d(x, kernel_size=ks, stride=1, padding=k)
    else:
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=ks, stride=1, padding=k)
    return x[0,0].clamp(0.0, 1.0)

def load_rvm(weights, device="cuda", backbone="mobilenetv3", fp16=True):
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net = MattingNetwork("mobilenetv3" if backbone=="mobilenetv3" else "resnet50")
    net.load_state_dict(sd, strict=False)
    net = net.to(device).eval()
    if fp16: net = net.half()
    try:
        net = net.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return net

# ---- Main ----
def run(ffmpeg_bin, decklink_device, force_size, raw_format,
        weights, cuda_index=0, long_side=640,
        decklink_format_code=None, decklink_connection=None, force_output_fps=None,
        spout_name_alpha="RT_RVM_ALPHA",
        spout_name_video="RT_RVM_VIDEO",
        tx_delay_frames=0, tx_delay_max=120,
        tx_delay_ms=0,
        fps_cap=0.0, stats=1, spout_framesync=True, ffmpeg_threads=6,
        enable_cmd=1,
        alpha_ema=0.60, alpha_unsharp=0.25, alpha_sigma=0.75,
        input_type="decklink",
        directshow_device=None,
        input_file=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cuda_index))
    torch.cuda.set_device(int(cuda_index))
    device = "cuda"

    # GL + Spout
    win = gl_init_hidden()

    # FFmpeg pipe: 입력 종류 선택
    if input_type == "directshow":
        src = directshow_device or decklink_device
    elif input_type == "file":
        src = input_file
    else:
        src = decklink_device

    if not src:
        raise ValueError(f"input_type='{input_type}' 인데 유효한 입력 소스가 없습니다.")

    proc, frame_bytes, W, H, fmt = open_ffmpeg_pipe(
        ffmpeg_bin, src, force_size, pixfmt=raw_format,
        decklink_format_code=decklink_format_code,
        decklink_connection=decklink_connection,
        force_output_fps=force_output_fps,
        threads=ffmpeg_threads,
        input_type=input_type
    )

    # Spout senders
    s_alpha = SpoutSender()
    s_alpha.setSenderName(spout_name_alpha)
    s_video = SpoutSender()
    s_video.setSenderName(spout_name_video)
    if spout_framesync:
        try:
            s_alpha.setFrameSync(True)
            s_video.setFrameSync(True)
        except Exception:
            pass

    # GL textures + PBOs
    tex_a = make_tex_rgba8(W, H)
    pbo_a = make_pbo(W*H*4)
    inter_a = CudaGLPBO(pbo_a, W*H*4)

    tex_v = make_tex_rgba8(W, H)
    pbo_v = make_pbo(W*H*4)
    inter_v = CudaGLPBO(pbo_v, W*H*4)

    print(f"[Spout] ready alpha='{spout_name_alpha}', video='{spout_name_video}' ({W}x{H})")

    # RVM
    rvm = load_rvm(weights, device=device, backbone="mobilenetv3", fp16=True)
    rec = [None, None, None, None]
    downsample = float(long_side) / float(max(W, H))

    # TX delay ring buffer now stores (timestamp, frame_alpha, frame_video)
    ring = collections.deque(maxlen=int(tx_delay_max))
    delay_frames = int(tx_delay_frames)
    delay_ms = max(0, int(tx_delay_ms))

    # Temporal EMA & unsharp
    alpha_prev = None
    g_cache = {}

    # Matte edge controls
    edge_px = 0
    feather_sigma = 0.0
    edge_contrast = 1.0
    matte_gamma = 1.0
    clip_lo, clip_hi = 0.0, 1.0

    # FPS tracking
    last_print = 0.0
    fps_cap_val = parse_fps(fps_cap) if fps_cap else 0.0
    fps_ema = fps_cap_val if fps_cap_val > 0 else 0.0
    fps_alpha = 0.10

    # stdin command thread
    cmd_queue = queue.Queue()
    if enable_cmd:
        start_cmd_thread(cmd_queue)

    # one-shot catch-up flag
    catch_up_once = False

    def frames_to_ms(frames, ref_fps):
        if ref_fps <= 0:  # fallback
            ref_fps = 60.0
        return int(round(frames * 1000.0 / ref_fps))

    def ms_to_frames(ms, ref_fps):
        if ref_fps <= 0:
            ref_fps = 60.0
        return ms * ref_fps / 1000.0

    # fps cap → sleep interval
    frame_interval = (1.0/fps_cap_val) if fps_cap_val>0 else 0.0

    def _tick(dt):
        nonlocal last_print, fps_ema
        if dt <= 0: return
        dt = max(dt, 1.0/240.0)
        inst = 1.0/dt
        if fps_ema <= 0:
            fps_ema = inst
        else:
            fps_ema = (1.0 - fps_alpha)*fps_ema + fps_alpha*inst

        now = time.time()
        if stats and (now - last_print) >= 0.3:
            ref_fps = fps_cap_val if fps_cap_val>0 else fps_ema
            if delay_ms > 0:
                f_eq = ms_to_frames(delay_ms, ref_fps)
                dtxt = f"{delay_ms}ms (~{f_eq:.1f}f @ {ref_fps:.2f}fps)"
            else:
                dtxt = f"{delay_frames}f (legacy)"
            print(f"[FPS] {inst:5.1f} | refFPS={ref_fps:.2f} | delay={dtxt} | "
                  f"ema={alpha_ema:.2f} | sharp={alpha_unsharp:.2f} | sigma={alpha_sigma:.2f} | "
                  f"edge={edge_px} | feather={feather_sigma:.2f} | contrast={edge_contrast:.2f} "
                  f"| gamma={matte_gamma:.2f} | clip=({clip_lo:.2f},{clip_hi:.2f}) | long={int(long_side)}")
            last_print = now

    def select_time_delayed(now_s: float, target_ms: int):
        if not ring:
            return None
        if target_ms <= 0:
            return ring[-1][1], ring[-1][2]
        target_s = target_ms / 1000.0
        for t, fa, fv in reversed(ring):
            if (now_s - t) >= target_s:
                return fa, fv
        return ring[0][1], ring[0][2]

    def handle_cmd(line: str):
        nonlocal delay_frames, delay_ms, fps_cap_val, frame_interval, fps_ema
        nonlocal stats, alpha_ema, alpha_unsharp, alpha_sigma, long_side, downsample
        nonlocal catch_up_once, ring
        nonlocal edge_px, feather_sigma, edge_contrast, matte_gamma, clip_lo, clip_hi
        parts = line.strip().split()
        if not parts: return
        cmd = parts[0].lower()
        try:
            if cmd == "delayms" and len(parts)==2:
                delay_ms = max(0, int(float(parts[1])))
                print(f"[CMD] delay_ms={delay_ms} ms (time-based)")
            elif cmd == "delay":
                ref_fps = fps_cap_val if fps_cap_val>0 else (fps_ema if fps_ema>0 else 60.0)
                if len(parts)==2 and parts[1].startswith(("+","-")):
                    step = int(parts[1])
                    delay_frames = max(0, delay_frames + step)
                elif len(parts)==2:
                    delay_frames = max(0, int(float(parts[1])))
                delay_ms = frames_to_ms(delay_frames, ref_fps)
                print(f"[CMD] delay_frames={delay_frames} → delay_ms={delay_ms} (@refFPS={ref_fps:.2f})")
            elif cmd == "ema" and len(parts)==2:
                alpha_ema = max(0.0, min(1.0, float(parts[1])))
                print(f"[CMD] alpha_ema={alpha_ema}")
            elif cmd == "sharp" and len(parts)==2:
                alpha_unsharp = max(0.0, float(parts[1]))
                print(f"[CMD] alpha_unsharp={alpha_unsharp}")
            elif cmd == "sigma" and len(parts)==2:
                alpha_sigma = max(0.0, float(parts[1])); g_cache.clear()
                print(f"[CMD] alpha_sigma={alpha_sigma}")
            elif cmd == "long" and len(parts)==2:
                long_side = max(64, int(float(parts[1]))); downsample = float(long_side) / float(max(W, H))
                print(f"[CMD] long_side={long_side} (downsample_ratio={downsample:.4f})")
            elif cmd == "cap" and len(parts)==2:
                fps_cap_val = parse_fps(parts[1]); frame_interval = (1.0/fps_cap_val) if fps_cap_val>0 else 0.0
                print(f"[CMD] fps_cap={fps_cap_val}")
            elif cmd == "stats" and len(parts)==2:
                stats = int(float(parts[1])) != 0; print(f"[CMD] stats={int(stats)}")
            elif cmd in ("now", "!"):
                catch_up_once = True; print("[CMD] one-shot catch-up: next output will use the LATEST frame")
            elif cmd == "flush":
                ring.clear(); print("[CMD] ring buffer flushed")
            elif cmd == "panic":
                delay_ms = 0; delay_frames = 0; print("[CMD] PANIC: delay disabled")
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
                clip_lo, clip_hi = lo, hi; print(f"[CMD] clip_lo={clip_lo:.3f}, clip_hi={clip_hi:.3f}")
            elif cmd == "save":
                cmds = [
                    f"ema {alpha_ema}",
                    f"sharp {alpha_unsharp}",
                    f"sigma {alpha_sigma}",
                    f"edge {edge_px}",
                    f"feather {feather_sigma}",
                    f"contrast {edge_contrast}",
                    f"gamma {matte_gamma}",
                    f"clip {clip_lo} {clip_hi}",
                    f"long {long_side}",
                    f"delayms {delay_ms}",
                    f"cap {fps_cap_val}",
                ]
                try:
                    with open(PRESET_FILE, "w", encoding="utf-8") as f:
                        for c in cmds:
                            f.write(c + "\n")
                    print(f"[CMD] preset saved to {PRESET_FILE}")
                except Exception as e:
                    print(f"[CMD] preset save error: {e}")
            elif cmd == "help":
                print("delay N|+1|-1 ; delayms N ; ema X.Y ; sharp X.Y ; sigma X.Y ; long N ; cap FPS ; "
                      "stats 0/1 ; now/! ; flush ; panic ; edge N ; feather X.Y ; contrast X.Y ; gamma X.Y ; "
                      "clip L H ; save ; quit")
            elif cmd == "quit":
                print("[CMD] quit requested"); raise KeyboardInterrupt
            else:
                print(f"[CMD] unknown: {line}")
        except Exception as e:
            print(f"[CMD] error: {e}")

    # ---- Startup: load preset file if exists ----
    if os.path.exists(PRESET_FILE):
        print(f"[PRESET] loading from {PRESET_FILE}")
        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    handle_cmd(line)
        except Exception as e:
            print(f"[PRESET] load error: {e}")

    # Startup: if frames set and ms not set, try to initialize ms after few frames
    startup_pending_frames_to_ms = (delay_ms == 0 and delay_frames > 0)

    # read buffer
    buf = bytearray(frame_bytes)
    mv = memoryview(buf)
    t_prev = time.time()

    try:
        while True:
            # process live commands
            try:
                while True:
                    line = cmd_queue.get_nowait()
                    handle_cmd(line)
            except queue.Empty:
                pass

            # read one frame from FFmpeg
            got = 0
            while got < frame_bytes:
                n = proc.stdout.readinto(mv[got:frame_bytes])
                if not n:
                    raise RuntimeError("FFmpeg pipe closed")
                got += n

            t_now = time.time()
            _tick(t_now - t_prev)
            t_prev = t_now

            # On startup, convert frames→ms once we have a reasonable fps_ema
            if startup_pending_frames_to_ms and (fps_cap_val > 0 or fps_ema > 0):
                ref_fps = fps_cap_val if fps_cap_val>0 else fps_ema
                delay_ms = int(round(delay_frames * 1000.0 / ref_fps))
                print(f"[Init] delay_frames={delay_frames} → delay_ms={delay_ms} (@refFPS={ref_fps:.2f})")
                startup_pending_frames_to_ms = False

            # ---- 입력 포맷별 CPU→GPU 업로드 ----
            if fmt.lower() == "uyvy422":
                arr = np.frombuffer(mv, dtype=np.uint8, count=frame_bytes).reshape(H, W//2, 4)
                t_u8 = torch.from_numpy(arr)
                try: t_u8 = t_u8.pin_memory()
                except Exception: pass
                uyvy = t_u8.to(device, non_blocking=True)
                rgb_u8 = uyvy422_to_rgb_u8_cuda(uyvy, H, W, device)  # [H,W,3] uint8 (GPU)
            else:
                ch = 4 if fmt.lower() in ("bgra","rgba") else 3
                arr = np.frombuffer(mv, dtype=np.uint8, count=frame_bytes).reshape(H, W, ch)
                t_u8 = torch.from_numpy(arr)
                try: t_u8 = t_u8.pin_memory()
                except Exception: pass
                if fmt.lower() == "bgra":
                    rgb_u8 = t_u8[..., [2,1,0]].to(device, non_blocking=True)
                elif fmt.lower() == "rgba":
                    rgb_u8 = t_u8[..., [0,1,2]].to(device, non_blocking=True)
                elif fmt.lower() == "bgr24":
                    rgb_u8 = t_u8[..., [2,1,0]].to(device, non_blocking=True)
                else:  # rgb24
                    rgb_u8 = t_u8.to(device, non_blocking=True)

            # torch input [1,3,H,W], fp16
            x = rgb_u8.permute(2,0,1).contiguous().to(torch.float16)
            x = (x*(1.0/255.0)).unsqueeze(0).contiguous(memory_format=torch.channels_last)

            # RVM inference
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                fgr, pha, *rec = rvm(x, *rec, downsample_ratio=float(long_side)/float(max(W, H)))
            alpha = pha[0,0].float().clamp(0,1)

            # Temporal EMA (on GPU)
            if alpha_prev is None:
                alpha_prev = alpha
            else:
                alpha_prev = alpha_ema*alpha_prev + (1.0 - alpha_ema)*alpha
            alpha_smooth = alpha_prev

            # Unsharp mask (GPU)
            if alpha_unsharp > 0.0:
                blurred = gaussian_blur_alpha(alpha_smooth, float(alpha_sigma), g_cache)
                alpha_sharp = torch.clamp(alpha_smooth + alpha_unsharp*(alpha_smooth - blurred), 0.0, 1.0)
            else:
                alpha_sharp = alpha_smooth

            # ----- Matte edge post-process (GPU) -----
            alpha_pp = alpha_sharp

            # (1) 침식/팽창
            if edge_px != 0:
                alpha_pp = morph_alpha(alpha_pp, edge_px)
            # (2) 페더
            if feather_sigma > 0.0:
                alpha_pp = gaussian_blur_alpha(alpha_pp, float(feather_sigma), g_cache)
            # (3) 엣지 하드니스(대비)
            if abs(edge_contrast - 1.0) > 1e-3:
                c = float(edge_contrast)
                alpha_pp = torch.sigmoid((alpha_pp - 0.5) * 8.0 * c)
            # (4) 감마
            if abs(matte_gamma - 1.0) > 1e-3:
                alpha_pp = torch.clamp(alpha_pp, 0, 1) ** float(matte_gamma)
            # (5) 클리핑 & 리맵
            if (clip_lo > 0.0) or (clip_hi < 1.0):
                lo, hi = clip_lo, clip_hi
                alpha_pp = torch.clamp(alpha_pp, lo, hi)
                alpha_pp = (alpha_pp - lo) / max(1e-6, (hi - lo))

            # ---- Compose RGBA frames (GPU) ----
            a_u8 = (alpha_pp*255.0 + 0.5).to(torch.uint8)
            zeros3 = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
            rgba_alpha = torch.cat([zeros3, a_u8.unsqueeze(-1)], dim=2).contiguous()

            onesA = torch.full((H, W, 1), 255, dtype=torch.uint8, device=device)
            rgba_video = torch.cat([rgb_u8, onesA], dim=2).contiguous()

            # Append to ring with timestamp
            ring.append((t_now, rgba_alpha, rgba_video))

            # Select output frames (time-based delay OR one-shot catch-up)
            if catch_up_once:
                out_a = ring[-1][1]
                out_v = ring[-1][2]
                catch_up_once = False
            else:
                if delay_ms > 0:
                    sel = select_time_delayed(time.time(), delay_ms)
                    if sel is None:
                        out_a, out_v = ring[-1][1], ring[-1][2]
                    else:
                        out_a, out_v = sel
                else:
                    out_a, out_v = ring[-1][1], ring[-1][2]

            # Spout upload & send (ALPHA)
            inter_a.upload_from_torch_u8(out_a)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_a)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_a)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            SpoutSender.sendTexture(s_alpha, int(tex_a), int(GL.GL_TEXTURE_2D), W, H, False, int(GL_RGBA))
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

            # Spout upload & send (VIDEO)
            inter_v.upload_from_torch_u8(out_v)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_v)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_v)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            SpoutSender.sendTexture(s_video, int(tex_v), int(GL.GL_TEXTURE_2D), W, H, False, int(GL_RGBA))
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

            # optional fps cap
            if frame_interval > 0.0:
                t_after = time.time()
                remain = frame_interval - (t_after - t_now)
                if remain > 0:
                    time.sleep(remain)

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt")
    finally:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            s_alpha.release()
        except Exception:
            pass
        try:
            s_video.release()
        except Exception:
            pass
        try:
            GL.glDeleteTextures([tex_a])
        except Exception:
            pass
        try:
            GL.glDeleteTextures([tex_v])
        except Exception:
            pass
        try:
            GL.glDeleteBuffers(1, [pbo_a])
        except Exception:
            pass
        try:
            GL.glDeleteBuffers(1, [pbo_v])
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    # 입력
    p.add_argument("--ffmpeg-bin", type=str, required=True)
    p.add_argument("--decklink-device", type=str, required=False, default=None)
    p.add_argument("--force-size", type=str, default="1920x1080")
    p.add_argument("--raw-format", type=str, default="bgra",
                   choices=["bgra","rgba","bgr24","rgb24","uyvy422"])
    p.add_argument("--decklink-format-code", type=str, default=None)
    p.add_argument("--decklink-connection", type=str, default=None)
    p.add_argument("--force-output-fps", type=str, default=None)
    p.add_argument("--ffmpeg-threads", type=int, default=6)

    # 새 입력 타입 옵션
    p.add_argument("--input-type", type=str, default="decklink",
                   choices=["decklink", "directshow", "file"],
                   help="decklink / directshow(OBS Virtual Camera 등) / file")
    p.add_argument("--directshow-device", type=str, default=None,
                   help="DirectShow 장치 이름 (예: 'OBS Virtual Camera')")
    p.add_argument("--input-file", type=str, default=None,
                   help="input-type=file 일 때 사용할 미디어 파일 경로")

    # RVM
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--cuda-index", type=int, default=0)
    p.add_argument("--long", type=int, default=640)
    # Spout
    p.add_argument("--spout-name-alpha", type=str, default="RT_RVM_ALPHA")
    p.add_argument("--spout-name-video", type=str, default="RT_RVM_VIDEO")
    # TX delay
    p.add_argument("--tx-delay-frames", type=int, default=0, help="initial FRAME input; will auto-convert to ms")
    p.add_argument("--tx-delay-max", type=int, default=120, help="ring buffer capacity (frames)")
    p.add_argument("--tx-delay-ms", type=int, default=0, help="initial TIME-based delay in ms; if >0, used directly")
    # 기타
    p.add_argument("--fps-cap", type=str, default="0")
    p.add_argument("--stats", type=int, default=1)
    p.add_argument("--spout-framesync", type=int, default=1)
    p.add_argument("--enable-cmd", type=int, default=1, help="enable stdin command thread")
    # Alpha tuning
    p.add_argument("--alpha-ema", type=float, default=0.60)
    p.add_argument("--alpha-unsharp", type=float, default=0.25)
    p.add_argument("--alpha-sigma", type=float, default=0.75)
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
        spout_name_video=args.spout_name_video,
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
        input_type=args.input_type,
        directshow_device=args.directshow_device,
        input_file=args.input_file,
    )
