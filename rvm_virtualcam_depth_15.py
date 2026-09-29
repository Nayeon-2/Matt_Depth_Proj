#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rvm_virtualcam_depth_15.py  (energy color hotkey + smooth gradient)
- 입력: module.vedioio.inputs.create_input_source (DeckLink/ffmpeg 또는 DirectShow)
- 추론: module.models.rvm (RVM) + module.models.midas (비동기 옵션)
- 출력: OBS 가상카메라(옵션), DeckLink 출력(옵션), 프리뷰 창
- 즉시 종료 방지: 부울 안전 파싱(as_bool), 프리뷰/출력 최소 1개 강제 확보

추가: 런타임 에너지 색 변경(핫키)
  C        : 다음 색 프리셋
  1..9,0   : 해당 인덱스 색 직선택 (0 → 인덱스 0)
부드러운 전환 옵션
  --energy-color-lerp-ms N       # N ms 동안 선형 보간(기본 250). 0이면 비활성
  --energy-color-exp-tau-ms N    # 지수 감쇠형 전환 시간상수 ms. lerp_ms=0일 때만 적용
"""

import os, sys, types, threading, queue, time, math
import torch

# ── torchvision 구버전 유틸 호환 shim ──
from torch.hub import load_state_dict_from_url as _load_state_dict_from_url
_tv_utils = types.ModuleType("torchvision.models.utils")
_tv_utils.load_state_dict_from_url = _load_state_dict_from_url
sys.modules["torchvision.models.utils"] = _tv_utils

ROOT = os.path.abspath(os.path.dirname(__file__))

# ── 분리된 모듈들 ──
from module.models.rvm import load_rvm
from module.models.midas import MidasDepth, MidasDepthConfig, DepthProcess
from module.vedioio.inputs import create_input_source
from module.engine.loop import run_loop
from module.engine.color_change import (
    EnergyColorState, EnergyHotkey, PencilSweepController,
)

try:
    from module.vedioio.decklink_out import DeckLinkOut
except Exception:
    from decklink_out import DeckLinkOut  # 필요 시 수정

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --------- 백엔드 튜닝 ---------
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def as_bool(x):
    """안전 부울 파서: '0'→False, '1'→True, yes/on/true도 True"""
    if isinstance(x, bool): return x
    if isinstance(x, int):  return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1","true","yes","on","y","t")
    return bool(x)


# ---------- pyvirtualcam opener (버전 호환) ----------
def open_virtualcam(W, H, fps_float):
    import pyvirtualcam
    try:
        return pyvirtualcam.Camera(width=W, height=H, fps=int(round(fps_float)),
                                   fmt=pyvirtualcam.PixelFormat.BGR)
    except Exception:
        try:
            return pyvirtualcam.Camera(width=W, height=H, fps=int(round(fps_float)),
                                       fourcc=pyvirtualcam.FOURCC.BGR3, device=None)
        except Exception:
            try:
                return pyvirtualcam.Camera(width=W, height=H, fps=int(round(fps_float)),
                                           fourcc=pyvirtualcam.FOURCC.YUY2, device=None)
            except Exception as e3:
                print("[WARN] OBS VirtualCam 초기화 실패:", e3)
                return None


# ---------- OBS Sender Thread ----------
class ObsSender(threading.Thread):
    def __init__(self, virtualcam, use_sleep: bool):
        super().__init__(daemon=True)
        self.cam = virtualcam
        self.use_sleep = use_sleep
        self.q = queue.Queue(maxsize=1)
        self.alive = True
    def run(self):
        while self.alive:
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.cam.send(frame)
                if self.use_sleep:
                    self.cam.sleep_until_next_frame()
            except Exception:
                pass
    def submit(self, frame_np):
        if not self.alive:
            return
        if self.q.full():
            try: self.q.get_nowait()
            except queue.Empty: pass
        try: self.q.put_nowait(frame_np)
        except queue.Full: pass
    def stop(self):
        self.alive = False


# ---------- DeckLink Sender 어댑터 ----------
class DeckLinkOutSender:
    def __init__(self, decklink_out: DeckLinkOut):
        self.dl = decklink_out
        self.alive = True
    def submit(self, frame_np):
        if not self.alive or self.dl is None:
            return
        self.dl.send(frame_np)
    def stop(self):
        self.alive = False
        try:
            if self.dl: self.dl.close()
        except Exception:
            pass


class MultiSender:
    def __init__(self, senders):
        self.senders = [s for s in (senders or []) if s is not None]
    def submit(self, frame_np):
        for s in self.senders:
            try: s.submit(frame_np)
            except Exception: pass
    def stop(self):
        for s in self.senders:
            try: s.stop()
            except Exception: pass

# ---------- Main orchestrator ----------
def run_virtualcam(**kwargs):
    print("[BOOT] run_virtualcam start")

     # ===== 0) 런타임 상태/핫키 준비 =====
    energy_enabled = as_bool(kwargs.get("energy", False))
    energy_color_str = str(kwargs.get("energy_color", "0,255,255"))
    energy_color_presets = str(kwargs.get("energy_color_presets",
                                          "255,255,255;255,0,0;0,255,0;0,128,255;255,255,0"))
    energy_color_lerp_ms = int(kwargs.get("energy_color_lerp_ms", 1000))
    energy_color_exp_tau_ms = int(kwargs.get("energy_color_exp_tau_ms", 0))

    # Pencil sweep 세팅
    pencil_enabled    = as_bool(kwargs.get("pencil_enabled", 0))
    pencil_sweep_ms = int(kwargs.get("pencil_sweep_ms", 900))

    # V 스윕용 컨트롤러
    pencil_sweep_ctrl = PencilSweepController(
        duration_ms=pencil_sweep_ms,
        initial_visible=pencil_enabled,
    )

    # 에너지 색 상태
    state = EnergyColorState(
        start_color=energy_color_str,
        presets=energy_color_presets,
        lerp_ms=energy_color_lerp_ms,
        exp_tau_ms=energy_color_exp_tau_ms,
    )

    # EnergyHotkey: V 키 → 컨트롤러에 위임
    hk = EnergyHotkey(
        state=state,
        on_change=lambda col: None,
        on_cycle=None,
        on_sweep=pencil_sweep_ctrl.trigger_sweep,
        on_toggle_pencil=pencil_sweep_ctrl.toggle_visible,
    )
    hk.start()

    # ===== 1) 입력 소스 열기 =====
    ffmpeg_in        = kwargs.get("ffmpeg_in", "dshow")                # "dshow" | "decklink"
    dshow_name       = kwargs.get("dshow_name", "OBS Virtual Camera")
    decklink_device  = kwargs.get("decklink_device", None)
    decklink_format  = kwargs.get("decklink_format", None)
    force_size       = kwargs.get("force_size", "1920x1080")
    force_fps        = kwargs.get("force_fps", "60000/1001")
    rtbufsize        = kwargs.get("rtbufsize", "1024M")
    raw_format       = kwargs.get("raw_format", "bgr24")
    ffmpeg_bin       = kwargs.get("ffmpeg_bin", "ffmpeg")

    grab, W, H, obs_fps_from_input = create_input_source(
        ffmpeg_in=ffmpeg_in,
        dshow_name=dshow_name,
        force_size=force_size,
        force_fps=force_fps,
        rtbufsize=rtbufsize,
        raw_format=raw_format,
        ffmpeg_bin=ffmpeg_bin,
        decklink_device=decklink_device,
        decklink_format=decklink_format,
    )
    print(f"[INFO] Input ready: ffmpeg_in='{ffmpeg_in}' decklink_device='{decklink_device}' decklink_format='{decklink_format}' size={W}x{H}")

    # ===== 2) OBS 가상카메라 (선택) =====
    obs_out      = as_bool(kwargs.get("obs_out", False))
    obs_out_sync = as_bool(kwargs.get("obs_out_sync", False))
    target_fps   = kwargs.get("target_fps", "60000/1001")

    obs_sender = None
    virtualcam = None
    if obs_out:
        try:
            def _parse_fps(s):
                s = str(s)
                if "/" in s:
                    a, b = s.split("/")
                    return float(a) / float(b)
                return float(s)
            obs_fps = obs_fps_from_input if ffmpeg_in == "decklink" else _parse_fps(target_fps)
            virtualcam = open_virtualcam(W, H, obs_fps)
            if virtualcam is None:
                print("[WARN] OBS VirtualCam 사용 불가. 계속 진행.")
            else:
                obs_sender = ObsSender(virtualcam, use_sleep=obs_out_sync)
                obs_sender.start()
                print(f"[INFO] OBS VirtualCam opened {W}x{H}@{int(round(obs_fps))}fps")
        except Exception as e:
            print("[WARN] OBS VirtualCam 초기화 실패:", e)
            virtualcam = None

    # ===== 3) DeckLink OUT (선택) =====
    dl_out        = as_bool(kwargs.get("dl_out", False))
    dl_out_device = kwargs.get("dl_out_device", None)     # "DeckLink Quad (2)"
    dl_out_code   = kwargs.get("dl_out_format", None)     # "720p5994" 등
    dl_out_fps    = kwargs.get("dl_out_fps", force_fps)   # 기본은 입력 fps 문자
    dl_out_pix    = (kwargs.get("dl_out_pixfmt", "bgr24") or "bgr24").lower()
    dl_out_outpix = (kwargs.get("dl_out_outpix", "uyvy422") or "uyvy422").lower()
    dl_out_scale  = kwargs.get("dl_out_scale", None)

    decklink_sender = None
    if dl_out and dl_out_device and dl_out_code:
        try:
            dl = DeckLinkOut(
                out_device=dl_out_device,
                format_code=dl_out_code,
                size=(W, H),
                fps=dl_out_fps,
                in_pix_fmt=dl_out_pix,
                ffmpeg_bin=ffmpeg_bin,
                out_pix_fmt=dl_out_outpix,
                scale_to=dl_out_scale,
                decklink_opts=["-queue_size", "2"],
            )
            decklink_sender = DeckLinkOutSender(dl)
            print(f"[INFO] DeckLink OUT ready: device='{dl_out_device}', code='{dl_out_code}', in_pix='{dl_out_pix}', out_pix='{dl_out_outpix}'")
        except Exception as e:
            print("[WARN] DeckLink OUT 초기화 실패:", e)
            decklink_sender = None

    # ===== 4) Sender 통합 =====
    combined_sender = (MultiSender([obs_sender, decklink_sender])
                       if (obs_sender and decklink_sender)
                       else (obs_sender or decklink_sender))

    # ===== 5) RVM 로드 =====
    backbone = kwargs.get("backbone", "mobilenetv3")
    weights  = kwargs.get("weights", None)
    device   = kwargs.get("device", "cuda")
    fp16     = as_bool(kwargs.get("fp16", True))

    if weights is None:
        weights = os.path.join(ROOT, "rvm_mobilenetv3.pth" if backbone == "mobilenetv3" else "rvm_resnet50.pth")
    if not os.path.isfile(weights):
        raise RuntimeError(f"가중치 파일이 없습니다: {weights}")
    rvm = load_rvm(weights, backbone, device)
    if fp16 and isinstance(device, str) and device.startswith("cuda"):
        rvm.half(); print("[INFO] FP16 enabled.")
    try:
        rvm = rvm.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # ===== 6) Depth 준비 =====
    use_depth    = as_bool(kwargs.get("use_depth", True))
    depth_model  = None
    depth_worker = None
    if use_depth:
        depth_model_name = kwargs.get("depth_model_name", "MiDaS_small")
        depth_long       = int(kwargs.get("depth_long", 384))
        depth_async      = as_bool(kwargs.get("depth_async", True))
        depth_fps        = float(kwargs.get("depth_fps", 12.0))
        device = kwargs.get("device", "cuda")
        dev_str = "cuda" if (isinstance(device, str) and device.startswith("cuda")) else "cpu"

        if depth_async:
            cfg = MidasDepthConfig(
                device=dev_str,
                model_name=depth_model_name,
                work_size=(384, 384),
                out_size=(H, W),
                max_fps=depth_fps,
            )
            class _DepthAdapter:
                def __init__(self, proc):
                    self.proc = proc
                def submit(self, bgr_np):
                    self.proc.submit(bgr_np)
                @property
                def latest(self):
                    arr, ts = self.proc.get_latest(copy=False)
                    return arr
                def stop(self, timeout: float = 1.0):
                    self.proc.stop(timeout=timeout)
            depth_worker = _DepthAdapter(DepthProcess(cfg).start())
            depth_model = None
        else:
            depth_worker = None
            depth_model = MidasDepth(
                device=("cuda" if isinstance(device, str) and device.startswith("cuda") else "cpu"),
                model_name=depth_model_name, max_long=depth_long
            )
    # ===== YOLOv8 Seg 준비 =====
    yolo_model = None
    if bool(kwargs.get("yolo", False)):
        if YOLO is None:
            print("[WARN] ultralytics 미설치 → YOLO 비활성.")
        else:
            try:
                yolo_model = YOLO(kwargs.get("yolo_weights", "yolov8n-seg.pt"))
                print(f"[INFO] YOLOv8-seg loaded: {kwargs.get('yolo_weights')}")
            except Exception as e:
                print("[WARN] YOLO 로드 실패:", e)
                yolo_model = None

    # ===== 7) 프리뷰/옵션 =====
    no_preview    = as_bool(kwargs.get("no_preview", False))
    preview_scale = float(kwargs.get("preview_scale", 0.5))
    show_depth    = as_bool(kwargs.get("show_depth", False))

    if (combined_sender is None) and no_preview:
        print("[INFO] No sender & no_preview=True → forcing preview ON for visibility")
        no_preview = False

    # ===== 8) 루프 실행 =====
    print("[BOOT] run_loop starting …")
    try:
        run_loop(
            # 필수
            grab=grab, W=W, H=H, rvm=rvm, device=device, fp16=fp16,

            # 루프 옵션
            target_long=int(kwargs.get("long", kwargs.get("target_long", 768))),
            target_fps_str=str(kwargs.get("target_fps", "60000/1001")),
            force_fps_str=str(force_fps),
            ffmpeg_in=str(ffmpeg_in),

            # Depth
            use_depth=use_depth,
            depth_model=depth_model,
            depth_worker=depth_worker,
            depth_invert=as_bool(kwargs.get("depth_invert", True)),
            depth_bokeh=as_bool(kwargs.get("depth_bokeh", False)),
            blur_weak=float(kwargs.get("blur_weak", 1.2)),
            blur_strong=float(kwargs.get("blur_strong", 4.0)),

            # Alpha
            alpha_sigma=float(kwargs.get("alpha_sigma", 0.4)),
            alpha_unsharp=float(kwargs.get("alpha_unsharp", 0.35)),
            alpha_close=int(kwargs.get("alpha_close", 3)),
            alpha_dilate=int(kwargs.get("alpha_dilate", 3)),
            alpha_ema=float(kwargs.get("alpha_ema", 0.8)),
            ema_adaptive=as_bool(kwargs.get("ema_adaptive", True)),

            # 멀티 선택
            multi_mode=str(kwargs.get("multi_mode", "all")),

            # 에너지 (초기값 그대로 전달)
            energy=energy_enabled,
            energy_color=str(energy_color_str),
            energy_width=int(kwargs.get("energy_width", 5)),
            energy_blur=float(kwargs.get("energy_blur", 12.0)),
            energy_gain=float(kwargs.get("energy_gain", 0.85)),
            energy_on_bg=as_bool(kwargs.get("energy_on_bg", True)),
            energy_pulse=as_bool(kwargs.get("energy_pulse", False)),
            energy_pulse_speed=float(kwargs.get("energy_pulse_speed", 1.0)),
            energy_pulse_min=float(kwargs.get("energy_pulse_min", 0.7)),
            energy_pulse_max=float(kwargs.get("energy_pulse_max", 1.3)),
            energy_pulse_shape=str(kwargs.get("energy_pulse_shape", "sine")),

            energy_y_band_mode = str(kwargs.get("energy_y_band_mode", "off")),
            energy_y_band_center = float(kwargs.get("energy_y_band_center", 50.0)),
            energy_y_band_width = float(kwargs.get("energy_y_band_width", 10.0)),
            energy_y_band_sigma = float(kwargs.get("energy_y_band_sigma", 0.0)), 
            energy_debug=as_bool(kwargs.get("energy_debug", False)),


            # ripple
            ripple_enabled = bool(kwargs.get("ripple_enabled", 0)),
            ripple_count = int(kwargs.get("ripple_count", 3)),
            ripple_width = int(kwargs.get("ripple_width", 3)),
            ripple_spacing = int(kwargs.get("ripple_spacing", 18)),
            ripple_speed = float(kwargs.get("ripple_speed", 120.0)),
            ripple_start = int(kwargs.get("ripple_start", 0)),
            ripple_color = kwargs.get("ripple_color", "0,255,255"),
            ripple_blend = kwargs.get("ripple_blend", "add"),
            ripple_on_bg = bool(kwargs.get("ripple_on_bg", 1)),

            # pencil
            pencil_enabled = bool(kwargs.get("pencil_enabled", 0)),
            pencil_color   = str(kwargs.get("pencil_color", "255,128,0")),
            pencil_color2  = (None if kwargs.get("pencil_color2") in (None,"","None") else str(kwargs.get("pencil_color2"))),
            pencil_multicolor = int(kwargs.get("pencil_multicolor", 0)),
            pencil_width   = int(kwargs.get("pencil_width", 3)),
            pencil_soft    = int(kwargs.get("pencil_soft", 1)),
            pencil_gain    = float(kwargs.get("pencil_gain", 1.0)),
            pencil_opacity = float(kwargs.get("pencil_opacity", 0.8)),
            pencil_thresh  = float(kwargs.get("pencil_thresh", 0.08)),
            pencil_rough   = float(kwargs.get("pencil_rough", 0.7)),
            pencil_jitter  = float(kwargs.get("pencil_jitter", 0.8)),
            pencil_dash    = int(kwargs.get("pencil_dash", 0)),
            pencil_dash_len= int(kwargs.get("pencil_dash_len", 12)),
            pencil_gap_len = int(kwargs.get("pencil_gap_len", 8)),
            pencil_on_bg   = bool(kwargs.get("pencil_on_bg", 1)),
            pencil_on_fg   = bool(kwargs.get("pencil_on_fg", 1)),
            pencil_blend   = str(kwargs.get("pencil_blend", "alpha")),
            pencil_palette = str(kwargs.get("pencil_palette", "")),
            pencil_bands   = int(kwargs.get("pencil_bands", 7)),
            pencil_repeat  = int(kwargs.get("pencil_repeat", 1)),
            pencil_axis    = str(kwargs.get("pencil_axis", "x")),
            pencil_sweep_ctrl   = pencil_sweep_ctrl,
            pencil_sweep_soft_px = int(kwargs.get("pencil_sweep_soft_px", 24)),
            pencil_sweep_axis    = str(kwargs.get("pencil_sweep_axis", "y")),

            yolo=yolo_model,
            yolo_img=int(kwargs.get("yolo_img", 640)),
            yolo_conf=float(kwargs.get("yolo_conf", 0.5)),

            # 프리뷰/출력
            sender=combined_sender,
            no_preview=no_preview,
            preview_scale=preview_scale,
            show_depth=show_depth,

            # 로그
            show_fps=as_bool(kwargs.get("show_fps", True)),
            log_detail=int(kwargs.get("log_detail", 1)),

            # ★ 런타임 에너지 색 공급자(핵심 추가)
            energy_color_source=state.get_color_rgb_tuple,

            # --- energy split (좌/우 색 분할) ---
            energy_split   = str(kwargs.get("energy_split", "off")),   # "off" | "center" | "abs" | "pct"
            energy_split_x = float(kwargs.get("energy_split_x", 0.5)),  # abs: px, pct/center: 0.5 권장
            left_color     = str(kwargs.get("left_color",  "255,0,0")),
            right_color    = str(kwargs.get("right_color", "0,160,255")),
        )
    finally:
        print("[BOOT] cleanup …")
        try:
            if isinstance(combined_sender, MultiSender):
                combined_sender.stop()
            elif combined_sender is not None:
                combined_sender.stop()
            if hk: hk.stop()
            if depth_worker: depth_worker.stop()
            grab.release()
            if virtualcam is not None:
                virtualcam.close()
        except Exception:
            pass


# ---------- CLI ----------
if __name__ == "__main__":
    from module.config.cli import build_parser, args_to_run_kwargs
    from module.config.presets import apply_preset  # 선택적
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == "cuda":
        torch.cuda.set_device(device)

    parser = build_parser()
    args = parser.parse_args()
    apply_preset(args)  # 필요하면 내부에서 기본값 조정

    # 에너지 색 프리셋 CLI 파라미터(없으면 기본값)
    if not hasattr(args, "energy_color_presets"):
        setattr(args, "energy_color_presets", "255,255,255;255,0,0;0,255,0;0,128,255;255,255,0")
    if not hasattr(args, "energy_color_lerp_ms"):
        setattr(args, "energy_color_lerp_ms", 250)
    if not hasattr(args, "energy_color_exp_tau_ms"):
        setattr(args, "energy_color_exp_tau_ms", 0)

    kwargs = args_to_run_kwargs(args)

    # 위에서 기본값 주입한 2개도 kwargs에 반영됐는지 확인(구현체에 따라 다름)
    kwargs.setdefault("energy_color_presets", "255,255,255;255,0,0;0,255,0;0,128,255;255,255,0")
    kwargs.setdefault("energy_color_lerp_ms", 250)
    kwargs.setdefault("energy_color_exp_tau_ms", 0)

    run_virtualcam(**kwargs)
