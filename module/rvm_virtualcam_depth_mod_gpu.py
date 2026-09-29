#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rvm_virtualcam_depth_10_mod.py  (CPU 병목 모니터링 프린트 포함판)

추가 기능
- Grab / RVM / Depth / OBS 송출 단계별 시간(ms) 측정
- N프레임마다 집계 출력 (기본 60프레임)
- 선택: psutil 설치 시 1초 주기 CPU 사용률(전체/프로세스) 출력

사용 팁
- 상세 로그: --perf-every 1
- 프레임당 소요시간이 매우 작아 GPU Util이 낮게 보이는 상황에서
  어떤 단계가 상대적으로 시간을 잡아먹는지 빠르게 파악 가능합니다.
"""

import os, sys, types, threading, queue, time, contextlib
import torch

# ── torchvision 구버전 유틸 호환 shim ──
from torch.hub import load_state_dict_from_url as _load_state_dict_from_url
_tv_utils = types.ModuleType("torchvision.models.utils")
_tv_utils.load_state_dict_from_url = _load_state_dict_from_url
sys.modules["torchvision.models.utils"] = _tv_utils

ROOT = os.path.abspath(os.path.dirname(__file__))

# ── 분리된 모듈들 ──
from module.models.rvm import load_rvm
from module.models.midas import MidasDepth, DepthWorker
from module.vedioio.inputs import create_input_source
from module.engine.loop import run_loop

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


# ---------- 간단한 타이머/집계 유틸 ----------
class PerfAgg:
    """구간별 누적 측정치를 N프레임마다 출력"""
    def __init__(self, every=60, tag="PERF"):
        self.every = max(1, int(every))
        self.tag = tag
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_grab = 0.0
        self.sum_infer = 0.0
        self.sum_depth = 0.0
        self.sum_send = 0.0
        self.sum_total = 0.0

    @contextlib.contextmanager
    def section(self, name):
        t0 = time.time()
        try:
            yield
        finally:
            dt = (time.time() - t0) * 1000.0
            if name == "grab":
                self.sum_grab += dt
            elif name == "infer":
                self.sum_infer += dt
            elif name == "depth":
                self.sum_depth += dt
            elif name == "send":
                self.sum_send += dt
            elif name == "total":
                self.sum_total += dt

    def end_frame(self):
        self.n += 1
        if self.n >= self.every:
            avg = lambda s: (s / self.n) if self.n else 0.0
            print(f"[{self.tag}] n={self.n} | "
                  f"grab={avg(self.sum_grab):.2f}ms | "
                  f"infer={avg(self.sum_infer):.2f}ms | "
                  f"depth={avg(self.sum_depth):.2f}ms | "
                  f"send={avg(self.sum_send):.2f}ms | "
                  f"total={avg(self.sum_total):.2f}ms")
            self.reset()


# ---------- 선택: CPU 사용률 로거(1초 주기) ----------
class CPULogger(threading.Thread):
    def __init__(self, enable=True):
        super().__init__(daemon=True)
        self.enable = enable
        self.alive = True
        self._psutil = None
        if self.enable:
            try:
                import psutil  # type: ignore
                self._psutil = psutil
                self.proc = psutil.Process()
                # 첫 호출 워밍업
                psutil.cpu_percent(interval=None)
                self.proc.cpu_percent(interval=None)
            except Exception:
                self.enable = False

    def run(self):
        if not self.enable:
            return
        while self.alive:
            try:
                total = self._psutil.cpu_percent(interval=1.0)
                proc = self.proc.cpu_percent(interval=None)
                rss = self.proc.memory_info().rss / (1024**2)
                print(f"[CPU] total={total:.1f}% | proc={proc:.1f}% | RSS={rss:.0f}MB")
            except Exception:
                time.sleep(1.0)

    def stop(self):
        self.alive = False


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
    def __init__(self, virtualcam, use_sleep: bool, perf: PerfAgg | None = None):
        super().__init__(daemon=True)
        self.cam = virtualcam
        self.use_sleep = use_sleep
        self.q = queue.Queue(maxsize=1)
        self.alive = True
        self.perf = perf

    def run(self):
        while self.alive:
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                t0 = time.time()
                self.cam.send(frame)
                if self.use_sleep:
                    self.cam.sleep_until_next_frame()
                if self.perf:
                    self.perf.sum_send += (time.time() - t0) * 1000.0
            except Exception:
                pass

    def submit(self, frame_np):
        if not self.alive:
            return
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.q.put_nowait(frame_np)
        except queue.Full:
            pass

    def stop(self):
        self.alive = False


# ---------- Probe 래퍼들 ----------
class GrabProbe:
    """grab.read() 시간 측정용 래퍼"""
    def __init__(self, grab, perf: PerfAgg):
        self._grab = grab
        self.perf = perf

    def read(self, *a, **kw):
        with self.perf.section("grab"):
            return self._grab.read(*a, **kw)

    def release(self):
        return self._grab.release()


class RVMProbe(torch.nn.Module):
    """RVM __call__ 시간 측정용 래퍼(GPU sync 포함)"""
    def __init__(self, rvm, perf: PerfAgg, use_cuda=True):
        super().__init__()
        self.rvm = rvm
        self.perf = perf
        self.use_cuda = use_cuda

    def forward(self, *a, **kw):
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.rvm(*a, **kw)
        if self.use_cuda:
            torch.cuda.synchronize()
        self.perf.sum_infer += (time.time() - t0) * 1000.0
        return out

    # 모듈 인터페이스 위임
    def __getattr__(self, name):
        if name in ("rvm", "perf", "use_cuda"):
            return super().__getattribute__(name)
        return getattr(self.rvm, name)


class DepthModelProbe:
    """MiDaS depth.infer 시간 측정용 래퍼"""
    def __init__(self, depth_model: MidasDepth, perf: PerfAgg, use_cuda=True):
        self.depth_model = depth_model
        self.perf = perf
        self.use_cuda = use_cuda

    def infer(self, *a, **kw):
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        d = self.depth_model.infer(*a, **kw)
        if self.use_cuda:
            torch.cuda.synchronize()
        self.perf.sum_depth += (time.time() - t0) * 1000.0
        return d

    def __getattr__(self, name):
        if name in ("depth_model", "perf", "use_cuda"):
            return super().__getattribute__(name)
        return getattr(self.depth_model, name)


# ---------- Main orchestrator ----------
def run_virtualcam(**kwargs):
    """
    NOTE: 인자는 모두 **kwargs로 받습니다.
    - CLI에서 넘어온 값은 그대로 kwargs에 들어오고,
    - 여기서 필요한 것만 꺼내 기본값과 함께 사용합니다.
    """

    # ===== PERF 옵션 =====
    perf_every = int(kwargs.get("perf_every", 60))  # N프레임마다 요약
    perf = PerfAgg(every=perf_every, tag="PERF")
    cpu_logger = CPULogger(enable=bool(kwargs.get("cpu_log", True)))
    cpu_logger.start()

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
        decklink_device=decklink_device,
        decklink_format=decklink_format,
        force_size=force_size,
        force_fps=force_fps,
        rtbufsize=rtbufsize,
        raw_format=raw_format,
        ffmpeg_bin=ffmpeg_bin,
    )
    print(f"[INFO] Input size: {W}x{H}")

    # grab에 프로브 부착
    grab = GrabProbe(grab, perf=perf)

    # ===== 2) OBS 가상카메라 (선택) =====
    obs_out      = bool(kwargs.get("obs_out", False))
    obs_out_sync = bool(kwargs.get("obs_out_sync", False))
    target_fps   = kwargs.get("target_fps", "60000/1001")

    virtualcam = None
    sender = None
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
                print("[WARN] OBS VirtualCam 사용 불가. 계속 진행(프리뷰만).")
            else:
                sender = ObsSender(virtualcam, use_sleep=obs_out_sync, perf=perf)
                sender.start()
                print(f"[INFO] OBS Virtual Camera opened ({W}x{H}@{int(round(obs_fps))}fps)")
        except Exception as e:
            print("[WARN] OBS VirtualCam 초기화 실패:", e)
            virtualcam = None

    # ===== 3) RVM 로드 =====
    backbone = kwargs.get("backbone", "mobilenetv3")   # "mobilenetv3" | "resnet50"
    weights  = kwargs.get("weights", None)
    device   = kwargs.get("device", "cuda")
    fp16     = bool(kwargs.get("fp16", True))

    if weights is None:
        weights = os.path.join(
            ROOT, "rvm_mobilenetv3.pth" if backbone == "mobilenetv3" else "rvm_resnet50.pth"
        )
    if not os.path.isfile(weights):
        raise RuntimeError(f"가중치 파일이 없습니다: {weights}")
    rvm = load_rvm(weights, backbone, device)
    if fp16 and isinstance(device, str) and device.startswith("cuda"):
        rvm.half(); print("[INFO] FP16 enabled.")
    try:
        rvm = rvm.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # RVM 프로브(추론 시간 측정)
    rvm = RVMProbe(rvm, perf=perf, use_cuda=(isinstance(device, str) and device.startswith("cuda")))
    try:
        pdev = next(rvm.parameters()).device  # 내부 실제 모듈의 파라미터 확인
    except StopIteration:
        pdev = torch.device(device if isinstance(device, str) else "cpu")
    print(f"[CHECK] RVM parameter device = {pdev}")

    # ===== 4) Depth 준비 =====
    use_depth       = bool(kwargs.get("use_depth", True))
    depth_model     = None
    depth_worker    = None
    if use_depth:
        depth_model_name = kwargs.get("depth_model_name", "MiDaS_small")
        depth_long       = int(kwargs.get("depth_long", 384))
        depth_async      = bool(kwargs.get("depth_async", True))
        depth_fps        = float(kwargs.get("depth_fps", 12.0))

        dm = MidasDepth(
            device=("cuda" if isinstance(device, str) and device.startswith("cuda") else "cpu"),
            model_name=depth_model_name, max_long=depth_long
        )
        # Depth 모델도 프로브로 감싸기(동기/비동기 모두 infer 호출 지점 시간 측정)
        depth_model = DepthModelProbe(dm, perf=perf,
                                      use_cuda=(isinstance(device, str) and device.startswith("cuda")))
        if depth_async:
            depth_worker = DepthWorker(depth_model, max_fps=depth_fps)
            depth_worker.start()

    # ===== 5) 루프 실행 =====
    # run_loop 안에서 프레임마다 perf.section("total")을 못 감싸므로,
    # 최소한 run_loop 전체의 per-frame 종료 지점에서 perf.end_frame()를
    # 호출할 수 있도록 콜백을 전달한다. run_loop가 콜백 인자를 지원하지 않으면,
    # 아래 try/finally에서 초당 프레임 수치만 대략 출력되는 구조로 둔다.
    #
    # 현재 버전에서는 run_loop 인자에 perf 객체를 넘기며,
    # run_loop 내부에서 다음 패턴으로 사용하도록 구현했다고 가정:
    #   with perf.section("total"):
    #       ... grab.read ...
    #       ... rvm(...) ...
    #       ... depth_model.infer(...) ...
    #       ... sender.submit(...) ...
    #   perf.end_frame()
    #
    # 만약 run_loop가 이런 훅을 아직 지원하지 않는다면,
    # 그래도 Grab/RVM/Depth/Send는 래퍼가 시간을 집계하므로
    # 구간별 병목은 확인 가능합니다.

    try:
        run_loop(
            # 필수
            grab=grab, W=W, H=H, rvm=rvm, device=device, fp16=fp16,

            # 루프 옵션 (CLI에서 넘어온 값 그대로 연결)
            target_long=int(kwargs.get("long", kwargs.get("target_long", 768))),
            target_fps_str=str(kwargs.get("target_fps", "60000/1001")),
            force_fps_str=str(force_fps),
            ffmpeg_in=str(ffmpeg_in),

            # Depth
            use_depth=use_depth,
            depth_model=depth_model,
            depth_worker=depth_worker,
            depth_invert=bool(kwargs.get("depth_invert", True)),
            depth_bokeh=bool(kwargs.get("depth_bokeh", False)),
            blur_weak=float(kwargs.get("blur_weak", 1.2)),
            blur_strong=float(kwargs.get("blur_strong", 4.0)),

            # Alpha
            alpha_sigma=float(kwargs.get("alpha_sigma", 0.4)),
            alpha_unsharp=float(kwargs.get("alpha_unsharp", 0.35)),
            alpha_close=int(kwargs.get("alpha_close", 3)),
            alpha_dilate=int(kwargs.get("alpha_dilate", 3)),
            alpha_ema=float(kwargs.get("alpha_ema", 0.8)),
            ema_adaptive=bool(kwargs.get("ema_adaptive", True)),

            # 멀티 선택
            multi_mode=str(kwargs.get("multi_mode", "all")),

            # 에너지
            energy=bool(kwargs.get("energy", False)),
            energy_color=str(kwargs.get("energy_color", "0,255,255")),
            energy_width=int(kwargs.get("energy_width", 3)),
            energy_blur=float(kwargs.get("energy_blur", 12.0)),
            energy_gain=float(kwargs.get("energy_gain", 0.85)),
            energy_on_bg=bool(kwargs.get("energy_on_bg", True)),
            energy_pulse=bool(kwargs.get("energy_pulse", False)),
            energy_pulse_speed=float(kwargs.get("energy_pulse_speed", 1.0)),
            energy_pulse_min=float(kwargs.get("energy_pulse_min", 0.7)),
            energy_pulse_max=float(kwargs.get("energy_pulse_max", 1.3)),
            energy_pulse_shape=str(kwargs.get("energy_pulse_shape", "sine")),

            # 스파크
            spark=bool(kwargs.get("spark", False)),
            spark_count=int(kwargs.get("spark_count", 300)),
            spark_spawn=int(kwargs.get("spark_spawn", 80)),
            spark_color=str(kwargs.get("spark_color", "255,200,80")),
            spark_size=int(kwargs.get("spark_size", -5)),
            spark_speed=float(kwargs.get("spark_speed", 220.0)),
            spark_jitter=float(kwargs.get("spark_jitter", 0.6)),
            spark_gravity=float(kwargs.get("spark_gravity", 500.0)),
            spark_lifespan=float(kwargs.get("spark_lifespan", 0.80)),
            spark_trail=float(kwargs.get("spark_trail", 0.0)),
            spark_gain=float(kwargs.get("spark_gain", 1.0)),
            spark_on_bg=bool(kwargs.get("spark_on_bg", True)),
            spark_seed=int(kwargs.get("spark_seed", 1234)),

            # 프리뷰/출력
            sender=sender,
            no_preview=bool(kwargs.get("no_preview", False)),
            preview_scale=float(kwargs.get("preview_scale", 0.5)),
            show_depth=bool(kwargs.get("show_depth", False)),

            # 로그
            show_fps=bool(kwargs.get("show_fps", True)),
            log_detail=int(kwargs.get("log_detail", 1)),

            # 성능 집계 객체(옵션): run_loop가 지원한다면 내부에서 사용
            perf=perf,
        )
    finally:
        # 정리
        try:
            if sender: sender.stop()
            if depth_worker: depth_worker.stop()
            grab.release()
            if virtualcam is not None:
                virtualcam.close()
        except Exception:
            pass
        try:
            cpu_logger.stop()
        except Exception:
            pass


# ---------- CLI ----------
if __name__ == "__main__":
    from module.config.cli import build_parser, args_to_run_kwargs
    from module.config.presets import apply_preset  # 선택적

    parser = build_parser()
    # 추가 CLI 옵션
    parser.add_argument("--perf-every", type=int, default=60,
                        help="몇 프레임마다 성능 요약을 출력할지 (기본 60)")
    parser.add_argument("--cpu-log", type=int, default=1,
                        help="1이면 psutil로 1초 주기 CPU% 출력(미설치시 자동무시)")
    args = parser.parse_args()
    apply_preset(args)  # 필요하면 내부에서 기본값 조정

    kwargs = args_to_run_kwargs(args)
    # 위에서 추가한 CLI 옵션 반영
    kwargs["perf_every"] = getattr(args, "perf_every", 60)
    kwargs["cpu_log"] = bool(getattr(args, "cpu_log", 1))

    run_virtualcam(**kwargs)
