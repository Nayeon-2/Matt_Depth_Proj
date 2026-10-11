# ====================== run_loop (Y-band only, no beam) ======================
from __future__ import annotations
import cv2, torch, numpy as np, time
import queue
import torch.nn.functional as F
import math

from .alpha import (
    blur_alpha_gpu, unsharp_mask_alpha,
    dilate_gpu, erode_gpu, close_gpu,
    ema_smooth_alpha,
)

from ..effects.selector import select_component
from ..effects.ripple_lines_1 import draw_moving_outlines
from ..effects.edge_pencil import draw_pencil_outline
from ..effects.energy import apply_energy
from ..effects.per_person_core import parse_palette, connected_instances_from_alpha
from .person_distance import encode_distance, encode_logo, chest_anchor

from ..models.deepsort.deepsort_tracker import DeepSortTracker

# 프리뷰 창 키 → 런타임 명령 (콘솔 핫키도 같은 명령을 console_cmds 큐로 보낸다)
PREVIEW_KEY_CMDS = {
    ord('q'): "quit",
    ord('l'): "logo",
    ord('b'): "bg_reset",
    ord('t'): "target_next",
    ord('y'): "target_reset",
}

# --- add: small helpers for energy ---
def parse_rgb(c, device=None, dtype=torch.float32):
    if isinstance(c, str):
        try:
            r, g, b = [float(x) for x in c.split(",")]
        except Exception:
            r, g, b = 0.0, 255.0, 255.0
        t = torch.tensor([r, g, b], dtype=dtype, device=device)
    elif isinstance(c, torch.Tensor):
        t = c.to(dtype=dtype, device=device).flatten()[:3]
    else:  # list/tuple
        r, g, b = c
        t = torch.tensor([float(r), float(g), float(b)], dtype=dtype, device=device)
    if t.max().item() > 1.0:
        t = t / 255.0
    return t.clamp_(0.0, 1.0)

def compute_energy_rgb(
    alpha_1chw: torch.Tensor,
    on_bg: bool,
    width: int,
    blur: float,
    color,
    gain: float,
    device,
    dtype
) -> torch.Tensor:
    edge = (alpha_1chw - erode_gpu(alpha_1chw, int(max(1, width)))).clamp(0, 1)
    mask = (1.0 - alpha_1chw) if on_bg else torch.ones_like(alpha_1chw)
    if float(blur) > 0:
        edge = blur_alpha_gpu(edge, float(blur))
    col = parse_rgb(color, device=device, dtype=dtype).view(3, 1, 1)
    energy = (edge * max(0.0, float(gain))) * mask
    return (col * energy.squeeze(1)).clamp(0.0, 1.0)

def _compute_ds_ratio(h, w, target_long: int) -> float:
    return float(min(target_long / max(h, w), 1.0))

def _parse_fps(s):
    s = str(s)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def gaussian_blur_gpu(x_3hw: torch.Tensor, ks: int, sigma: float) -> torch.Tensor:
    """Small separable Gaussian blur used by the optional depth-bokeh path."""
    ks = max(3, int(ks) | 1)
    sigma = max(float(sigma), 1e-3)
    radius = ks // 2
    coords = torch.arange(-radius, radius + 1, device=x_3hw.device, dtype=torch.float32)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = (kernel / kernel.sum()).to(dtype=x_3hw.dtype)
    channels = int(x_3hw.shape[0])
    x = x_3hw.unsqueeze(0)
    kh = kernel.view(1, 1, 1, ks).expand(channels, 1, 1, ks)
    kv = kernel.view(1, 1, ks, 1).expand(channels, 1, ks, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kh, groups=channels)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kv, groups=channels)
    return x.squeeze(0)

def compute_energy_mask(alpha_1chw: torch.Tensor, on_bg: bool, width: int, blur: float, gain: float) -> torch.Tensor:
    # [1,1,H,W] -> [1,1,H,W] 스칼라 강도
    e = (alpha_1chw - erode_gpu(alpha_1chw, int(max(1, width)))).clamp(0, 1)
    if float(blur) > 0:
        e = blur_alpha_gpu(e, float(blur))
    if on_bg:
        e = e * (1.0 - alpha_1chw)
    return e * max(0.0, float(gain))

def _pick_center_person_bbox_from_yolo(dets, W: int, H: int, person_class_id: int = 0):
    """
    dets: [(x1,y1,x2,y2,cls_id,conf), ...]
    리턴: (bbox, idx)
      - bbox: (x1,y1,x2,y2) 또는 None
      - idx : dets 리스트에서의 인덱스(없으면 -1)
    """
    if not dets:
        return None, -1

    cx_f, cy_f = W * 0.5, H * 0.5
    best = None
    best_d2 = 1e18
    best_i = -1

    for i, d in enumerate(dets):
        x1, y1, x2, y2, cls_id, conf = d[:6]
        if cls_id != person_class_id:
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        d2 = (cx - cx_f) ** 2 + (cy - cy_f) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (x1, y1, x2, y2)
            best_i = i

    return best, best_i


def _bbox_to_mask_1hw(bbox, H: int, W: int, device, dtype=torch.float32):
    """
    bbox: (x1,y1,x2,y2) → [1,H,W] float 마스크
    """
    m = torch.zeros((1, H, W), dtype=dtype, device=device)
    if bbox is None:
        return m
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(W - 1, x1)))
    x2 = int(max(0, min(W,     x2)))
    y1 = int(max(0, min(H - 1, y1)))
    y2 = int(max(0, min(H,     y2)))
    if x2 > x1 and y2 > y1:
        m[:, y1:y2, x1:x2] = 1.0
    return m


def run_loop(
    *,
    # 입력/출력
    grab,                   # .read(timeout)->(ok, frame)
    W: int, H: int,
    rvm,
    device: str = "cuda",
    fp16: bool = True,

    # RVM 해상도/프레임
    target_long: int = 768,
    target_fps_str: str = "60000/1001",
    force_fps_str: str = "60000/1001",
    ffmpeg_in: str = "decklink",
    input_timeout_s: float = 0.5,
    max_input_timeouts: int = 10,

    # Depth (비활성시 무시)
    use_depth: bool = False,
    depth_model=None,
    depth_worker=None,
    depth_invert: bool = True,
    depth_bokeh: bool = False,
    blur_weak: float = 1.2,
    blur_strong: float = 4.0,

    # Alpha
    alpha_sigma: float = 0.4,
    alpha_unsharp: float = 0.35,
    alpha_close: int = 3,
    alpha_dilate: int = 3,
    alpha_ema: float = 0.8,
    ema_adaptive: bool = True,

    # 멀티 선택
    multi_mode: str = "all",

    # ===== ENERGY (기본 + Y-band 옵션) =====
    energy: bool = False,
    energy_color: str = "0,255,255",
    energy_width: int = 6,
    energy_blur: float = 16.0,
    energy_gain: float = 2.0,
    energy_on_bg: bool = True,
    energy_pulse: bool = False,
    energy_pulse_speed: float = 1.0,
    energy_pulse_min: float = 0.7,
    energy_pulse_max: float = 1.3,
    energy_pulse_shape: str = "sine",
    energy_color_source=None,
    energy_debug: bool = False,

    # Tracking (DeepSORT)
    use_deepsort: bool = False,
    ds_iou: float = 0.3,
    ds_max_age: int = 30,
    ds_max_dist: float = 0.2,
    ds_model: str | None = None,
    ds_device: str | None = None,

    # Y-band 옵션
    energy_y_band_mode: str = "off",      # "off" | "abs" | "bbox"
    energy_y_band_center: float = 50.0,   # 0..100 (위=0, 아래=100)
    energy_y_band_width: float = 14.0,    # 박스 폭(%)
    energy_y_band_sigma: float = 4.0,     # >0 가우시안 소프트마스크

    # --- Energy per-person 전달 인자 ---
    energy_per_person: bool = False,
    energy_colors: str = "255,0,0|0,255,0|0,128,255|255,255,0",
    energy_max_instances: int = 4,
    energy_instance_thr: float = 0.5,
    energy_min_area: int = 2000,
    energy_sort_by: str = "area",
    energy_count: int = 3,
    energy_blend: str = "screen",

     # Aura (flow-like) modulation
    aura: bool = True,            # 아우라 활성
    aura_amp: float = 0.35,       # 진폭(0~) 0.25~0.45 권장
    aura_freq: float = 1.6,       # 공간 주파수(저주파 추천: 0.8~2.0)
    aura_speed: float = 0.6,      # 시간 속도(Hz 개념)
    aura_twirl: float = 0.8,      # 소용돌이 강도(각도 의존 위상)
    aura_rgb_phase: float = 0.0,  # 채널별 미세 위상차(0이면 비활성; 0.5~1.0 추천)
    aura_down : int= 1,
    aura_roi : float= 0.0,

    # Ripple
    ripple_enabled: bool = False,
    ripple_count: int = 3,
    ripple_width: int = 2,
    ripple_spacing: int = 28,
    ripple_speed: float = 60.0,
    ripple_start: int = 8,
    ripple_color: str = "255,180,80",
    ripple_blend: str = "screen",
    ripple_on_bg: bool = True,
    ripple_gain: float = 1.0,

    # Pencil (colored pencil outline)
    pencil_enabled: bool = False,
    pencil_color: str = "255,128,0",
    pencil_color2: str | None = None,
    pencil_multicolor: int = 0,
    pencil_width: int = 3,
    pencil_soft: int = 1,
    pencil_gain: float = 1.0,
    pencil_opacity: float = 0.8,
    pencil_thresh: float = 0.08,
    pencil_rough: float = 0.7,
    pencil_jitter: float = 0.8,
    pencil_dash: int = 0,
    pencil_dash_len: int = 12,
    pencil_gap_len: int = 8,
    pencil_on_bg: bool = True,
    pencil_on_fg: bool = True,
    pencil_blend: str = "alpha",
    pencil_palette: str = "",
    pencil_bands: int = 7,
    pencil_repeat: int = 1,
    pencil_axis: str = "x",
    pencil_hatch: int = 0,
    pencil_hatch_mode: str = "normal",  # "global" | "tangent" | "normal"
    pencil_hatch_angle: float = 30.0,   # global에서만 사용
    pencil_hatch_period: float = 6.0,   # px
    pencil_hatch_width: float = 2.0,    # px
    pencil_hatch_phase: float = 0.0,    # 0~1
    pencil_hatch_jitter: float = 0.30,
    pencil_sweep_ctrl=None,     
    pencil_sweep_soft_px: int = 24,          # 경계 소프트 폭(px)
    pencil_sweep_axis: str = "y",

    yolo=None, yolo_img: int = 640, yolo_conf: float = 0.5,
    energy_split: str = "off",      # "off" | "center" | "abs" | "pct"
    energy_split_x: float = 0.5,    # abs: px, pct: 0..1
    left_color: str = "255,0,0",
    right_color: str = "0,160,255",
    energy_split_soft: float = 0.0,

    # 프리뷰/출력
    sender=None,                # Obs/DeckLink Sender 또는 None
    matte_sender=None,          # 사람 누끼(BGRA: RVM fgr + alpha)를 받는 Sender 또는 None
    bg_sender=None,             # 배경(BGR)을 받는 Sender 또는 None
    bg_mode: str = "clean",     # "clean": 사람 없는 영역만 누적한 배경 / "raw": 입력 프레임 그대로
    bg_rate: float = 0.15,      # clean 모드에서 보이는 배경 픽셀의 갱신 속도(0~1)
    bg_margin: int = 15,        # clean 모드에서 사람 주변을 추가로 제외할 여유(px)
    pose_worker=None,           # PoseWorker (사람 거리 추정용) 또는 None
    distance_estimator=None,    # PersonDistanceEstimator 또는 None → 거리값을 matte 좌상단 픽셀에 인코딩
    seg_worker=None,            # YoloSegWorker (사람별 인스턴스 마스크) 또는 None
    target_selector=None,       # TargetPersonSelector (효과를 줄 한 사람 선택) 또는 None
    target_sender=None,         # 타겟 마스크(BGRA: B=타겟 영역, G=다른 사람 알파, R=타겟 알파)를 받는 Sender 또는 None
    console_cmds=None,          # queue.Queue: 콘솔 핫키가 넣는 명령("quit", "logo", "bg_reset", "target_next", "target_reset")
    logo_duration: float = 4.0, # L 키 로고 연출 길이(초): 뚫고 나옴 → 머묾 → 사라짐
    logo_every: float = 0.0,    # >0 이면 이 간격(초)마다 로고 연출을 자동 실행 (테스트/자동 반복용)
    no_preview: bool = False,
    preview_scale: float = 0.5,
    show_depth: bool = False,

    # 로그
    show_fps: bool = True,
    log_detail: int = 1,
):
    """메인 루프 (beam 제거 버전, Y-band 제한 적용)"""

    preview_title = "RVM Preview - Q to quit"
    if not no_preview:
        # Create a normal, resizable window up front so launchers and Windows
        # reliably surface it instead of waiting for the first implicit imshow.
        cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            preview_title,
            max(320, int(W * preview_scale)),
            max(180, int(H * preview_scale)),
        )

    # 상태 변수
    base_long_hi = int(target_long)
    prev_ds = None
    rec = [None, None, None, None]
    alpha_prev = None
    prev_keep = None
    last_depth = None

    # Reuse pinned buffers instead of allocating/copying pinned memory every frame.
    use_cuda = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()
    input_cpu = torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True) if use_cuda else None
    input_cpu_np = input_cpu.numpy() if input_cpu is not None else None
    cpu_buf = torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=True) if sender is not None else None
    matte_cpu_buf = (torch.empty((H, W, 4), dtype=torch.uint8, pin_memory=use_cuda)
                     if matte_sender is not None else None)
    bg_cpu_buf = (torch.empty((H, W, 3), dtype=torch.uint8, pin_memory=use_cuda)
                  if bg_sender is not None else None)
    target_cpu_buf = (torch.empty((H, W, 4), dtype=torch.uint8, pin_memory=use_cuda)
                      if target_sender is not None else None)
    bg_plate = None   # clean 모드 배경 누적 버퍼 [3,H,W] RGB
    bg_seen = None    # 한 번이라도 사람 없이 보인 픽셀 [1,H,W] bool
    bg_fill = None    # 아직 못 본 영역을 채울 인페인팅 결과 [3,H,W]
    bg_frames = 0
    bg_reset_all = False
    person_z, person_z_valid, last_pose_ts = 0.0, False, 0.0
    logo_t0 = None                 # L 키를 누른 시각 (로고 연출 시작)
    logo_anchor = (0.5, 0.55)      # 사람 가슴의 화면 위치 (0..1)
    logo_last_auto = time.perf_counter()
    pending_cmds: list[str] = []   # 이번 프레임에 처리할 런타임 명령

    # Tracker
    tracker = DeepSortTracker(
        max_age=int(ds_max_age),
        iou_thresh=float(ds_iou),
        max_dist=float(ds_max_dist),
        model=ds_model,
        device=ds_device
    ) if use_deepsort else None

    # track_id -> color index 고정 매핑(루프 전역)
    color_map: dict[int, int] = {}
    next_color = 0
    try:
        pal0 = parse_palette(str(energy_colors), device='cpu')
        palette_len = int(pal0.shape[0])
        if palette_len <= 0:
            palette_len = 4
    except Exception:
        palette_len = 4

    # 루프
    t0 = time.perf_counter(); frames = 0
    consecutive_timeouts = 0
    while True:
        loop_start = time.time()  # used by alpha EMA's wall-clock decay
        loop_perf_start = time.perf_counter()
        ok, frame = grab.read(timeout=max(0.001, float(input_timeout_s)))
        if not ok:
            consecutive_timeouts += 1
            if consecutive_timeouts == 1 or consecutive_timeouts % 5 == 0:
                print(f"[WARN] 입력 프레임 대기 시간 초과 ({consecutive_timeouts}/{max_input_timeouts})")
            if int(max_input_timeouts) > 0 and consecutive_timeouts >= int(max_input_timeouts):
                print("[ERROR] 입력이 장시간 중단되어 처리를 종료합니다.")
                break
            continue
        consecutive_timeouts = 0

        # 사람 거리: 포즈 워커에 최신 프레임 공급, 새 결과가 나오면 추정기 갱신
        if pose_worker is not None and distance_estimator is not None:
            pose_worker.submit(frame)
            pose_worker.need_once(1.0)
            st = pose_worker.state
            if st.get("ts", 0.0) > last_pose_ts:
                last_pose_ts = st["ts"]
                kps = st["kps"][0] if st.get("ok") and st.get("kps") is not None else None
                person_z, person_z_valid = distance_estimator.update(
                    kps, st.get("vis") if kps is not None else None, st.get("bbox"), now=last_pose_ts)
                a = chest_anchor(kps, st.get("vis") if kps is not None else None, W, H)
                if a is not None:   # light smoothing so the logo path does not jitter
                    logo_anchor = (logo_anchor[0] * 0.6 + a[0] * 0.4, logo_anchor[1] * 0.6 + a[1] * 0.4)

        # 타겟 선택: YOLO-seg 워커에 최신 프레임 공급 (결과는 알파 계산 뒤에 사용)
        if seg_worker is not None:
            seg_worker.submit(frame)

        # depth 비동기 갱신
        if use_depth and depth_worker is not None:
            depth_worker.submit(frame)
        else:
            last_depth = None

        # 입력 변환(BGR->RGB)
        prep_t0 = time.perf_counter()
        frame_c = np.ascontiguousarray(frame)
        if input_cpu is not None:
            np.copyto(input_cpu_np, frame_c, casting="no")
            u8 = input_cpu.to(device, non_blocking=True)
        else:
            u8 = torch.from_numpy(np.array(frame_c, copy=True, order="C")).to(device)
        u8 = u8.permute(2,0,1)  # BGR->CHW
        rgb01 = u8[[2,1,0], ...].to(dtype=(torch.float16 if fp16 else torch.float32)) * (1.0/255.0)
        t_in = rgb01.unsqueeze(0)
        try: t_in = t_in.contiguous(memory_format=torch.channels_last)
        except: pass
        prep_t1 = time.perf_counter()

        # depth 단일 추론(선택)
        if use_depth and depth_worker is not None and depth_worker.latest is not None:
            last_depth = depth_worker.latest
        elif use_depth and (depth_worker is None) and (depth_model is not None):
            d = depth_model.infer(frame)
            last_depth = (1.0 - d) if depth_invert else d

        # ===== RVM 추론 =====
        infer_t0 = time.perf_counter()
        cur_long = base_long_hi
        ds = _compute_ds_ratio(frame.shape[0], frame.shape[1], cur_long)
        if (prev_ds is None) or (abs(ds - prev_ds) > 1e-6):
            rec = [None, None, None, None]
            alpha_prev = None
            prev_ds = ds

        with torch.inference_mode():
            if fp16 and isinstance(device, str) and device.startswith("cuda"):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    fgr, pha, *rec = rvm(t_in, *rec, downsample_ratio=ds)
            else:
                fgr, pha, *rec = rvm(t_in, *rec, downsample_ratio=ds)
        infer_t1 = time.perf_counter()

        # ===== 알파 후처리 =====
        alpha = pha[0:1].clamp(0,1)
        if alpha_sigma > 0: alpha = blur_alpha_gpu(alpha, float(alpha_sigma))
        if alpha_unsharp > 0: alpha = unsharp_mask_alpha(alpha, amount=float(alpha_unsharp), radius=1.0)

        alpha_2d = alpha.squeeze(0).squeeze(0)
        keep = select_component(
            alpha_2d,
            last_depth if (use_depth and last_depth is not None and depth_invert) else
            (1.0 - last_depth if (use_depth and last_depth is not None) else None),
            mode=multi_mode, center_xy=(W//2, H//2)
        )
        alpha_2d = (alpha_2d * keep).clamp(0,1)
        alpha = alpha_2d.unsqueeze(0).unsqueeze(0)

        # 시야 급변 시 rec/EMA 리셋
        if prev_keep is None:
            prev_keep = keep
        else:
            inter = (keep * prev_keep).sum().item()
            union = torch.clamp(keep + prev_keep, 0, 1).sum().item()
            iou = (inter / union) if union > 0 else 0.0
            if iou < 0.5:
                alpha_prev = None
                rec = [None, None, None, None]
            prev_keep = keep

        # EMA
        alpha, alpha_prev, prev_motion_mag = ema_smooth_alpha(
            alpha, alpha_prev, base_ema=float(alpha_ema),
            adaptive=bool(ema_adaptive), loop_start_ts=loop_start
        )

        # 형태학
        if alpha_close > 0: alpha = close_gpu(alpha, int(alpha_close))
        if alpha_dilate > 0: alpha = dilate_gpu(alpha, int(alpha_dilate))

        # 합성 준비
        alpha2d = alpha.squeeze(0).clamp(0, 1)   # [1,H,W]
        fgr_t   = fgr[0].clamp(0, 1)             # [3,H,W] RGB
        base_bg = t_in[0].clamp(0, 1)            # [3,H,W] RGB
        alpha   = alpha.float(); alpha2d = alpha2d.float(); fgr_t = fgr_t.float(); base_bg = base_bg.float()

        # 타겟 가중치 [1,H,W]: 타겟 사람(+grow 여유) 쪽 1, 다른 사람 쪽 0. 타겟이 없으면 None
        target_w = None
        if seg_worker is not None and target_selector is not None:
            target_w = target_selector.update(seg_worker.state)

        # ===== 배경(보케 선택) =====
        bg_t = base_bg
        if use_depth and depth_bokeh and (last_depth is not None):
            # 프로젝트 내 gaussian_blur_gpu가 존재한다고 가정 (기존 코드와 동일)
            depth_t = torch.from_numpy(last_depth).to(device=base_bg.device, dtype=base_bg.dtype)
            if depth_t.ndim == 2: depth_t = depth_t.unsqueeze(0)  # [1,H,W]
            depth_t = depth_t.clamp(0, 1)
            if depth_invert: depth_t = 1.0 - depth_t

            weak_blur   = gaussian_blur_gpu(base_bg, ks=11, sigma=float(blur_weak))
            strong_blur = gaussian_blur_gpu(base_bg, ks=21, sigma=float(blur_strong))
            bg_mask = (1.0 - alpha2d).clamp(0, 1)
            bokeh_mix = weak_blur * (1.0 - depth_t) + strong_blur * depth_t
            bg_t = base_bg * (1.0 - bg_mask) + bokeh_mix * bg_mask

        # ===== 합성 =====
        comp_t = fgr_t * alpha2d + bg_t * (1.0 - alpha2d)

        main_person_mask_1hw = None

        # Ripple
        if bool(ripple_enabled):
            _dtype = comp_t.dtype
            comp_t = comp_t.float()
            comp_t = draw_moving_outlines(
                comp_rgb_3hw=comp_t,
                alpha_1chw=alpha,
                dilate_fn=dilate_gpu,
                t_seconds=time.time(),
                count=int(ripple_count),
                width_px=int(ripple_width),
                spacing_px=int(ripple_spacing),
                speed_px_s=float(ripple_speed),
                start_offset_px=int(ripple_start),
                color=str(ripple_color),
                blend=str(ripple_blend),
                on_bg_only=bool(ripple_on_bg),
                ripple_gain=float(ripple_gain),
            ).to(dtype=_dtype)

        fade_alpha = 1.0
        _prog = None
        pencil_runtime_on = bool(pencil_enabled)
        
        pencil_visible = bool(pencil_enabled)
        if pencil_sweep_ctrl is not None:
            fade_alpha = pencil_sweep_ctrl.get_fade_alpha()   # 0~1 디졸브
            _prog = pencil_sweep_ctrl.get_progress()          # 0~1 or None
            # 원래 pencil_enabled가 1이면서 fade가 거의 0 아니면 적용
            pencil_runtime_on = bool(pencil_enabled) and (fade_alpha > 1e-3)
        else:
            pencil_runtime_on = bool(pencil_enabled)

        # --- YOLO로 센터 사람 1명 bbox → 마스크 [1,H,W] (연필 켜져 있을 때만) ---
        if pencil_runtime_on and (yolo is not None):
            try:
                # Ultralytics YOLOv8-seg 기준
                yolo_results = yolo(
                    frame,                      # BGR 그대로 써도 대부분 동작함
                    imgsz=int(yolo_img),
                    conf=float(yolo_conf),
                    verbose=False,
                )
                res0 = yolo_results[0]
                boxes = res0.boxes

                xyxy = boxes.xyxy.detach().cpu().numpy()  # [N,4]
                cls  = boxes.cls.detach().cpu().numpy()   # [N]
                conf = boxes.conf.detach().cpu().numpy()  # [N]

                dets = []
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cid = int(cls[i])
                    cf  = float(conf[i])
                    dets.append((x1, y1, x2, y2, cid, cf))

                # 화면 중앙에 가장 가까운 사람 detection 고르기
                bbox, det_idx = _pick_center_person_bbox_from_yolo(
                    dets, W=W, H=H, person_class_id=0
                )

                masks_obj = getattr(res0, "masks", None)
                if (det_idx >= 0) and (masks_obj is not None) and (masks_obj.data is not None):
                    # YOLOv8-seg: masks.data → [N, h, w]
                    masks_data = masks_obj.data  # [N, h, w]
                    if det_idx < masks_data.shape[0]:
                        m_lo = masks_data[det_idx:det_idx+1]        # [1, h, w]
                        m_lo = m_lo.to(device=alpha.device, dtype=alpha.dtype)

                        # 고해상도로 업샘플 → [1, H, W]
                        m_hi = F.interpolate(
                            m_lo.unsqueeze(0),        # [1,1,h,w]
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)                   # [1,H,W]

                        main_person_mask_1hw = (m_hi > 0.5).float()
                    else:
                        # 인덱스 범위 이상하면 네모 bbox fallback
                        main_person_mask_1hw = _bbox_to_mask_1hw(
                            bbox, H=H, W=W,
                            device=alpha.device,
                            dtype=alpha.dtype,
                        )
                else:
                    # seg 정보 없으면 bbox fallback
                    if bbox is not None:
                        main_person_mask_1hw = _bbox_to_mask_1hw(
                            bbox, H=H, W=W,
                            device=alpha.device,
                            dtype=alpha.dtype,
                        )

            except Exception as _e:
                if energy_debug:
                    print("[YOLO] center-person mask fail:", _e)
        
        if main_person_mask_1hw is not None:
            try:
                expand_px =  25       # ← 경계 얼마나 넓힐지 (픽셀 단위, 3~10 사이 조절)
                feather_sigma = 1.5    # ← 가장자리 부드럽게 (가우시안 블러 강도)

                # [1,H,W] → [1,1,H,W]
                m = main_person_mask_1hw.unsqueeze(0)

                # 1) 팽창: 경계 바깥으로 expand_px 만큼 불려줌
                m = dilate_gpu(m, int(expand_px))

                # 2) feather: 너무 딱딱하지 않게 한 번 더 블러
                m = blur_alpha_gpu(m, float(feather_sigma))

                # 다시 [1,H,W]로
                main_person_mask_1hw = m.squeeze(0).clamp(0.0, 1.0)

            except Exception as _e:
                if energy_debug:
                    print("[YOLO] mask expand fail:", _e)

        # Pencil
        if pencil_runtime_on:
            comp_t = draw_pencil_outline(
                comp_rgb_3hw = comp_t,
                alpha_1hw    = alpha.squeeze(0),  # [1,H,W]
                pencil       = 1,
                pencil_color = str(pencil_color),
                pencil_color2= (None if pencil_color2 in (None, "None", "") else str(pencil_color2)),
                pencil_multicolor = int(pencil_multicolor),
                pencil_width = int(pencil_width),
                pencil_soft  = int(pencil_soft),
                pencil_gain  = float(pencil_gain),
                pencil_opacity = float(pencil_opacity)* float(fade_alpha),
                pencil_thresh  = float(pencil_thresh),
                pencil_rough   = float(pencil_rough),
                pencil_jitter  = float(pencil_jitter),
                pencil_dash    = int(pencil_dash),
                pencil_dash_len= int(pencil_dash_len),
                pencil_gap_len = int(pencil_gap_len),
                pencil_on_bg   = bool(pencil_on_bg),
                pencil_on_fg   = bool(pencil_on_fg),
                pencil_blend   = str(pencil_blend),
                pencil_palette = str(pencil_palette),
                pencil_bands   = int(pencil_bands),
                pencil_repeat  = int(pencil_repeat),
                pencil_axis    = str(pencil_axis),
                pencil_hatch         = int(pencil_hatch),
                pencil_hatch_mode    = str(pencil_hatch_mode),
                pencil_hatch_angle   = float(pencil_hatch_angle),
                pencil_hatch_period  = float(pencil_hatch_period),
                pencil_hatch_width   = float(pencil_hatch_width),
                pencil_hatch_phase   = float(pencil_hatch_phase),
                pencil_hatch_jitter  = float(pencil_hatch_jitter),
                pencil_sweep_prog=_prog,
                pencil_sweep_soft_px = int(pencil_sweep_soft_px),
                pencil_sweep_axis    = str(pencil_sweep_axis),
                main_person_mask_1hw = main_person_mask_1hw,
            )


        # ===== 에너지 (Y-band 포함) =====
        if bool(energy):
            # ripple 파라미터 동기 기본값
            try: ripple_spacing
            except NameError: ripple_spacing = 18
            try: ripple_speed
            except NameError: ripple_speed = 80.0
            try: ripple_start
            except NameError: ripple_start = 2

            # 최종 게인(펄스면 변조)
            eff_gain = float(energy_gain)
            if energy_pulse:
                tt = time.time()
                f = max(0.0, float(energy_pulse_speed))
                pulse = (0.5 * (math.sin(2.0*math.pi*f*tt) + 1.0)) if (energy_pulse_shape!="triangle") else \
                        (2.0 * (0.5 - abs((tt*f)%1.0 - 0.5)))
                k0, k1 = sorted([float(energy_pulse_min), float(energy_pulse_max)])
                eff_gain *= (k0 + (k1 - k0) * pulse)

            # ---- Tracking + 고정 색매핑 ----
            label_ids = []
            try:
                masks_src = instance_masks
                if masks_src is None:
                    masks_src = connected_instances_from_alpha(
                        alpha_1chw=alpha,
                        thr=float(energy_instance_thr if 'energy_instance_thr' in locals() else 0.5),
                        min_area_px=int(energy_min_area if 'energy_min_area' in locals() else 2000),
                        max_instances=int(energy_max_instances if 'energy_max_instances' in locals() else 4),
                        sort_by=str(energy_sort_by if 'energy_sort_by' in locals() else 'area'),
                    )

                if masks_src is not None and len(masks_src) > 0 and (tracker is not None):
                    dets = build_detections_from_masks(masks_src)  # [{'bbox':(x1,y1,x2,y2), ...}]
                    assigns = tracker.update(dets, frame_bgr=frame)  # [(tid, det_idx), ...]

                    if assigns:
                        ordered = []
                        for tid, di in assigns:
                            if tid not in color_map:
                                color_map[tid] = next_color
                                next_color = (next_color + 1) % palette_len
                            ordered.append((color_map[tid], tid, di))

                        ordered.sort(key=lambda x: x[0])  # 팔레트 순으로 고정
                        instance_masks = [masks_src[di] for (_, _, di) in ordered]
                        label_ids      = [tid          for (_, tid, _) in ordered]

                        if energy_debug:
                            print("[EnergyDebug][TRACK] assigns:", " ".join([f"{tid}:{di}" for (_, tid, di) in ordered]))
                    else:
                        label_ids = list(range(1, len(masks_src) + 1))
                        instance_masks = masks_src
                else:
                    label_ids = []
            except Exception as _e:
                if energy_debug:
                    print("[TRACK] fail:", _e)

            # ---- 렌더링 ----
            if str(energy_split).lower().strip() in ("center","abs","pct"):
                Hh, Ww = H, W
                # 1) split 위치/소프트폭
                xs = torch.arange(Ww, device=comp_t.device, dtype=torch.float32).view(1,1,1,Ww)  # float32 권장(밴딩↓)

                mode = str(energy_split).lower().strip()
                if mode == "center":
                    split_x = Ww * 0.5
                elif mode == "abs":
                    split_x = float(energy_split_x)
                else:  # "pct"
                    split_x = float(energy_split_x) * Ww

                soft = max(0.0, float(energy_split_soft))

                # 2) smoothstep s(x): 0..1 (좌→우)
                if soft <= 0.0:
                    s = (xs >= split_x).float()  # 하드컷
                else:
                    t = torch.clamp((xs - (split_x - soft*0.5)) / max(soft, 1e-6), 0.0, 1.0).to(torch.float32)
                    s = t*t*(3.0 - 2.0*t)  # smoothstep

                # 3) 픽셀별 컬러 보간: col(x)=lerp(left,right,s)
                colL = parse_rgb(left_color,  device=comp_t.device, dtype=torch.float32).view(3,1,1,1)  # [3,1,1,1]
                colR = parse_rgb(right_color, device=comp_t.device, dtype=torch.float32).view(3,1,1,1)
                colx = colL*(1.0 - s) + colR*s                                  # [3,1,1,W]
                colx = colx.expand(-1, 1, Hh, Ww).squeeze(1)                    # [3,H,W]

                # 4) 에너지 마스크(스칼라) 한 번만
                em = compute_energy_mask(alpha_1chw=alpha, on_bg=bool(energy_on_bg),
                                        width=int(energy_width), blur=float(energy_blur), gain=float(eff_gain))  # [1,1,H,W]
                em2d = em.squeeze(0).squeeze(0)  # [H,W]

                # 5) 최종 over_rgb = col(x) * energy
                over_rgb = torch.clamp(colx * em2d, 0.0, 1.0)  # [3,H,W]

                # 6) 블렌딩
                bmode = str(energy_blend).lower().strip()
                if bmode == "add":
                    comp_t = torch.clamp(comp_t + over_rgb, 0.0, 1.0)
                elif bmode == "screen":
                    comp_t = 1.0 - (1.0 - comp_t) * (1.0 - over_rgb)
                else:  # normal
                    comp_t = over_rgb + comp_t * (1.0 - em2d)  # energy가 알파 역할
            else:
                # (기존 단색 에너지) — split이 off일 때만
                runtime_color = energy_color_source() if (energy_color_source is not None) else None
                color_arg = runtime_color if (runtime_color is not None) else energy_color
                comp_t = apply_energy(
                    comp_rgb_3hw=comp_t, alpha_1chw=alpha, color=color_arg,
                    width=int(energy_width), blur=float(energy_blur), gain=float(eff_gain),
                    on_bg=bool(energy_on_bg), pulse=bool(energy_pulse),
                    pulse_speed=float(energy_pulse_speed),
                    pulse_min=float(energy_pulse_min), pulse_max=float(energy_pulse_max),
                    pulse_shape=str(energy_pulse_shape),
                    y_mode=str(energy_y_band_mode),
                    y_center=float(energy_y_band_center),
                    y_width=float(energy_y_band_width),
                    y_sigma=float(energy_y_band_sigma),
                    y_bbox_down_h=240,
                    aura=bool(aura),
                    aura_amp=float(aura_amp),
                    aura_freq=float(aura_freq),
                    aura_speed=float(aura_speed),
                    aura_twirl=float(aura_twirl),
                    aura_rgb_phase=float(aura_rgb_phase),
                    aura_down=int(aura_down),
                    aura_roi_thr=float(aura_roi),
                )
                
        # ========= END: ENERGY BLOCK =========

        # (예외 squeeze 방지)
        if comp_t.dim() == 4 and comp_t.size(0) == 1:
            comp_t = comp_t.squeeze(0)   # [1,3,H,W] -> [3,H,W]

        # ===== 출력 준비(torch -> numpy BGR u8) =====
        # Preview-only mode performs its own scaled transfer below. Avoid a
        # second full-resolution GPU->CPU copy when no external sender exists.
        bgr_out = None
        if sender is not None:
            out_u8_bgr = (
                comp_t[[2,1,0], ...]
                .permute(1,2,0)
                .mul(255.0).clamp(0,255).to(torch.uint8).contiguous()
            )
            # The NumPy consumer runs immediately, so this copy must complete
            # before exposing cpu_buf. non_blocking=True caused torn/stale frames.
            cpu_buf.copy_(out_u8_bgr, non_blocking=False)
            bgr_out = cpu_buf.numpy()

        # ---- Debug: draw per-person IDs on output frame ----
        try:
            if energy_debug and bgr_out is not None:
                masks_for_labels = None
                try:
                    if (instance_masks is not None) and len(instance_masks) > 0:
                        masks_for_labels = instance_masks
                except Exception:
                    masks_for_labels = None
                if masks_for_labels is None:
                    try:
                        masks_for_labels = connected_instances_from_alpha(
                            alpha_1chw=alpha,
                            thr=float(energy_instance_thr if 'energy_instance_thr' in locals() else 0.5),
                            min_area_px=int(energy_min_area if 'energy_min_area' in locals() else 2000),
                            max_instances=int(energy_max_instances if 'energy_max_instances' in locals() else 4),
                            sort_by=str(energy_sort_by if 'energy_sort_by' in locals() else 'area'),
                        )
                    except Exception:
                        masks_for_labels = []

                label_pts = []
                for m1hw in (masks_for_labels or []):
                    try:
                        m = (m1hw[0] > 0.5)
                        idx = torch.nonzero(m, as_tuple=False)
                        if idx.numel() == 0:
                            continue
                        ys = idx[:, 0].to(torch.float32); xs = idx[:, 1].to(torch.float32)
                        x_mean = int(xs.mean().item()); y_top = int(ys.min().item())
                        label_pts.append((x_mean, max(0, y_top - 10)))
                    except Exception:
                        continue

                colors_bgr = []
                try:
                    pal = parse_palette(str(energy_colors if 'energy_colors' in locals() else '255,0,0|0,255,0|0,128,255|255,255,0'), device='cpu')
                    for r,g,b in pal.cpu().numpy().tolist():
                        colors_bgr.append((int(b*255), int(g*255), int(r*255)))
                except Exception:
                    pass

                for i, (lx, ly) in enumerate(label_pts):
                    col = colors_bgr[i % len(colors_bgr)] if colors_bgr else (0,255,255)
                    tid = (i < len(label_ids) and label_ids[i]) or (i+1)
                    cv2.putText(bgr_out, f"id:{tid}", (int(lx), int(ly)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        except Exception:
            pass

        # 전송
        if sender is not None and bgr_out is not None:
            sender.submit(bgr_out.copy())

        # 누끼 전송: 이펙트 없이 RVM 전경색 + 알파 (straight alpha, BGRA)
        if matte_sender is not None:
            matte_u8_bgra = (
                torch.cat([fgr_t[[2,1,0], ...], alpha2d], dim=0)
                .permute(1,2,0)
                .mul(255.0).clamp(0,255).to(torch.uint8).contiguous()
            )
            matte_cpu_buf.copy_(matte_u8_bgra, non_blocking=False)
            matte_np = matte_cpu_buf.numpy().copy()
            if distance_estimator is not None:
                encode_distance(matte_np, person_z, person_z_valid)
                if logo_every > 0 and (time.perf_counter() - logo_last_auto) >= float(logo_every):
                    logo_last_auto = logo_t0 = time.perf_counter()
                logo_phase = ((time.perf_counter() - logo_t0) / float(logo_duration)) if logo_t0 is not None else 1.0
                logo_active = logo_phase < 1.0
                if not logo_active:
                    logo_t0 = None
                encode_logo(matte_np, logo_phase if logo_active else 0.0, logo_active, logo_anchor)
            matte_sender.submit(matte_np)

        # 타겟 마스크 전송 (BGRA): R=타겟 알파(RVM 경계 × 타겟), G=다른 사람 알파, B=타겟 영역(grow 포함), A=255
        # 타겟이 없으면 R=B=0(언리얼에서 효과 꺼짐), G=전체 알파.
        if target_sender is not None:
            if target_w is not None:
                tw = target_w.to(alpha2d.dtype)
                target_bgra = torch.cat(
                    [tw, alpha2d * (1.0 - tw), alpha2d * tw, torch.ones_like(tw)], dim=0)
            else:
                target_bgra = torch.cat(
                    [torch.zeros_like(alpha2d), alpha2d, torch.zeros_like(alpha2d), torch.ones_like(alpha2d)], dim=0)
            target_u8 = target_bgra.permute(1,2,0).mul(255.0).clamp(0,255).to(torch.uint8).contiguous()
            target_cpu_buf.copy_(target_u8, non_blocking=False)
            target_sender.submit(target_cpu_buf.numpy().copy())

        # 배경 전송: clean 모드는 사람(+여유 영역)이 없는 픽셀만 천천히 갱신해 사람 없는 배경을 만든다.
        if bg_sender is not None:
            if str(bg_mode) == "clean":
                m = max(0, int(bg_margin))
                grown = F.max_pool2d(alpha, kernel_size=2*m+1, stride=1, padding=m)[0] if m > 0 else alpha[0]
                visible = (grown < 0.02).to(base_bg.dtype)                  # [1,H,W]
                if bg_plate is None:
                    bg_plate = base_bg.clone()
                    # B 키 리셋이면 화면 전체를 '본 배경'으로 인정, 시작 시에는 사람 영역을 아직 못 본 것으로 둔다.
                    bg_seen = torch.ones_like(visible, dtype=torch.bool) if bg_reset_all else visible.bool()
                    bg_reset_all = False
                    bg_fill = None
                    bg_frames = 0
                else:
                    bg_plate = bg_plate + (base_bg - bg_plate) * (visible * float(bg_rate))
                    bg_seen |= visible.bool()
                bg_frames += 1
                # 한 번도 드러나지 않은 영역은 첫 프레임 대신 주변 배경으로 채운다(저해상도 인페인팅, 15프레임마다).
                if not bool(bg_seen.all()):
                    if bg_fill is None or bg_frames % 15 == 1:
                        sw, sh = W // 4, H // 4
                        small = F.interpolate(bg_plate[None], size=(sh, sw), mode="area")[0]
                        small_u8 = (small[[2,1,0]].permute(1,2,0).clamp(0,1) * 255).to(torch.uint8).cpu().numpy()
                        hole = F.interpolate((~bg_seen).float()[None], size=(sh, sw), mode="area")[0, 0]
                        hole_u8 = ((hole > 0.01).to(torch.uint8) * 255).cpu().numpy()
                        hole_u8 = cv2.dilate(hole_u8, np.ones((5, 5), np.uint8))
                        filled = cv2.inpaint(small_u8, hole_u8, 5, cv2.INPAINT_TELEA)
                        filled_t = torch.from_numpy(filled).to(bg_plate.device).permute(2,0,1)[[2,1,0]].float() / 255.0
                        bg_fill = F.interpolate(filled_t[None], size=(H, W), mode="bilinear", align_corners=False)[0]
                    bg_src = torch.where(bg_seen, bg_plate, bg_fill)
                else:
                    bg_src = bg_plate
            else:
                bg_src = base_bg
            bg_u8_bgr = (
                bg_src[[2,1,0], ...]
                .permute(1,2,0)
                .mul(255.0).clamp(0,255).to(torch.uint8).contiguous()
            )
            bg_cpu_buf.copy_(bg_u8_bgr, non_blocking=False)
            bg_sender.submit(bg_cpu_buf.numpy().copy())

        # 프리뷰
        if not no_preview:
            disp = comp_t
            if target_w is not None:
                # 프리뷰에서만 타겟 사람을 붉게 표시
                ta = (alpha2d * target_w.to(alpha2d.dtype)) * 0.45
                red = torch.tensor([1.0, 0.1, 0.1], device=ta.device, dtype=ta.dtype).view(3, 1, 1)
                disp = disp * (1.0 - ta) + red * ta
            if preview_scale < 1.0:
                with torch.inference_mode():
                    disp = F.interpolate(
                        disp.unsqueeze(0).float(), scale_factor=preview_scale,
                        mode="bilinear", align_corners=False
                    ).squeeze(0)
            comp_np = (disp.permute(1,2,0).detach().clamp(0,1).mul(255.0).byte().cpu().numpy())
            bgr = comp_np[:, :, ::-1].copy()

            # ---- Debug: IDs on preview ----
            try:
                if energy_debug:
                    masks_for_labels = None
                    try:
                        if (instance_masks is not None) and len(instance_masks) > 0:
                            masks_for_labels = instance_masks
                    except Exception:
                        masks_for_labels = None
                    if masks_for_labels is None:
                        try:
                            masks_for_labels = connected_instances_from_alpha(
                                alpha_1chw=alpha,
                                thr=float(energy_instance_thr if 'energy_instance_thr' in locals() else 0.5),
                                min_area_px=int(energy_min_area if 'energy_min_area' in locals() else 2000),
                                max_instances=int(energy_max_instances if 'energy_max_instances' in locals() else 4),
                                sort_by=str(energy_sort_by if 'energy_sort_by' in locals() else 'area'),
                            )
                        except Exception:
                            masks_for_labels = []

                    pts = []
                    for m1hw in (masks_for_labels or []):
                        try:
                            m = (m1hw[0] > 0.5)
                            idx = torch.nonzero(m, as_tuple=False)
                            if idx.numel() == 0:
                                continue
                            ys = idx[:, 0].to(torch.float32); xs = idx[:, 1].to(torch.float32)
                            x_mean = int(xs.mean().item()); y_top = int(ys.min().item())
                            pts.append((x_mean, max(0, y_top - 10)))
                        except Exception:
                            continue

                    colors_bgr = []
                    try:
                        pal = parse_palette(str(energy_colors if 'energy_colors' in locals() else '255,0,0|0,255,0|0,128,255|255,255,0'), device='cpu')
                        for r,g,b in pal.cpu().numpy().tolist():
                            colors_bgr.append((int(b*255), int(g*255), int(r*255)))
                    except Exception:
                        pass

                    if pts:
                        sx = bgr.shape[1] / float(W)
                        sy = bgr.shape[0] / float(H)
                        for i, (lx, ly) in enumerate(pts):
                            col = colors_bgr[i % len(colors_bgr)] if colors_bgr else (0,255,255)
                            tid = (i < len(label_ids) and label_ids[i]) or (i+1)
                            cv2.putText(bgr, f"id:{tid}", (int(lx*sx), int(ly*sy)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            except Exception:
                pass

            if distance_estimator is not None:
                txt = (f"dist {person_z:.2f} m ({distance_estimator.source})" if person_z_valid else "dist -")
                cv2.putText(bgr, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if show_depth and (last_depth is not None):
                dvis = (np.clip(last_depth, 0, 1) * 255).astype(np.uint8)
                dvis = cv2.cvtColor(dvis, cv2.COLOR_GRAY2BGR)
                dvis = cv2.resize(dvis, (bgr.shape[1] // 5, bgr.shape[0] // 5))
                bgr[10:10+dvis.shape[0], 10:10+dvis.shape[1]] = dvis
            cv2.imshow(preview_title, bgr)
            cmd = PREVIEW_KEY_CMDS.get(cv2.waitKey(1) & 0xFF)
            if cmd:
                pending_cmds.append(cmd)

        # 런타임 명령: 프리뷰 창 키 + 콘솔(cmd 창) 키 → 같은 처리
        if console_cmds is not None:
            while True:
                try:
                    pending_cmds.append(console_cmds.get_nowait())
                except queue.Empty:
                    break
        quit_requested = False
        for cmd in pending_cmds:
            if cmd == "quit":
                quit_requested = True
            elif cmd == "logo" and distance_estimator is not None:
                logo_t0 = time.perf_counter()
                print("[INFO] Logo: play")
            elif cmd == "bg_reset" and bg_sender is not None:
                # 사람이 없을 때 누르면 현재 프레임 전체를 배경으로 다시 잡는다.
                bg_plate = None
                bg_reset_all = True
                print("[INFO] Spout BG: background plate reset to current frame")
            elif cmd == "target_next" and target_selector is not None:
                target_selector.cycle()
                print("[INFO] Target: next person")
            elif cmd == "target_reset" and target_selector is not None:
                target_selector.reset()
                print(f"[INFO] Target: re-pick by mode '{target_selector.mode}'")
        pending_cmds.clear()
        if quit_requested:
            print("[INFO] Quit requested")
            break

        # ===== FPS 로그 =====
        frames += 1
        if show_fps and (frames % 30 == 0):
            dt_sum = time.perf_counter() - t0
            fps = (frames / dt_sum) if dt_sum > 0 else 0.0
            prep_ms  = (prep_t1  - prep_t0) * 1000.0
            infer_ms = (infer_t1 - infer_t0) * 1000.0
            post_ms  = (time.perf_counter() - infer_t1) * 1000.0
            if bool(getattr(sender, "dl", False)):
                # run_virtualcam에서 dl_out_fps 넘겼다면 그 문자열을 run_loop 인자로 받아 쓰는게 베스트
                tgtfps = _parse_fps(force_fps_str)  # 임시: IN 또는 OUT 중 실제와 맞는 값으로
            else:
                tgtfps = _parse_fps(target_fps_str)
            capture_note = ""
            stats_fn = getattr(grab, "stats", None)
            if callable(stats_fn):
                capture_stats = stats_fn()
                capture_note = (f" | captured={capture_stats.get('captured', 0)}"
                                f" stale_drop={capture_stats.get('stale_dropped', 0)}")
            if distance_estimator is not None:
                capture_note += (f" | dist={person_z:.2f}m src={distance_estimator.source}"
                                 if person_z_valid else " | dist=-")
            if seg_worker is not None:
                seg_boxes = seg_worker.state.get("boxes")
                capture_note += (f" | people={0 if seg_boxes is None else len(seg_boxes)}"
                                 f" target={'on' if target_w is not None else '-'}")
            print(f"[INFO] {W}x{H} long={cur_long} ds={_compute_ds_ratio(H,W,cur_long):.3f} "
                  f"target={tgtfps:.2f} actual={fps:.1f} | prep={prep_ms:.1f}ms "
                  f"infer={infer_ms:.1f}ms post={post_ms:.1f}ms{capture_note}", flush=True)
            frames = 0; t0 = time.perf_counter()

        # ===== 타이밍 =====
        use_dl_clock = bool(sender) and bool(getattr(sender, "dl", False))
        if not use_dl_clock:
            # 2) 그렇지 않으면 target_fps_str로 슬립
            tf = _parse_fps(target_fps_str)
            if tf > 0 and not (sender and getattr(sender, "use_sleep", False)):
                elapsed = time.perf_counter() - loop_perf_start
                sleep_s = (1.0 / tf) - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    if not no_preview:
        cv2.destroyWindow(preview_title)
# ======================================================================
