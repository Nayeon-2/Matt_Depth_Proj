# engine/loop.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, math, queue
from typing import Optional

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from module.effects.alpha import (
    blur_alpha_gpu, unsharp_mask_alpha,
    dilate_gpu, erode_gpu, close_gpu,
    ema_smooth_alpha,
)
from module.effects.selector import select_component
from module.effects.bokeh import apply_depth_bokeh
from module.effects.energy import apply_energy
from module.effects.spark import SparkState, ensure_and_step_spark


def _compute_ds_ratio(h, w, target_long: int) -> float:
    return float(min(target_long / max(h, w), 1.0))


def _parse_fps(s):
    s = str(s)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def run_loop(
    *,
    # 필수 입출력
    grab,                   # .read(timeout)->(ok, frame) 를 지원하는 입력
    W: int, H: int,         # 입력 해상도
    rvm,                    # RVM 네트워크 (callable)
    device: str = "cuda",
    fp16: bool = True,

    # RVM 해상도/페이싱
    target_long: int = 768,
    target_fps_str: str = "60000/1001",
    force_fps_str: str = "60000/1001",   # decklink용
    ffmpeg_in: str = "dshow",

    # Depth
    use_depth: bool = True,
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

    # 에너지
    energy: bool = False,
    energy_color: str = "0,255,255",
    energy_width: int = 3,
    energy_blur: float = 12.0,
    energy_gain: float = 0.85,
    energy_on_bg: bool = True,
    energy_pulse: bool = False,
    energy_pulse_speed: float = 1.0,
    energy_pulse_min: float = 0.7,
    energy_pulse_max: float = 1.3,
    energy_pulse_shape: str = "sine",

    # 스파크
    spark: bool = False,
    spark_count: int = 300,
    spark_spawn: int = 80,
    spark_color: str = "255,200,80",
    spark_size: int = -5,
    spark_speed: float = 220.0,
    spark_jitter: float = 0.6,
    spark_gravity: float = 500.0,
    spark_lifespan: float = 0.80,
    spark_trail: float = 0.0,
    spark_gain: float = 1.0,
    spark_on_bg: bool = True,
    spark_seed: int = 1234,

    # 프리뷰/출력
    sender=None,                # ObsSender 또는 None
    no_preview: bool = False,
    preview_scale: float = 0.5,
    show_depth: bool = False,

    # 로그
    show_fps: bool = True,
    log_detail: int = 1,
):
    """메인 루프. 외부에서 준비된 입력/모델/옵션을 받아 프레임을 처리한다."""

    # 동적 long / 상태 변수
    prev_motion_mag = None
    base_long_hi = int(target_long)
    base_long_lo = max(512, base_long_hi - 128)
    prev_ds = None

    rec = [None, None, None, None]
    alpha_prev = None
    prev_keep = None

    # Depth 최신값
    last_depth = None

    # Spark 상태
    spark_state = SparkState()

    print("Press 'q' to quit.")
    t0 = time.time(); frames = 0

    while True:
        loop_start = time.time()
        ok, frame = grab.read(timeout=1.0)
        if not ok:
            print("[WARN] 프레임 읽기 실패."); break

        # Depth 비동기 제출
        if use_depth and depth_worker is not None:
            depth_worker.submit(frame)

        # ===== 전처리 =====
        prep_t0 = time.time()
        frame_c = np.ascontiguousarray(frame).copy()
        u8 = torch.from_numpy(frame_c)
        if torch.cuda.is_available():
            try: u8 = u8.pin_memory()
            except Exception: pass
        u8 = u8.to(device, non_blocking=True).permute(2,0,1)  # BGR->CHW
        rgb01 = u8[[2,1,0], ...].to(dtype=(torch.float16 if fp16 else torch.float32)) * (1.0/255.0)
        t_in = rgb01.unsqueeze(0)
        try: t_in = t_in.contiguous(memory_format=torch.channels_last)
        except Exception: pass
        prep_t1 = time.time()

        # ===== Depth 최신값 취득/동기 추론 =====
        if use_depth and depth_worker is not None and depth_worker.latest is not None:
            last_depth = depth_worker.latest
        elif use_depth and (depth_worker is None) and (depth_model is not None):
            d = depth_model.infer(frame)           # 동기로 추론
            last_depth = (1.0 - d) if depth_invert else d

        # ===== RVM 추론 =====
        infer_t0 = time.time()
        cur_long = base_long_hi if (prev_motion_mag is None or prev_motion_mag < 0.03) else base_long_lo
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
        infer_t1 = time.time()

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

        # 대상 변화 감지 → rec/EMA 리셋
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

        alpha2d = alpha.squeeze(0)        # [1,H,W]
        fgr_t   = fgr[0].clamp(0,1)       # RGB
        base_bg = t_in[0]                 # RGB

        # ===== 배경(보케) =====
        if use_depth and depth_bokeh:
            bg_t = apply_depth_bokeh(
                base_bg_3hw=base_bg,
                alpha_1hw=alpha2d,
                depth_np=(last_depth if last_depth is not None else None),
                invert_depth=bool(depth_invert),
                blur_weak=float(blur_weak),
                blur_strong=float(blur_strong),
            )
        else:
            bg_t = base_bg

        # 합성
        comp_t = fgr_t * alpha2d + bg_t * (1.0 - alpha2d)

        # ===== 에너지 =====
        if energy:
            comp_t = apply_energy(
                comp_rgb_3hw=comp_t,
                alpha_1chw=alpha,
                color=energy_color,
                width=int(energy_width),
                blur=float(energy_blur),
                gain=float(energy_gain),
                on_bg=bool(energy_on_bg),
                pulse=bool(energy_pulse),
                pulse_speed=float(energy_pulse_speed),
                pulse_min=float(energy_pulse_min),
                pulse_max=float(energy_pulse_max),
                pulse_shape=str(energy_pulse_shape),
            )

        # ===== 스파크 =====
        if spark:
            comp_t = ensure_and_step_spark(
                spark_state,
                H=H, W=W,
                alpha_1chw=alpha,
                comp_rgb_3hw=comp_t,
                spawn=int(spark_spawn), count=int(spark_count), color=str(spark_color),
                size_px=int(spark_size), speed=float(spark_speed), jitter=float(spark_jitter),
                gravity=float(spark_gravity), lifespan=float(spark_lifespan), trail=float(spark_trail),
                gain=float(spark_gain), on_bg=bool(spark_on_bg), seed=int(spark_seed),
            )

        # ===== 출력(BGR) =====
        out_np = (
            comp_t.permute(1, 2, 0)
                 .detach()
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .cpu()
                 .numpy()
        )
        bgr_out = np.empty_like(out_np)
        bgr_out[..., 0] = out_np[..., 2]
        bgr_out[..., 1] = out_np[..., 1]
        bgr_out[..., 2] = out_np[..., 0]

        # ===== Noise Static (치직 효과) =====
        bgr_out = noise_effect.apply(bgr_out)

        # OBS 전송
        if sender is not None:
            sender.submit(bgr_out.copy())

        # 프리뷰
        if not no_preview:
            disp = comp_t
            if preview_scale < 1.0:
                with torch.inference_mode():
                    disp = F.interpolate(disp.unsqueeze(0).float(),
                                         scale_factor=preview_scale,
                                         mode="bilinear", align_corners=False).squeeze(0)
            comp_np = (disp.permute(1,2,0).detach().clamp(0,1).mul(255.0).byte().cpu().numpy())
            bgr = comp_np[:, :, ::-1].copy()
            if show_depth and (last_depth is not None):
                dvis = (np.clip(last_depth, 0, 1) * 255).astype(np.uint8)
                dvis = cv2.cvtColor(dvis, cv2.COLOR_GRAY2BGR)
                dvis = cv2.resize(dvis, (bgr.shape[1] // 5, bgr.shape[0] // 5))
                bgr[10:10 + dvis.shape[0], 10:10 + dvis.shape[1]] = dvis
            cv2.imshow("RVM+Depth (multi-mode)", bgr)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        # ===== 로그 =====
        frames += 1
        if show_fps and (frames % 30 == 0):
            dt_sum = time.time() - t0
            fps = (frames / dt_sum) if dt_sum > 0 else 0.0
            prep_ms  = (prep_t1  - prep_t0) * 1000.0
            infer_ms = (infer_t1 - infer_t0) * 1000.0
            post_ms  = (time.time() - infer_t1) * 1000.0
            tgtfps = _parse_fps(force_fps_str if ffmpeg_in=='decklink' else target_fps_str)
            print(f"[INFO] {W}x{H} long(cur)={cur_long} ds={_compute_ds_ratio(H,W,cur_long):.3f} "
                  f"target={tgtfps:.2f} FPS≈{fps:.1f} | prep={prep_ms:.1f}ms "
                  f"infer={infer_ms:.1f}ms post={post_ms:.1f}ms", flush=True)
            frames = 0; t0 = time.time()

        # ===== 페이싱 =====
        tf = _parse_fps(target_fps_str)
        if tf > 0 and not (sender and getattr(sender, "use_sleep", False)):
            elapsed = time.time() - loop_start
            sleep_s = (1.0 / tf) - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
