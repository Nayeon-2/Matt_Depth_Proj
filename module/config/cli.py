# config/cli.py
from __future__ import annotations
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RVM + MiDaS Depth + (DeckLink/DirectShow) 입력 → OBS 가상캠/프리뷰 파이프라인"
    )

    # 입력 (DeckLink/ffmpeg)
    g = p.add_argument_group("Input / FFmpeg-DeckLink")
    # g.add_argument("--ffmpeg-in", type=str, default=None, choices=["decklink"],
    #                help="ffmpeg로 DeckLink를 사용할 때 'decklink' 지정 (미지정 시 DirectShow)")
    g.add_argument(
    "--ffmpeg-in",
    type=str,
    default="dshow",                 # 기본을 dshow로
    choices=["decklink", "dshow"],   # ← dshow 추가
    help="FFmpeg 입력 백엔드 선택 (decklink | dshow). 미지정 시 dshow(OBS 가상카메라 등)")
    g.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg 실행 파일 경로")
    g.add_argument("--decklink-device", type=str, default=None, help="DeckLink 장치명 (예: 'DeckLink Quad (1)')")
    g.add_argument("--decklink-format", type=str, default=None, help="(옵션) DeckLink 입력 포맷 표시용")
    g.add_argument("--raw-format", type=str, default="bgr24",
                   choices=["bgr24", "bgra", "yuyv422", "uyvy422"],
                   help="FFmpeg rawvideo 픽셀 포맷 (DirectShow는 bgr24 권장)")

    # DirectShow
    g = p.add_argument_group("Input / DirectShow")
    g.add_argument("--dshow-name", type=str, default="OBS Virtual Camera", help="DirectShow 장치 이름")
    g.add_argument("--force-size", type=str, default="1920x1080", help="입력 강제 해상도 (예: 1920x1080)")
    g.add_argument("--force-fps", type=str, default="60000/1001", help="입력 강제 fps (예: 60000/1001)")
    g.add_argument(
        "--rtbufsize",
        type=str,
        default="256M",
        help="FFmpeg/PyAV 실시간 입력 버퍼 크기 (기본값: 256M)",
    )
    g.add_argument("--input-timeout", type=float, default=0.5,
                   help="입력 프레임 1회 대기 시간(초)")
    g.add_argument("--max-input-timeouts", type=int, default=10,
                   help="연속 입력 타임아웃 허용 횟수(0이면 무제한)")

    # RVM
    g = p.add_argument_group("RVM (Robust Video Matting)")
    g.add_argument("--backbone", type=str, default="mobilenetv3", choices=["mobilenetv3", "resnet50"])
    g.add_argument("--weights", type=str, default=None,
                   help="RVM 가중치 경로. 생략하면 현재 프로젝트의 backbone별 .pth 사용")
    g.add_argument("--preset", type=str, default="balanced", choices=["speed", "balanced", "quality"])
    g.add_argument("--long", type=int, default=872, dest="long",
                   help="RVM 입력 다운샘플 기준 긴 변(target_long)")
    g.add_argument("--device", type=str, default="cuda", help="연산 디바이스 (cuda/cpu)")
    g.add_argument("--fp16", action="store_true", default=True, help="FP16 사용")
    g.add_argument("--no-fp16", dest="fp16", action="store_false", help="FP16 비활성화")

    # 프리뷰/로그
    g = p.add_argument_group("Preview / Logging")
    g.add_argument("--no-preview", dest="no_preview", action="store_true", help="프리뷰 창 끄기")
    g.add_argument("--preview", dest="no_preview", action="store_false", help="프리뷰 창 강제로 켜기")
    g.set_defaults(no_preview=False)
    g.add_argument("--preview-scale", type=float, default=0.5, help="프리뷰 스케일 팩터")
    g.add_argument("--show-fps", type=int, default=1, help="FPS 로그 출력(1/0)")
    g.add_argument("--show-depth", type=int, default=0, help="프리뷰에 depth 미니맵 표시(1/0)")

    # OBS 출력
    g = p.add_argument_group("OBS VirtualCam Output")
    g.add_argument("--obs-out", type=int, default=0, help="OBS 가상캠 출력 사용(1/0)")
    g.add_argument("--obs-out-sync", type=int, default=0,
                   help="pyvirtualcam.sleep_until_next_frame 사용(1/0)")

    # Spout output (Unreal Engine / TouchDesigner)
    g = p.add_argument_group("Spout Output")
    g.add_argument("--spout-out", type=int, default=0, help="Spout 출력 사용(1/0)")
    g.add_argument("--spout-name", type=str, default="RVM_OUTPUT", help="Spout sender 이름")
    g.add_argument("--spout-invert", type=int, default=0, help="Spout 텍스처 상하 반전(1/0)")
    g.add_argument("--spout-mode", type=str, default="comp", choices=["comp", "matte"],
                   help="comp: 프리뷰와 같은 합성 화면 / matte: 사람 누끼만 RGBA(알파 포함)로 전송")
    g.add_argument("--spout-bg", type=int, default=0, help="배경을 별도 Spout sender로 전송(1/0)")
    g.add_argument("--spout-bg-name", type=str, default="RVM_BG", help="배경 Spout sender 이름")
    g.add_argument("--spout-bg-mode", type=str, default="clean", choices=["clean", "raw"],
                   help="clean: 사람 없는 영역만 누적한 배경 / raw: 입력 프레임 그대로")
    g.add_argument("--spout-bg-rate", type=float, default=0.15, help="clean 배경 갱신 속도(0~1)")
    g.add_argument("--spout-bg-margin", type=int, default=15, help="clean 배경에서 사람 주변 제외 여유(px)")

    # Person distance (pose body size + foot contact) → matte 좌상단 픽셀로 언리얼에 전달
    g = p.add_argument_group("Person Distance")
    g.add_argument("--person-distance", type=int, default=0, help="사람 거리 추정 사용(1/0), matte 모드 필요")
    g.add_argument("--dist-hfov", type=float, default=60.0, help="입력 카메라 가로 화각(도)")
    g.add_argument("--dist-cam-height", type=float, default=None, help="카메라 높이(m). 미지정 시 자동 보정")
    g.add_argument("--dist-cam-pitch", type=float, default=None, help="카메라 아래 기울기(도). 미지정 시 자동 보정")
    g.add_argument("--dist-calib", type=str, default="ground_calib.json", help="자동 보정 결과 저장 파일")
    g.add_argument("--dist-pose-model", type=str, default="yolov8n-pose.pt", help="YOLO pose 가중치")
    g.add_argument("--dist-pose-imgsz", type=int, default=640, help="YOLO pose 추론 해상도")

    # Depth (MiDaS)
    g = p.add_argument_group("Depth (MiDaS)")
    g.add_argument("--use-depth", type=int, default=0, help="깊이 사용(1/0)")
    g.add_argument("--depth-model-name", type=str, default="MiDaS_small")
    g.add_argument("--depth-long", type=int, default=384, help="MiDaS 입력 긴 변")
    g.add_argument("--depth-async", type=int, default=1, help="깊이 비동기 추론(1/0)")
    g.add_argument("--depth-fps", type=float, default=12.0, help="깊이 워커 최대 FPS")
    g.add_argument("--depth-invert", type=int, default=1, help="깊이 반전(1/0)")
    g.add_argument("--depth-bokeh", type=int, default=0, help="깊이 기반 배경 보케(1/0)")
    g.add_argument("--blur-weak", type=float, default=1.2)
    g.add_argument("--blur-strong", type=float, default=4.0)

    # Alpha 튜닝
    g = p.add_argument_group("Alpha Postprocess")
    g.add_argument("--alpha-sigma", type=float, default=0.4)
    g.add_argument("--alpha-unsharp", type=float, default=0.35)
    g.add_argument("--alpha-ema", type=float, default=0.8)
    g.add_argument("--alpha-close", type=int, default=3)
    g.add_argument("--alpha-dilate", type=int, default=3)

    # Energy 효과
    g = p.add_argument_group("Energy Effect")
    g.add_argument("--energy", type=int, default=0)
    g.add_argument("--energy-color", type=str, default="0,255,255")
    g.add_argument("--energy-width", type=int, default=3)
    g.add_argument("--energy-blur", type=float, default=12.0)
    g.add_argument("--energy-gain", type=float, default=0.85)
    g.add_argument("--energy-on-bg", type=int, default=1)

    g.add_argument("--energy-edge-sobel", type=int, default=1)        # 1=Sobel(얇고 정확), 0=모폴로지 그라디언트
    g.add_argument("--energy-tight-attach", type=int, default=1)      # 블러 후 현재 경계 주변으로 재마스킹
    g.add_argument("--energy-attach-k", type=int, default=3)          # 재마스킹 범위(px)
    g.add_argument("--energy-thresh-lo", type=float, default=0.05)    # 약한 엣지 컷(low)
    g.add_argument("--energy-thresh-hi", type=float, default=0.15)    # 약한 엣지 컷(high)
    g.add_argument("--energy-temporal-beta", type=float, default=0.0)

    # Energy Pulse
    g = p.add_argument_group("Energy Pulse")
    g.add_argument("--energy-pulse", type=int, default=0)
    g.add_argument("--energy-pulse-speed", type=float, default=1.0)
    g.add_argument("--energy-pulse-min", type=float, default=0.7)
    g.add_argument("--energy-pulse-max", type=float, default=1.3)
    g.add_argument("--energy-pulse-shape", type=str, default="sine", choices=["sine", "triangle"])

    # --- Energy color smoothing (runtime) ---
    p.add_argument("--energy-color-lerp-ms", type=int, default=250,
                        help="에너지 색 전환 선형 보간 시간(ms). 0이면 비활성")
    p.add_argument("--energy-color-exp-tau-ms", type=int, default=0,
                        help="에너지 색 지수 감쇠 시간상수(ms). lerp=0일 때만 적용")
    
    # --- Energy debug ---
    p.add_argument("--energy-debug", type=int, default=0, help="Energy per-person debug logs (1/0)")

    # --- Energy vertical band gating ---
    p.add_argument("--energy-y-band-mode", type=str, default="off",
                    choices=["off", "abs", "bbox"],      # ← rel 대신 bbox 포함
                    help="에너지 표시를 세로 밴드로 제한 (off/abs/bbox)")
    p.add_argument("--energy-y-band-center", type=float, default=50.0,
                        help="밴드 중심 (abs: 0=화면 상단 px, bbox: 0~100 퍼센트)")
    p.add_argument("--energy-y-band-width", type=float, default=10.0,
                        help="밴드 폭 (abs: px, bbox: 퍼센트)")
    p.add_argument("--energy-y-band-sigma", type=float, default=0.0,
                        help="밴드 가장자리 소프트닝 정도(px 단위 가우시안 sigma)")
    
    p.add_argument("--aura", type=int, default=1)            # 1:on, 0:off
    p.add_argument("--aura-amp", type=float, default=0.35)
    p.add_argument("--aura-freq", type=float, default=1.6)
    p.add_argument("--aura-speed", type=float, default=0.6)
    p.add_argument("--aura-twirl", type=float, default=0.8)
    p.add_argument("--aura-rgb-phase", type=float, default=0.0)
    # 퍼포먼스용(선택)
    p.add_argument("--aura-down", type=int, default=1)       # 1=원본, 2/3 권장
    p.add_argument("--aura-roi", type=float, default=0.0)    # 0=전체, 0.05~0.2 권장
    
    g = p.add_argument_group("Energy Per-Person")
    g.add_argument("--energy-per-person",    type=int,   default=0, help="사람별 다른 색 적용(1/0)")
    g.add_argument("--energy-colors",        type=str,   default="255,0,0|0,255,0|0,128,255|255,255,0",
                   help='팔레트: "r,g,b|r,g,b|..." 또는 "r,g,b;r,g,b;..."')
    g.add_argument("--energy-max-instances", type=int,   default=4, help="최대 사람 수")
    g.add_argument("--energy-instance-thr",  type=float, default=0.5, help="인스턴스 마스크 임계(0..1)")
    g.add_argument("--energy-min-area",      type=int,   default=2000, help="최소 면적(px) 필터")
    g.add_argument("--energy-sort-by",       type=str,   default="area", choices=["area","x","y"],
                   help="사람 정렬 기준(색 할당 순서 안정화)")
    g.add_argument("--energy-count",         type=int,   default=3, help="사람당 라인 수")
    g.add_argument("--energy-blend",         type=str,   default="screen", choices=["add","screen","normal"],
                   help="에너지 블렌딩 모드")


    # 잔상/멀티 모드/페이싱
    g = p.add_argument_group("Misc / Pipeline")
    g.add_argument("--ema-adaptive", type=int, default=1)
    g.add_argument("--multi-mode", type=str, default="all", choices=["all","nearest","largest","center"])
    g.add_argument("--target-fps", type=str, default=None,
                   help="처리/출력 목표 FPS. 생략하면 --force-fps와 동일")
    g.add_argument("--log-detail", type=int, default=1)

    p.add_argument("--dl-out", type=int, default=0, help="DeckLink로 출력할지 여부 (0/1)")
    p.add_argument("--dl-out-device", type=str, default=None, help="DeckLink 출력 장치명 (예: 'DeckLink Quad (2)')")
    p.add_argument("--dl-out-format", type=str, default=None, help="DeckLink format_code (예: 720p5994, 1080i5994)")
    p.add_argument("--dl-out-fps", type=str, default=None, help="출력 FPS (예: 60000/1001)")
    p.add_argument("--dl-out-pixfmt", type=str, default="bgr24", help="파이썬→ffmpeg stdin raw 픽셀포맷 (bgr24|bgra)")
    p.add_argument("--dl-out-outpix", type=str, default="uyvy422", help="DeckLink 장치로 내보낼 픽셀포맷(uyvy422|v210)")
    p.add_argument("--dl-out-scale", type=str, default=None,         # 예: "1920x1080"
    help="출력 직전에 스케일링 (예: 1920x1080). 지정 시 ffmpeg -vf scale 적용")

    # ---- ripple ----
    p.add_argument("--ripple",              type=int,  default=0, help="1=on, 0=off")
    p.add_argument("--ripple-count",        type=int,  default=3)
    p.add_argument("--ripple-width",        type=int,  default=2)
    p.add_argument("--ripple-spacing",      type=int,  default=28)
    p.add_argument("--ripple-speed",        type=float,default=60.0)
    p.add_argument("--ripple-start",        type=int,  default=8)
    p.add_argument("--ripple-color",        type=str,  default="255,180,80")  # RGB 문자열
    p.add_argument("--ripple-blend",        type=str,  default="screen", choices=["add","screen","normal"])
    p.add_argument("--ripple-on-bg",        type=int,  default=1)             # 1=True, 0=False
    p.add_argument("--ripple-gain", type=float, default=0.0)
    p.add_argument('--ripple-aa', type=int, default=0,
                    help='Enable ripple AA (0/1)')
    p.add_argument('--ripple-aa-downscale', type=float, default=0.5,
                        help='AA work at this scale (0.3~0.7)')
    p.add_argument('--ripple-aa-window', type=int, default=10,
                        help='AA window px (orientation/jiggle)')
    p.add_argument('--ripple-aa-sigma', type=float, default=3.0,
                        help='AA gaussian sigma')
    p.add_argument('--ripple-aa-frame-stride', type=int, default=3,
                        help='Recompute AA every N frames')


    g = p.add_argument_group("Pencil Outline (colored pencil)")
    g.add_argument("--pencil", type=int, default=0, help="색연필 아웃라인 on/off")
    g.add_argument("--pencil-color", type=str, default="255,128,0", help="기본 색상 r,g,b (0..255)")
    g.add_argument("--pencil-color2", type=str, default=None, help="보조 색상 r,g,b (멀티컬러 혼합용)")
    g.add_argument("--pencil-multicolor", type=int, default=0, help="0 단색 / 1 노이즈 기반 색 혼합")
    g.add_argument("--pencil-width", type=int, default=3, help="스트로크 굵기(px)")
    g.add_argument("--pencil-soft", type=int, default=1, help="스트로크 소프트닝(px)")
    g.add_argument("--pencil-gain", type=float, default=1.0, help="에지 강도 스케일")
    g.add_argument("--pencil-opacity", type=float, default=0.8, help="최종 불투명도 0..1")
    g.add_argument("--pencil-thresh", type=float, default=0.08, help="에지 감지 임계 0..1")
    g.add_argument("--pencil-rough", type=float, default=0.7, help="질감 거칠기 0..1")
    g.add_argument("--pencil-jitter", type=float, default=0.8, help="선 떨림/굴곡 0..1")
    g.add_argument("--pencil-dash", type=int, default=0, help="0 실선 / 1 점선")
    g.add_argument("--pencil-dash-len", type=int, default=12, help="점선 길이(px)")
    g.add_argument("--pencil-gap-len", type=int, default=8, help="점선 간격(px)")
    g.add_argument("--pencil-on-bg", type=int, default=1, help="배경 위에만 그림")
    g.add_argument("--pencil-on-fg", type=int, default=1, help="전경 윤곽에도 일부 적용")
    g.add_argument("--pencil-blend", type=str, default="alpha", choices=["alpha","add","screen","overlay"],
                  help="블렌딩 모드")
    g.add_argument("--pencil-palette", default="", choices=["", "rainbow"])
    g.add_argument("--pencil-bands", type=int, default=7)
    g.add_argument("--pencil-repeat", type=int, default=1)
    g.add_argument("--pencil-axis", default="x", choices=["x","y", "around"])

    g = p.add_argument_group("Pencil Hatch (normal/tangent/global)")
    g.add_argument("--pencil-hatch",          type=int,   default=0,     help="0=off, 1=on")
    g.add_argument("--pencil-hatch-mode",     type=str,   default="normal", choices=["global","tangent","normal"])
    g.add_argument("--pencil-hatch-angle",    type=float, default=30.0,  help="global 모드 각도(도)")
    g.add_argument("--pencil-hatch-period",   type=float, default=6.0,   help="줄 간격(px)")
    g.add_argument("--pencil-hatch-width",    type=float, default=2.0,   help="줄 두께(px)")
    g.add_argument("--pencil-hatch-phase",    type=float, default=0.0,   help="0~1")
    g.add_argument("--pencil-hatch-jitter",   type=float, default=0.30,  help="0~1, 결 랜덤성")
    g.add_argument("--pencil-sweep-soft-px", type=int,   default=24,
               help="스윕 경계 소프트 폭(px). 큰 값일수록 경계가 부드럽게 퍼짐")
    g.add_argument("--pencil-sweep-axis",    type=str,   default="y", choices=["x","y"],
                help="스윕 진행 방향: x=좌→우, y=위→아래")
    g.add_argument("--pencil-sweep-ms",      type=int,   default=800,
                help="스윕 애니메이션 한 번 재생 시간(ms). V키 트리거 시 사용")


    
    # --- YOLOv8 Seg (person instance) ---
    g = p.add_argument_group("YOLOv8 Seg")
    g.add_argument("--yolo", type=int, default=0, help="YOLOv8-seg 사용(1/0)")
    g.add_argument("--yolo-weights", type=str, default="yolov8n-seg.pt", help="YOLOv8-seg 가중치 경로")
    g.add_argument("--yolo-img", type=int, default=640, help="YOLO 입력 해상도")
    g.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO confidence")

    # --- Tracking (DeepSORT) ---
    g = p.add_argument_group("Tracking")
    g.add_argument("--use-deepsort", type=int, default=0, help="DeepSORT tracker 사용(1/0)")
    g.add_argument("--ds-iou", type=float, default=0.3, help="DeepSORT IoU threshold")
    g.add_argument("--ds-max-age", type=int, default=30, help="DeepSORT 최대 age")
    g.add_argument("--ds-max-dist", type=float, default=0.2, help="ReID cosine distance threshold")
    g.add_argument("--ds-model", type=str, default=None, help="ReID 가중치 경로(예: osnet_x0_25_msmt17.pt)")
    g.add_argument("--ds-device", type=str, default=None, help="ReID 디바이스 (cuda:0/cpu)")

        # --- Side-based color mapping (per-person) ---
    g = p.add_argument_group("Per-Person Side Mapping")
    g.add_argument("--energy-split", type=str, default="off",
               choices=["off","center","abs","pct"],
               help="인스턴스 없이 X축 기준으로 좌/우 분할 칠하기")
    g.add_argument("--energy-split-x", type=float, default=0.5,
                help="분할 기준 (center: 무시, abs: px, pct: 0..1 비율)")
    g.add_argument("--left-color", type=str, default="255,0,0")
    g.add_argument("--right-color", type=str, default="0,160,255")
    p.add_argument("--energy-split-soft", type=float, default=0.0,
               help="경계 소프트 폭(px). 0이면 하드 컷")


    return p


def args_to_run_kwargs(args) -> dict:
    """
    run_virtualcam(...)에 그대로 넘길 수 있도록 dict로 매핑해주는 헬퍼.
    메인에서 하나하나 키워드로 넘기는 대신 이 함수를 쓰면 깔끔합니다.
    """
    return dict(
        ffmpeg_in=(args.ffmpeg_in or "dshow"),
        ffmpeg_bin=args.ffmpeg_bin,
        decklink_device=args.decklink_device,
        decklink_format=args.decklink_format,
        raw_format=args.raw_format,
        dshow_name=args.dshow_name,
        force_size=args.force_size,
        force_fps=args.force_fps,
        rtbufsize=args.rtbufsize,
        input_timeout_s=args.input_timeout,
        max_input_timeouts=args.max_input_timeouts,
        backbone=args.backbone,
        weights=args.weights,
        preset=args.preset,
        target_long=args.long,
        device=args.device,
        fp16=args.fp16,
        show_fps=bool(args.show_fps),
        no_preview=args.no_preview,
        preview_scale=args.preview_scale,
        show_depth=bool(args.show_depth),
        obs_out=bool(args.obs_out),
        obs_out_sync=bool(args.obs_out_sync),
        spout_out=bool(args.spout_out),
        spout_name=args.spout_name,
        spout_invert=bool(args.spout_invert),
        spout_mode=args.spout_mode,
        spout_bg=bool(args.spout_bg),
        spout_bg_name=args.spout_bg_name,
        spout_bg_mode=args.spout_bg_mode,
        spout_bg_rate=args.spout_bg_rate,
        spout_bg_margin=args.spout_bg_margin,
        person_distance=bool(args.person_distance),
        dist_hfov=args.dist_hfov,
        dist_cam_height=args.dist_cam_height,
        dist_cam_pitch=args.dist_cam_pitch,
        dist_calib=args.dist_calib,
        dist_pose_model=args.dist_pose_model,
        dist_pose_imgsz=args.dist_pose_imgsz,
        use_depth=bool(args.use_depth),
        depth_model_name=args.depth_model_name,
        depth_long=args.depth_long,
        depth_async=bool(args.depth_async),
        depth_fps=args.depth_fps,
        depth_invert=bool(args.depth_invert),
        depth_bokeh=bool(args.depth_bokeh),
        blur_weak=args.blur_weak,
        blur_strong=args.blur_strong,
        alpha_sigma=args.alpha_sigma,
        alpha_unsharp=args.alpha_unsharp,
        alpha_ema=args.alpha_ema,
        alpha_close=args.alpha_close,
        alpha_dilate=args.alpha_dilate,
        energy=bool(args.energy),
        energy_color=args.energy_color,
        energy_width=args.energy_width,
        energy_blur=args.energy_blur,
        energy_gain=args.energy_gain,
        energy_on_bg=bool(args.energy_on_bg),
        energy_edge_sobel=bool(args.energy_edge_sobel),
        energy_tight_attach=bool(args.energy_tight_attach),
        energy_attach_k=args.energy_attach_k,
        energy_thresh_lo=args.energy_thresh_lo,
        energy_thresh_hi=args.energy_thresh_hi,
        energy_temporal_beta=args.energy_temporal_beta,
        energy_pulse=bool(args.energy_pulse),
        energy_pulse_speed=args.energy_pulse_speed,
        energy_pulse_min=args.energy_pulse_min,
        energy_pulse_max=args.energy_pulse_max,
        energy_pulse_shape=args.energy_pulse_shape,
        energy_color_lerp_ms=args.energy_color_lerp_ms,
        energy_color_exp_tau_ms=args.energy_color_exp_tau_ms,
        energy_debug=bool(getattr(args, "energy_debug", 0)),
        energy_y_band_mode=args.energy_y_band_mode,
        energy_y_band_center=args.energy_y_band_center,
        energy_y_band_width=args.energy_y_band_width,
        energy_y_band_sigma=args.energy_y_band_sigma,
        energy_per_person = bool(args.energy_per_person),
        energy_colors = args.energy_colors,
        energy_max_instances = args.energy_max_instances,
        energy_instance_thr = args.energy_instance_thr,
        energy_min_area = args.energy_min_area,
        energy_sort_by = args.energy_sort_by,
        energy_count = args.energy_count,
        energy_blend = args.energy_blend,
        aura=bool(args.aura),
        aura_amp=args.aura_amp,
        aura_freq=args.aura_freq,
        aura_speed=args.aura_speed,
        aura_twirl=args.aura_twirl,
        aura_rgb_phase=args.aura_rgb_phase,
        aura_down=args.aura_down,
        aura_roi=args.aura_roi,
        ema_adaptive=bool(args.ema_adaptive),
        multi_mode=args.multi_mode,
        target_fps=args.target_fps,
        log_detail=args.log_detail,
        dl_out= args.dl_out,
        dl_out_device= args.dl_out_device,
        dl_out_format= args.dl_out_format,
        dl_out_fps= (args.dl_out_fps or args.force_fps),
        dl_out_pixfmt= args.dl_out_pixfmt,
        dl_out_outpix= args.dl_out_outpix,
        dl_out_scale  = args.dl_out_scale,
        ripple_enabled=bool(args.ripple),
        ripple_count=args.ripple_count,
        ripple_width=args.ripple_width,
        ripple_spacing=args.ripple_spacing,
        ripple_speed=args.ripple_speed,
        ripple_start=args.ripple_start,
        ripple_color=args.ripple_color,
        ripple_blend=args.ripple_blend,
        ripple_on_bg=args.ripple_on_bg,
        ripple_gain=args.ripple_gain,
        aa_enable=bool(args.ripple_aa),
        aa_downscale=args.ripple_aa_downscale,
        aa_window_px=args.ripple_aa_window,
        aa_sigma_px=args.ripple_aa_sigma,
        aa_frame_stride=args.ripple_aa_frame_stride,
        pencil_enabled=bool(args.pencil),
        pencil_color=args.pencil_color,
        pencil_color2=args.pencil_color2,
        pencil_multicolor=args.pencil_multicolor,
        pencil_width=args.pencil_width,
        pencil_soft=args.pencil_soft,
        pencil_gain=args.pencil_gain,
        pencil_opacity=args.pencil_opacity,
        pencil_thresh=args.pencil_thresh,
        pencil_rough=args.pencil_rough,
        pencil_jitter=args.pencil_jitter,
        pencil_dash=args.pencil_dash,
        pencil_dash_len=args.pencil_dash_len,
        pencil_gap_len=args.pencil_gap_len,
        pencil_on_bg=bool(args.pencil_on_bg),
        pencil_on_fg=bool(args.pencil_on_fg),
        pencil_blend=args.pencil_blend,
        pencil_palette=args.pencil_palette,
        pencil_bands=args.pencil_bands,
        pencil_repeat=args.pencil_repeat,
        pencil_axis=args.pencil_axis,
        pencil_hatch         = args.pencil_hatch,
        pencil_hatch_mode    = args.pencil_hatch_mode,
        pencil_hatch_angle   = args.pencil_hatch_angle,
        pencil_hatch_period  = args.pencil_hatch_period,
        pencil_hatch_width   = args.pencil_hatch_width,
        pencil_hatch_phase   = args.pencil_hatch_phase,
        pencil_hatch_jitter  = args.pencil_hatch_jitter,
        pencil_sweep_ms     =  args.pencil_sweep_ms,
        pencil_sweep_soft_px = args.pencil_sweep_soft_px,
        pencil_sweep_axis    = args.pencil_sweep_axis,
        yolo=bool(args.yolo),
        yolo_weights=args.yolo_weights,
        yolo_img=args.yolo_img,
        yolo_conf=args.yolo_conf,
        use_deepsort=bool(getattr(args, 'use_deepsort', 0)),
        ds_iou=getattr(args, 'ds_iou', 0.3),
        ds_max_age=getattr(args, 'ds_max_age', 30),
        ds_max_dist=getattr(args, 'ds_max_dist', 0.2),
        ds_model=getattr(args, 'ds_model', None),
        ds_device=getattr(args, 'ds_device', None),
        energy_split = args.energy_split,
        energy_split_x = args.energy_split_x,
        left_color = args.left_color,
        right_color = args.right_color,
        energy_split_soft = args.energy_split_soft,
    )
