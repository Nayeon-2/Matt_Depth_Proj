# config/presets.py
from __future__ import annotations

def apply_preset(args) -> None:
    """
    --preset 값에 따라 기본 파라미터를 살짝 조정.
    이미 CLI에서 직접 값 주면 그 값이 우선이므로,
    '값이 기본값일 때만' 바꿔주는 식으로 구현하는 걸 권장합니다.
    """
    if args.preset == "speed":
        # 빠른 추론(낮은 해상도, 완만한 후처리)
        if args.long == 768:           args.long = 640
        if args.depth_long == 384:     args.depth_long = 320
        if args.alpha_sigma == 0.4:    args.alpha_sigma = 0.25
        if args.alpha_unsharp == 0.35: args.alpha_unsharp = 0.2
        if args.depth_fps == 12.0:     args.depth_fps = 6.0

    elif args.preset == "quality":
        # 화질 우선(해상도↑, 후처리 조금 강화)
        if args.long == 768:           args.long = 896
        if args.depth_long == 384:     args.depth_long = 448
        if args.alpha_sigma == 0.4:    args.alpha_sigma = 0.5
        if args.alpha_unsharp == 0.35: args.alpha_unsharp = 0.4
        if args.depth_fps == 12.0:     args.depth_fps = 8.0

    # balanced는 기본값 유지
