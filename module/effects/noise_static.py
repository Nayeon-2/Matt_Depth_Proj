# effects/noise_static.py
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class NoiseStaticConfig:
    enabled: bool = False
    mode: str = "all"          # "all" | "alpha" | "bg"
    gain: float = 0.25         # 전체 노이즈 세기(0~1)
    grain: float = 0.7         # 그레인 거칠기(0~1)
    color_mix: float = 0.35    # 컬러 노이즈 섞임(0=흑백)
    flicker: float = 0.2       # 프레임별 밝기 흔들림
    scanlines: float = 0.15    # 스캔라인 대비
    band_prob: float = 0.12    # 수평 밴드 발생 확률(프레임)
    band_max: int = 2          # 프레임당 최대 밴드 수
    band_strength: int = 10    # 밴드 좌우 변위(px)
    saltpepper: float = 0.002  # 스파클 확률
    vignette: float = 0.10     # 비네트 강도
    scale: float = 0.5         # 노이즈 합성 해상도 비율(0.25~1.0)
    update_every: int = 3      # N프레임마다 새 노이즈 생성(그 외엔 재사용)
    alpha_thr: int = 127       # 알파 임계(전경/배경 선택용)

class NoiseStaticEffect:
    def __init__(self, cfg: NoiseStaticConfig):
        assert cfg.mode in ("all","alpha","bg")
        self.cfg = cfg
        self._rng = np.random.default_rng()
        # 캐시
        self._cached_size = None
        self._scanlines = None
        self._vignette = None
        self._last_noise = None
        self._last_frame_idx = -9999

    # ---- 프리컴퓨트 마스크들 ----
    def _ensure_masks(self, h, w):
        if self._cached_size == (h,w):
            return
        self._cached_size = (h,w)
        # 스캔라인
        y = np.arange(h, dtype=np.float32)[:, None]
        stripes = 0.5*(1.0 + np.sign(np.sin(2*np.pi*y/2.0)))
        stripes = cv2.GaussianBlur(stripes, (0,0), 0.7)
        stripes = (stripes - stripes.min()) / (stripes.max()-stripes.min()+1e-6)
        self._scanlines = stripes[..., None]  # [H,W,1]
        # 비네트
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w-1)/2.0, (h-1)/2.0
        r = np.sqrt(((xx-cx)/(0.5*w))**2 + ((yy-cy)/(0.5*h))**2)
        v = 1.0 - np.clip(r, 0, 1)
        self._vignette = v[..., None]  # [H,W,1]

    # ---- 저해상도 그레인 합성 ----
    def _film_grain_lr(self, h, w):
        s = np.clip(self.cfg.scale, 0.25, 1.0)
        nh, nw = max(1, int(h*s)), max(1, int(w*s))
        # grain 크기 조절: downsample 크기와 블러로 제어
        n = self._rng.standard_normal((nh, nw)).astype(np.float32)
        # 약간 부드럽게
        n = cv2.GaussianBlur(n, (0,0), sigmaX=0.6 + 1.2*self.cfg.grain)
        # 업샘플
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
        # 정규화
        n = (n - n.mean()) / (n.std() + 1e-6)
        return n  # ~N(0,1) [H,W]

    def _color_noise(self, h, w):
        base = self._film_grain_lr(h, w)
        if self.cfg.color_mix <= 1e-3:
            return np.stack([base, base, base], axis=-1)
        c1 = base
        c2 = (base*0.7 + self._rng.standard_normal((h,w)).astype(np.float32)*0.3)
        c3 = (base*0.6 + self._rng.standard_normal((h,w)).astype(np.float32)*0.4)
        out = np.stack([c1,c2,c3], axis=-1)
        g = np.stack([base, base, base], axis=-1)
        return (1.0 - self.cfg.color_mix)*g + self.cfg.color_mix*out

    def _tearing(self, frame_u8):
        if self.cfg.band_prob <= 0:
            return frame_u8
        h, w = frame_u8.shape[:2]
        out = frame_u8
        n_bands = 0
        if self._rng.random() < self.cfg.band_prob:
            n_bands = int(self._rng.integers(1, self.cfg.band_max+1))
        if n_bands == 0:
            return out
        # 밴드만큼 반복(짧고 얇게)
        for _ in range(n_bands):
            y = int(self._rng.integers(0, h))
            band_h = int(self._rng.integers(max(2, h//120), max(4, h//60)))
            shift = int(self._rng.integers(-self.cfg.band_strength, self.cfg.band_strength+1))
            y1, y2 = max(0, y - band_h//2), min(h, y + band_h//2)
            if y2 <= y1: continue
            roi = out[y1:y2]
            M = np.float32([[1,0,shift],[0,1,0]])
            out[y1:y2] = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        return out

    def _gen_noise_frame(self, h, w):
        # flicker 계수
        flick = 1.0 + (self._rng.standard_normal()*0.15)*self.cfg.flicker if self.cfg.flicker>0 else 1.0
        # 컬러 노이즈
        noise = self._color_noise(h, w) * (0.12 + 0.88*self.cfg.gain)
        out = np.clip(noise*0.25*self.cfg.gain, -1, 1)

        # 스캔라인
        if self.cfg.scanlines > 0:
            sl = self._scanlines
            out = np.clip(out*(1.0 - 0.5*self.cfg.scanlines) + sl*(0.5*self.cfg.scanlines), -1, 1)

        # 소금/후추
        if self.cfg.saltpepper > 0:
            sp = self._rng.random((h,w))
            salt = (sp < (self.cfg.saltpepper*0.5))
            pepper = (sp > (1.0 - self.cfg.saltpepper*0.5))
            out[salt] = 1.0
            out[pepper] = -1.0

        # 비네트
        if self.cfg.vignette > 0:
            v = (1.0 - 0.5*self.cfg.vignette + 0.5*self.cfg.vignette*self._vignette)
            out *= v

        return np.clip((out*flick), -1, 1)

    def apply(self, frame_bgr: np.ndarray, alpha_u8: np.ndarray | None = None, frame_index: int = 0) -> np.ndarray:
        if not self.cfg.enabled:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        self._ensure_masks(h, w)

        # N프레임마다 새 노이즈 생성
        if (self._last_noise is None) or (frame_index - self._last_frame_idx >= self.cfg.update_every):
            nf = self._gen_noise_frame(h, w)  # [-1,1], float32 [H,W,3]
            # 프레임 밝기에 더하기 위해 0~255 스케일로 맞춤
            self._last_noise = (np.clip(nf, -1, 1) * 255.0).astype(np.int16)  # int16로 보관(덧셈 안전)
            self._last_frame_idx = frame_index

        out = frame_bgr.astype(np.int16)  # int16 누적 → 나중에 clip
        out = out + self._last_noise

        # 수평 찢김(워핑)은 드물게만: 비용 절감
        if self.cfg.band_prob > 0.0:
            out_u8 = np.clip(out, 0, 255).astype(np.uint8)
            out_u8 = self._tearing(out_u8)
            out = out_u8.astype(np.int16)

        # 선택적 적용 (alpha/bg/all)
        if self.cfg.mode != "all" and alpha_u8 is not None:
            mask = (alpha_u8 > self.cfg.alpha_thr).astype(np.uint8)  # 전경=1
            if self.cfg.mode == "bg":
                mask = 1 - mask  # 배경=1
            mask3 = mask[..., None].astype(np.int16)
            src = frame_bgr.astype(np.int16)
            out = mask3 * out + (1 - mask3) * src

        return np.clip(out, 0, 255).astype(np.uint8)
