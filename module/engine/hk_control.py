# module/engine/color_change.py
from __future__ import annotations
import time, math, threading
from typing import Iterable, Tuple, Callable, Optional

def _clamp255(x: float) -> int:
    return int(round(max(0.0, min(255.0, x))))

def parse_rgb(s: str, default=(0,255,255)) -> Tuple[int,int,int]:
    try:
        r,g,b = [int(float(x)) for x in s.split(",")]
        return (_clamp255(r), _clamp255(g), _clamp255(b))
    except Exception:
        return default

def parse_palette(presets_str: str) -> list[Tuple[int,int,int]]:
    lst = []
    for part in str(presets_str or "").split(";"):
        part = part.strip()
        if not part:
            continue
        lst.append(parse_rgb(part))
    return lst

class EnergyColorState:
    """
    - get_color_rgb_tuple()를 프레임마다 호출하면,
      현재색이 target 색으로 '부드럽게' 수렴해 반환됩니다.
    - 두 가지 스무딩 제공: lerp_ms(선형, smoothstep) / exp_tau_ms(지수). 둘 다 주면 lerp 우선.
    """
    def __init__(
        self,
        start_color: str = "0,0,255",
        presets: Optional[Iterable[Tuple[int,int,int]] | str] = None,
        lerp_ms: int = 250,
        exp_tau_ms: int = 0,
    ):
        if isinstance(presets, str) or presets is None:
            pr = parse_palette(presets or "")
            if not pr:
                pr = [(255,0,0)]
        else:
            pr = [tuple(map(int, p)) for p in presets]
        self.presets: list[Tuple[int,int,int]] = pr

        start = parse_rgb(start_color)
        if start not in self.presets:
            self.presets.insert(0, start)

        self._idx = self.presets.index(start)
        self._cur  = [float(start[0]), float(start[1]), float(start[2])]
        self._tgt  = [float(start[0]), float(start[1]), float(start[2])]
        self._t_set = time.perf_counter()
        self._cur_at_set = self._cur[:]
        self._lerp_ms = max(0, int(lerp_ms))
        self._exp_tau_ms = max(0, int(exp_tau_ms))
        self._lock = threading.Lock()

    # ---------- 외부 조작 ----------
    def next(self) -> Tuple[int,int,int]:
        with self._lock:
            self._idx = (self._idx + 1) % len(self.presets)
            self._set_target(self.presets[self._idx])
            return self.get_target()

    def set_index(self, k: int) -> Tuple[int,int,int]:
        with self._lock:
            if 0 <= k < len(self.presets):
                self._idx = k
                self._set_target(self.presets[self._idx])
            return self.get_target()

    def get_target(self) -> Tuple[int,int,int]:
        return tuple(int(round(self._tgt[i])) for i in range(3))

    # ---------- 프레임마다 호출 ----------
    def get_color_rgb_tuple(self) -> Tuple[int,int,int]:
        with self._lock:
            now = time.perf_counter()
            if self._lerp_ms > 0:
                u = (now - self._t_set) * 1000.0 / self._lerp_ms
                if u >= 1.0:
                    self._cur[:] = self._tgt[:]
                else:
                    u = 0.0 if u <= 0.0 else u
                    u_sm = u*u*(3.0 - 2.0*u)  # smoothstep
                    for i in range(3):
                        self._cur[i] = self._cur_at_set[i] + (self._tgt[i] - self._cur_at_set[i]) * u_sm
            elif self._exp_tau_ms > 0:
                dt_ms = (now - self._t_set) * 1000.0
                w = 1.0 - math.exp(-dt_ms / max(1e-6, self._exp_tau_ms))
                w = max(0.0, min(1.0, w))
                for i in range(3):
                    self._cur[i] += (self._tgt[i] - self._cur[i]) * w
                self._t_set = now
            else:
                self._cur[:] = self._tgt[:]
            return (_clamp255(self._cur[0]), _clamp255(self._cur[1]), _clamp255(self._cur[2]))

    # ---------- 내부 ----------
    def _set_target(self, rgb_tuple: Tuple[int,int,int]) -> None:
        self._tgt[:] = [float(rgb_tuple[0]), float(rgb_tuple[1]), float(rgb_tuple[2])]
        self._cur_at_set = self._cur[:]
        self._t_set = time.perf_counter()

# --- Pencil sweep gate (one-shot 0->1 progress) -----------------------------
class PencilSweepController:
    """
    - 연필 효과 on/off 토글
    - 스윕(0->1) 진행도 관리
    사용 패턴:
      ctrl = PencilSweepCtrl(duration_ms=900, initial_visible=True)

      # 키 입력 시:
      ctrl.toggle_visible()   # 예: 'B' 키
      ctrl.trigger_sweep()    # 예: 'V' 키

      # 매 프레임:
      prog = ctrl.get_progress()
      enabled = ctrl.is_visible()
    """
    def __init__(self, duration_ms: int = 900, initial_visible: bool = True, fade_ms: int = 500):
        import time, threading
        self._dur_ms = max(1, int(duration_ms))
        self._t0 = None              # 스윕 시작 시각(perf_counter)

        self._visible = bool(initial_visible)      # 논리 상태 (on/off)
        self._fade_alpha = 1.0 if initial_visible else 0.0  # 현재 알파(0~1)
        self._fade_target = self._fade_alpha       # 목표 알파(0 or 1)
        self._fade_ms = max(1, int(fade_ms))       # 페이드에 걸리는 시간(ms)
        self._last_fade_t = None                   # 마지막 업데이트 시간

        self._visible = bool(initial_visible)
        self._time = time
        self._lock = threading.Lock()

    # ---- 토글: 연필 효과 전체 on/off ------------------------------------
    def toggle_visible(self) -> bool:
        with self._lock:
            self._visible = not self._visible
            self._fade_target = 1.0 if self._visible else 0.0
            self._last_fade_t = self._time.perf_counter()
            return self._visible

    def set_visible(self, v: bool):
        with self._lock:
            self._visible = bool(v)
            self._fade_target = 1.0 if self._visible else 0.0
            self._last_fade_t = self._time.perf_counter()

    def is_visible(self) -> bool:
        with self._lock:
            return self._visible

    # ---- 스윕 제어 -------------------------------------------------------
    def trigger_sweep(self) -> bool:
        """예: V 키 눌릴 때 호출 → 새 스윕 0부터 시작."""
        with self._lock:
            self._visible = True
            self._fade_target = 1.0           # V 누르면 반드시 켜짐
            self._last_fade_t = self._time.perf_counter()
            self._t0 = self._time.perf_counter()
        print("[PencilSweepCtrl] sweep triggered")
        return True

    def get_progress(self):
        """
        0..1 진행도 or None
        - 스윕 안 하는 중 → None
        - 한 번 끝나면 자동으로 정지 후 1.0 반환 한 번.
        """
        with self._lock:
            t0 = self._t0
            dur = self._dur_ms
        if t0 is None:
            return None
        now = self._time.perf_counter()
        u = (now - t0) * 1000.0 / float(dur)
        if u >= 1.0:
            # 한 번 끝나면 자동 정지
            with self._lock:
                self._t0 = None
            return 1.0
        return max(0.0, float(u))
    
    def get_fade_alpha(self) -> float:
        """
        매 프레임 호출.
        - _fade_target(0 또는 1)을 향해 천천히 이동하는 값 리턴
        """
        now = self._time.perf_counter()
        with self._lock:
            if self._last_fade_t is None:
                self._last_fade_t = now
                return float(self._fade_alpha)

            dt_ms = (now - self._last_fade_t) * 1000.0
            self._last_fade_t = now

            if self._fade_ms <= 0:
                self._fade_alpha = self._fade_target
            else:
                step = dt_ms / float(self._fade_ms)   # 1.0까지 가는 비율
                if self._fade_alpha < self._fade_target:
                    self._fade_alpha = min(self._fade_target,
                                           self._fade_alpha + step)
                elif self._fade_alpha > self._fade_target:
                    self._fade_alpha = max(self._fade_target,
                                           self._fade_alpha - step)

            return float(self._fade_alpha)

class EnergyHotkey(threading.Thread):
    """
    - Windows 콘솔(msvcrt)용 기본 핫키:
      'C' → 다음 프리셋,  '1'~'9','0' → 해당 인덱스 (0은 0번)
    - stop()으로 종료.
    """
    def __init__(self, state: EnergyColorState, 
                 on_change: Optional[Callable[[Tuple[int,int,int]], None]] = None, 
                 on_cycle: Optional[Callable[[], bool]] = None,
                 on_sweep: Optional[Callable[[], None]] = None,
                 on_toggle_pencil: Optional[Callable[[], bool]] = None,):
        super().__init__(daemon=True)
        self.state = state
        self._on_change = on_change
        self._on_cycle  = on_cycle
        self._on_sweep  = on_sweep 
        self._on_toggle_pencil = on_toggle_pencil 
        try:
            import msvcrt
            self._msvcrt = msvcrt
        except Exception:
            self._msvcrt = None
        self.alive = True

    def run(self):
        if self._msvcrt is None:
            print("[EnergyHotkey] msvcrt 없음. 핫키 비활성.")
            return
        print("[EnergyHotkey] 'C'=다음, '1..9/0'=색 선택. target:", self.state.get_target())
        while self.alive:
            try:
                if self._msvcrt.kbhit():
                    ch = self._msvcrt.getwch()
                    if ch in ('v','V'): 
                        print("[HK] key='V' pressed")
                        if self._on_sweep:
                            try: self._on_sweep()
                            except Exception: pass
                        continue  
                    if ch in ('b','B'):
                        print("[HK] key='B' pressed")
                        if self._on_toggle_pencil:
                            try:
                                vis = self._on_toggle_pencil()  # toggle_visible() 예상
                                print(f"[HK] pencil visible = {vis}")
                            except Exception:
                                pass
                        continue
                    if ch in ('c','C'):
                        col = self.state.next()
                    elif ch in ('1','2','3','4','5','6','7','8','9','0'):
                        k = 0 if ch == '0' else (ord(ch) - ord('1'))
                        col = self.state.set_index(k)
                    else:
                        time.sleep(0.01); continue
                    print(f"[Energy] target -> {col}")
                    if self._on_change:
                        try: self._on_change(col)
                        except Exception: pass
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def stop(self): self.alive = False



