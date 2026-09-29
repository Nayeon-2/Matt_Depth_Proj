# effects/spark.py
from __future__ import annotations
import math, time
import torch

def make_gaussian_kernel(rad: int, sigma: float=None, device=None, dtype=torch.float32):
    rad = max(1, int(rad))
    if sigma is None: sigma = rad/2.0
    xs = torch.arange(-rad, rad+1, device=device, dtype=dtype)
    g1 = torch.exp(-0.5*(xs/sigma)**2)
    ker = torch.outer(g1, g1)
    ker = ker / (ker.max()+1e-8)
    return ker  # (K,K)

class SparkSystem:
    def __init__(self, H, W, max_count, color_rgb, size_px, speed, jitter, gravity, mean_life, trail, gain, on_bg,
                 device, dtype=torch.float32, seed=1234):
        self.H, self.W = H, W
        self.N = int(max_count)
        self.device = device
        self.dtype = dtype
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.pos = torch.zeros(self.N, 2, device=device, dtype=dtype)  # y,x
        self.vel = torch.zeros(self.N, 2, device=device, dtype=dtype)
        self.life = torch.zeros(self.N, device=device, dtype=dtype)    # 남은 수명(초)
        self.trail = float(trail)
        self.gain = float(gain)
        self.on_bg = bool(on_bg)
        self.color = color_rgb.to(device=device, dtype=dtype).view(3,1,1)
        self.kernel = make_gaussian_kernel(size_px, sigma=None, device=device, dtype=dtype)  # (K,K)
        self.kR = self.kernel.shape[0]//2
        self.rand_cpu = g

    @torch.no_grad()
    def _sample_edge_points(self, edge01, want):
        e = edge01[0,0]
        if e.max() <= 1e-5:
            return None, 0
        s = 4
        eh = torch.nn.functional.avg_pool2d(e.unsqueeze(0).unsqueeze(0), s, s, ceil_mode=True)[0,0]
        flat = eh.flatten()
        k = min(want, flat.numel())
        _, idx = torch.topk(flat, k, largest=True)
        yy = (idx // eh.shape[1]).to(torch.int32)
        xx = (idx %  eh.shape[1]).to(torch.int32)
        oy = torch.randint(0, s, (k,), generator=self.rand_cpu, device='cpu').to(yy.device)
        ox = torch.randint(0, s, (k,), generator=self.rand_cpu, device='cpu').to(xx.device)
        y = (yy*s + oy).clamp(0, self.H-1).to(self.dtype)
        x = (xx*s + ox).clamp(0, self.W-1).to(self.dtype)
        return torch.stack([y,x], dim=1).to(self.device, self.dtype), k

    @torch.no_grad()
    def spawn_from_edge(self, edge01, max_spawn, speed, jitter, mean_life):
        dead = (self.life <= 0.0).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0: return
        want = min(int(max_spawn), dead.numel())
        pts, k = self._sample_edge_points(edge01, want)
        if k == 0 or pts is None: return
        idx = dead[:k]
        theta = torch.rand(k, device=self.device, dtype=self.dtype) * (2*math.pi)
        speed0 = float(speed) * (1.0 - float(jitter) * (torch.rand(k, device=self.device, dtype=self.dtype)*2-1).abs())
        vx = torch.cos(theta) * speed0
        vy = -torch.sin(theta) * speed0
        self.pos[idx,0] = pts[:,0]
        self.pos[idx,1] = pts[:,1]
        self.vel[idx,0] = vy
        self.vel[idx,1] = vx
        self.life[idx] = torch.clamp(torch.randn(k, device=self.device, dtype=self.dtype)*0.2 + float(mean_life), 0.25, 1.5*float(mean_life))

    @torch.no_grad()
    def step(self, dt, gravity):
        if dt <= 0: return
        self.vel[:,0] += float(gravity) * dt
        self.pos += self.vel * dt
        y,x = self.pos[:,0], self.pos[:,1]
        out = (y< -10) | (y> self.H+10) | (x< -10) | (x> self.W+10)
        self.life[out] = 0.0
        self.life -= dt

    @torch.no_grad()
    def render(self, canvasHW):
        canvasHW.mul_(float(self.trail))
        alive = (self.life > 0.0).nonzero(as_tuple=False).flatten()
        if alive.numel() == 0: return canvasHW
        YX = self.pos[alive]
        K = self.kernel
        r = self.kR
        H,W = self.H, self.W
        for i in range(YX.shape[0]):
            y,x = YX[i]
            cy = int(y.item()); cx = int(x.item())
            y0 = max(0, cy-r); y1 = min(H, cy+r+1)
            x0 = max(0, cx-r); x1 = min(W, cx+r+1)
            ky0 = r-(cy-y0); ky1 = ky0 + (y1-y0)
            kx0 = r-(cx-x0); kx1 = kx0 + (x1-x0)
            if y0<y1 and x0<x1:
                canvasHW[0, y0:y1, x0:x1] += K[ky0:ky1, kx0:kx1]
        canvasHW.clamp_(0.0, 1.0)
        return canvasHW

# ---- 편의 함수 ----
def parse_rgb(s: str, device=None, dtype=torch.float32):
    try:
        r, g, b = [float(x) for x in s.split(",")]
    except Exception:
        r, g, b = (255.0, 200.0, 80.0)
    t = torch.tensor([r/255.0, g/255.0, b/255.0], dtype=dtype, device=device)
    return t.clamp(0.0, 1.0)

class SparkState:
    def __init__(self):
        self.sys: SparkSystem | None = None
        self.canvas: torch.Tensor | None = None
        self.last_t: float = time.time()

@torch.no_grad()
def ensure_and_step_spark(
    state: SparkState,
    *,
    H: int, W: int,
    alpha_1chw: torch.Tensor,
    comp_rgb_3hw: torch.Tensor,
    spawn: int, count: int, color: str,
    size_px: int, speed: float, jitter: float, gravity: float,
    lifespan: float, trail: float, gain: float, on_bg: bool, seed: int
) -> torch.Tensor:
    """Spark 생성/업데이트/합성 후 comp 반환"""
    if state.sys is None:
        color_t = parse_rgb(color, device=comp_rgb_3hw.device, dtype=comp_rgb_3hw.dtype)
        state.sys = SparkSystem(H, W, count, color_t, size_px, speed, jitter,
                                gravity, lifespan, trail, gain, on_bg,
                                device=comp_rgb_3hw.device, dtype=comp_rgb_3hw.dtype, seed=seed)
        state.canvas = torch.zeros(1, H, W, device=comp_rgb_3hw.device, dtype=comp_rgb_3hw.dtype)
        state.last_t = time.time()

    now_t = time.time()
    dt_s = max(1e-3, now_t - state.last_t); state.last_t = now_t

    # 간단한 edge: (alpha - erode(k=2))
    k = 2
    pad = 1
    edge01 = (alpha_1chw - (1.0 - torch.nn.functional.max_pool2d(1.0 - alpha_1chw, kernel_size=3, stride=1, padding=pad))).clamp(0,1)

    state.sys.spawn_from_edge(edge01, spawn, speed, jitter, lifespan)
    state.sys.step(dt_s, gravity)
    state.canvas = state.sys.render(state.canvas)

    spark_rgb = state.sys.color * state.canvas  # 3xHxW
    if state.sys.on_bg:
        spark_rgb = spark_rgb * (1.0 - alpha_1chw.squeeze(0))
    return (comp_rgb_3hw + spark_rgb * state.sys.gain).clamp(0.0, 1.0)
