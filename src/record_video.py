"""
record_video.py -- closed-loop MuJoCo recorder for the solver demos.

One frame per control step. Optional overlay shows the per-step solve time
and the running solver wall-clock total; the full per-step CSV is dropped
next to the mp4 (consumed by the side-by-side stitch).

Run from the repository root:
    python3 src/record_video.py --solver c_admm --traj fig8 \
        --output videos/c_admm_fig8.mp4 --overlay "C ADMM tuned"
"""
import argparse
import sys
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver
from mpc_osqp import OSQP_MPC, MPCParams


# ---- References ---------------------------------------------------------

def fig8_ref(duration, dt, center=(0.0, 0.0), radius=0.5, height=1.0, period=4.0):
    """Figure-8 with differential-flatness feedforward attitude. The roll/pitch
    columns matter: dropping them takes linear MPC from 3.4 mm to ~18 mm RMSE
    because the controller can't anticipate the centripetal tilt."""
    N = int(duration / dt)
    t = np.arange(N) * dt
    w = 2 * np.pi / period
    g = 9.81
    ref = np.zeros((12, N))
    ref[0] = center[0] + radius * np.sin(w * t)
    ref[1] = center[1] + radius * np.sin(2 * w * t) / 2
    ref[2] = height
    ref[3] = radius * w * np.cos(w * t)
    ref[4] = radius * w * np.cos(2 * w * t)
    ax = -radius * w ** 2 * np.sin(w * t)
    ay = -radius * w ** 2 * 2 * np.sin(2 * w * t)
    ref[6] = -ay / g
    ref[7] = ax / g
    return ref


def helix_ref(duration, dt, center=(0.0, 0.0), radius=0.4, z0=0.5, z1=1.5, period=3.0):
    N = int(duration / dt)
    t = np.arange(N) * dt
    w = 2 * np.pi / period
    ref = np.zeros((12, N))
    ref[0] = center[0] + radius * np.cos(w * t)
    ref[1] = center[1] + radius * np.sin(w * t)
    ref[2] = z0 + (z1 - z0) * t / duration
    ref[3] = -radius * w * np.sin(w * t)
    ref[4] = radius * w * np.cos(w * t)
    ref[5] = (z1 - z0) / duration
    return ref


# ---- Canonical benchmark problem ----------------------------------------

def benchmark_problem():
    """Q/R, linearization, and box constraints matching the benchmark config
    used in solver_reluqp.py __main__ (same Q/R as the MPCExpert in dagger.py)."""
    p = QuadParams()
    dt = 0.01
    N = 20
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt)
    Q = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
    R = np.array([30, 1.5e3, 1.5e3, 1.5e3])
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    gravity = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ u_hover
    # Angle limits at 35 deg, not 30, to stay primal-feasible under these Q/R.
    BIG = 1e10
    x_min = np.array([-BIG]*3 + [-BIG]*3 + [-np.radians(35), -np.radians(35), -BIG] + [-BIG]*3)
    x_max = np.array([ BIG]*3 + [ BIG]*3 + [ np.radians(35),  np.radians(35),  BIG] + [ BIG]*3)
    return dict(p=p, dt=dt, N=N, Ad=Ad, Bd=Bd, Q=Q, R=R,
                u_hover=u_hover, gravity=gravity, x_min=x_min, x_max=x_max)


def build_solver(name, cfg):
    if name == "c_admm":
        return CADMMSolver(cfg["Ad"], cfg["Bd"], cfg["Q"], cfg["R"], cfg["N"],
                           cfg["p"].u_min, cfg["p"].u_max,
                           cfg["x_min"], cfg["x_max"],
                           cfg["u_hover"], cfg["gravity"],
                           rho=3.0, max_iter=50, eps_abs=1e-3)
    if name == "osqp":
        params = MPCParams(N=cfg["N"], dt=cfg["dt"],
                           Q_diag=cfg["Q"], R_diag=cfg["R"],
                           phi_max=np.radians(35), theta_max=np.radians(35))
        return OSQP_MPC(cfg["Ad"], cfg["Bd"], cfg["p"].u_min, cfg["p"].u_max,
                        cfg["u_hover"], params)
    if name == "reluqp":
        from solver_reluqp import ReLUQP_MPC
        return ReLUQP_MPC(cfg["Ad"], cfg["Bd"], cfg["Q"], cfg["R"], cfg["N"],
                          cfg["p"].u_min, cfg["p"].u_max,
                          cfg["x_min"], cfg["x_max"],
                          cfg["u_hover"], cfg["gravity"])
    if name == "policy":
        from policy_inference_py import DistilledPolicy
        return DistilledPolicy(cfg["p"].u_min, cfg["p"].u_max)
    raise ValueError(name)


# ---- Overlay ------------------------------------------------------------

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def _font(size, _cache={}):
    if size not in _cache:
        try:
            _cache[size] = ImageFont.truetype(_FONT_PATH, size)
        except OSError:
            _cache[size] = ImageFont.load_default()
    return _cache[size]


def burn_overlay(img, title, step_us, total_s):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    white, black = (255, 255, 255), (0, 0, 0)
    draw.text((20, 18), title, fill=white, font=_font(28),
              stroke_width=2, stroke_fill=black)
    draw.text((20, 54), f"step:  {step_us:7.1f} us", fill=white, font=_font(22),
              stroke_width=2, stroke_fill=black)
    draw.text((20, 82), f"total: {total_s:7.3f} s", fill=white, font=_font(22),
              stroke_width=2, stroke_fill=black)
    return np.array(pil)


# ---- Closed-loop recording ----------------------------------------------

def run(solver_name, traj, out_path, duration=10.0, fps=30, overlay_text=None,
        width=1280, height=720):
    cfg = benchmark_problem()
    dt, N = cfg["dt"], cfg["N"]
    # Pad the ref a bit past the simulated duration so the lookahead window
    # never runs off the end.
    ref_duration = duration + N * dt + 1.0

    if traj == "fig8":
        ref = fig8_ref(ref_duration, dt)
    elif traj == "helix":
        ref = helix_ref(ref_duration, dt)
    else:
        raise ValueError(traj)

    scene = str(ROOT / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    env = CrazyflieEnv(model_path=scene, dt_sim=0.002, dt_ctrl=dt)
    solver = build_solver(solver_name, cfg)

    # Menagerie's stock offscreen framebuffer is 640x480; bump it before the renderer.
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 1.8
    cam.azimuth = 135
    cam.elevation = -45
    cam.lookat[:] = [0.0, 0.0, 1.0]

    x = env.reset(pos=ref[0:3, 0])
    total_steps = int(duration / dt)
    frame_every = max(1, int(round((1.0 / fps) / dt)))
    frames = []
    solve_times = []

    t_loop = time.perf_counter()
    for k in range(total_steps):
        ref_window = ref[:, k:k + N + 1]

        t0 = time.perf_counter()
        result = solver.solve(x, ref_window)
        solve_times.append(time.perf_counter() - t0)

        u = result[0] if isinstance(result, tuple) else result
        x = env.step(u)
        if hasattr(solver, "warm_shift"):
            solver.warm_shift()

        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            img = renderer.render()
            if overlay_text:
                img = burn_overlay(img, overlay_text,
                                   step_us=solve_times[-1] * 1e6,
                                   total_s=sum(solve_times))
            frames.append(img)

    out = Path(out_path)
    csv_path = out.with_suffix("").as_posix() + "_solvetimes.csv"
    with open(csv_path, "w") as f:
        f.write("step_index,solve_time_s\n")
        for i, t in enumerate(solve_times):
            f.write(f"{i},{t:.9f}\n")

    print(f"[{solver_name}/{traj}] {total_steps} steps in {time.perf_counter()-t_loop:.1f}s wall; "
          f"solver cumulative {sum(solve_times):.3f}s "
          f"(median {1e6 * np.median(solve_times):.1f} us/step), "
          f"{len(frames)} frames @ {fps} fps -> {out_path}, csv -> {csv_path}")
    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True,
                    choices=["c_admm", "osqp", "reluqp", "policy"])
    ap.add_argument("--traj", default="fig8", choices=["fig8", "helix"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--overlay", default=None)
    args = ap.parse_args()
    run(args.solver, args.traj, args.output,
        duration=args.duration, fps=args.fps, overlay_text=args.overlay)
