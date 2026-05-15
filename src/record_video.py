#!/usr/bin/env python3
"""Record MuJoCo closed-loop tracking videos for the paper.

Uses the canonical benchmark config (dt=0.01, N=20, Q/R matching the MPCExpert
in solver_reluqp.py __main__) so the visuals match the headline numbers.

Run from the repository root, e.g.:
    python3 src/record_video.py --solver c_admm --traj helix \
        --output results/c_admm_helix.mp4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import imageio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver
from mpc_osqp import OSQP_MPC, MPCParams


# ----- references -----------------------------------------------------------

def gen_fig8_ff(c, r, h, per, dur, dt):
    """Figure-8 with feedforward attitude (matches dagger.py and the paper)."""
    N = int(dur / dt)
    t = np.arange(N) * dt
    w = 2 * np.pi / per
    g = 9.81
    ref = np.zeros((12, N))
    ref[0] = c[0] + r * np.sin(w * t)
    ref[1] = c[1] + r * np.sin(2 * w * t) / 2
    ref[2] = h
    ref[3] = r * w * np.cos(w * t)
    ref[4] = r * w * np.cos(2 * w * t)
    ax = -r * w ** 2 * np.sin(w * t)
    ay = -r * w ** 2 * 2 * np.sin(2 * w * t)
    ref[6] = -ay / g
    ref[7] = ax / g
    return ref


def gen_helix(c, r, z0, z1, per, dur, dt):
    """Helix climbing from z0 to z1 with constant XY-radius r."""
    N = int(dur / dt)
    t = np.arange(N) * dt
    w = 2 * np.pi / per
    ref = np.zeros((12, N))
    ref[0] = c[0] + r * np.cos(w * t)
    ref[1] = c[1] + r * np.sin(w * t)
    ref[2] = z0 + (z1 - z0) * t / dur
    ref[3] = -r * w * np.sin(w * t)
    ref[4] = r * w * np.cos(w * t)
    ref[5] = (z1 - z0) / dur
    return ref


# ----- solver factories -----------------------------------------------------

def benchmark_config():
    p = QuadParams()
    dt = 0.01
    N = 20
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt)
    Q_diag = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
    R_diag = np.array([30, 1.5e3, 1.5e3, 1.5e3])
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    gravity = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ u_hover
    INF = 1e10
    x_min = np.array([-INF] * 3 + [-INF] * 3 +
                     [-np.radians(35), -np.radians(35), -INF] + [-INF] * 3)
    x_max = np.array([INF] * 3 + [INF] * 3 +
                     [np.radians(35), np.radians(35), INF] + [INF] * 3)
    return dict(
        p=p, dt=dt, N=N, Ad=Ad, Bd=Bd,
        Q_diag=Q_diag, R_diag=R_diag,
        u_hover=u_hover, gravity=gravity,
        x_min=x_min, x_max=x_max,
    )


def make_solver(name, cfg):
    if name == "c_admm":
        return CADMMSolver(
            cfg["Ad"], cfg["Bd"], cfg["Q_diag"], cfg["R_diag"], cfg["N"],
            cfg["p"].u_min, cfg["p"].u_max, cfg["x_min"], cfg["x_max"],
            cfg["u_hover"], cfg["gravity"],
            rho=3.0, max_iter=50, eps_abs=1e-3,
        )
    if name == "osqp":
        params = MPCParams(N=cfg["N"], dt=cfg["dt"],
                           Q_diag=cfg["Q_diag"], R_diag=cfg["R_diag"],
                           phi_max=np.radians(35), theta_max=np.radians(35))
        return OSQP_MPC(cfg["Ad"], cfg["Bd"],
                        cfg["p"].u_min, cfg["p"].u_max,
                        cfg["u_hover"], params)
    if name == "reluqp":
        from solver_reluqp import ReLUQP_MPC
        return ReLUQP_MPC(
            cfg["Ad"], cfg["Bd"], cfg["Q_diag"], cfg["R_diag"], cfg["N"],
            cfg["p"].u_min, cfg["p"].u_max, cfg["x_min"], cfg["x_max"],
            cfg["u_hover"], cfg["gravity"],
        )
    if name == "policy":
        from policy_inference_py import DistilledPolicy
        return DistilledPolicy(cfg["p"].u_min, cfg["p"].u_max)
    raise ValueError(name)


# ----- recording loop -------------------------------------------------------

_FONT_CACHE = {}


def _font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont
    try:
        f = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def overlay(img, title, step_us=None, total_s=None):
    """Draw the title + (optionally) per-step timing readout."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text((20, 18), title, fill=(255, 255, 255), font=_font(28),
              stroke_width=2, stroke_fill=(0, 0, 0))
    if step_us is not None:
        draw.text((20, 54), f"step:  {step_us:7.1f} us",
                  fill=(255, 255, 255), font=_font(22),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    if total_s is not None:
        draw.text((20, 82), f"total: {total_s:7.3f} s",
                  fill=(255, 255, 255), font=_font(22),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(pil)


def run(solver_name, traj, out_path, duration=10.0, fps=30, overlay_text=None,
        width=1280, height=720):
    cfg = benchmark_config()
    dt = cfg["dt"]
    N = cfg["N"]
    ref_duration = duration + N * dt + 1.0

    if traj == "fig8":
        ref = gen_fig8_ff([0.0, 0.0], 0.5, 1.0, 4.0, ref_duration, dt)
    elif traj == "helix":
        ref = gen_helix([0.0, 0.0], 0.4, 0.5, 1.5, 3.0, ref_duration, dt)
    else:
        raise ValueError(traj)

    mp = str(ROOT / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    env = CrazyflieEnv(model_path=mp, dt_sim=0.002, dt_ctrl=dt)
    solver = make_solver(solver_name, cfg)

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
    frames = []
    solve_times_s = []
    frame_every = max(1, int(round((1.0 / fps) / dt)))

    t_loop = time.perf_counter()
    for k in range(total_steps):
        ref_window = ref[:, k:k + N + 1]
        if ref_window.shape[1] < N + 1:
            pad = np.repeat(ref[:, -1:], N + 1 - ref_window.shape[1], axis=1)
            ref_window = np.concatenate([ref_window, pad], axis=1)

        t_solve = time.perf_counter()
        result = solver.solve(x, ref_window)
        solve_times_s.append(time.perf_counter() - t_solve)

        u_opt = result[0] if isinstance(result, tuple) else result
        x = env.step(u_opt)
        if hasattr(solver, "warm_shift"):
            solver.warm_shift()

        if k % frame_every == 0:
            renderer.update_scene(env.data, cam)
            img = renderer.render()
            if overlay_text is not None:
                img = overlay(img, overlay_text,
                              step_us=solve_times_s[-1] * 1e6,
                              total_s=sum(solve_times_s))
            frames.append(img)

    out = Path(out_path)
    csv_path = out.with_suffix("").as_posix() + "_solvetimes.csv"
    with open(csv_path, "w") as f:
        f.write("step_index,solve_time_s\n")
        for i, t in enumerate(solve_times_s):
            f.write(f"{i},{t:.9f}\n")

    cum = sum(solve_times_s)
    print(f"[{solver_name}/{traj}] {total_steps} steps in "
          f"{time.perf_counter()-t_loop:.1f}s wall; "
          f"solver cumulative = {cum:.3f}s "
          f"(median {1e6*np.median(solve_times_s):.1f} us/step), "
          f"{len(frames)} frames @ {fps} fps -> {out_path}, "
          f"csv -> {csv_path}")
    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solver", required=True,
                   choices=["c_admm", "osqp", "reluqp", "policy"])
    p.add_argument("--traj", default="fig8", choices=["fig8", "helix"])
    p.add_argument("--output", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--overlay", default=None)
    a = p.parse_args()
    run(a.solver, a.traj, a.output, duration=a.duration, fps=a.fps,
        overlay_text=a.overlay)
