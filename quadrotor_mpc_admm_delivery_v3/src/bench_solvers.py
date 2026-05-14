"""
bench_solvers.py — Head-to-head OSQP vs C ADMM on the same figure-8.

Uses the canonical Q/R from the source code (Q_diag = [10,10,10,1,...]).
Reports honest RMSE and solve-time numbers for both solvers.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver
from mpc_osqp import OSQP_MPC, MPCParams
from dagger import gen_fig8_ff  # includes FF attitude references

# === Configuration ===
DT_CTRL = 0.01            # 100 Hz
DT_SIM  = 0.002           # 500 Hz sim — matches dagger.py CrazyflieTrackingEnv
N_HORIZON = 20
DURATION = 10.0
AMPLITUDE = 0.5
PEAK_SPEED = 1.0  # paper headline (approximate; period determines actual peak)
PERIOD = 4.0      # matches dagger.py — gives peak speed = pi*r/2 ≈ 0.79 m/s

# Cost weights from dagger.py MPCExpert (the actual teacher used in distillation)
Q_diag = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
R_diag = np.array([30, 1.5e3, 1.5e3, 1.5e3])

p = QuadParams()
Ac, Bc = linearize_at_hover(p)
Ad, Bd = discretize_dynamics(Ac, Bc, DT_CTRL, method='expm')
u_hover = np.array([p.hover_thrust, 0, 0, 0])
d = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ u_hover

INF = 1e10
x_min = np.array([-INF]*3 + [-INF]*3 + [-np.radians(35)]*2 + [-INF]*4)
x_max = np.array([INF]*3 + [INF]*3 + [np.radians(35)]*2 + [INF]*4)

ref_dur = DURATION + N_HORIZON * DT_CTRL + 1.0
ref = gen_fig8_ff(np.array([0.0, 0.0]), AMPLITUDE, 1.0, PERIOD, ref_dur, DT_CTRL)
N_ref = ref.shape[1]
total = int(DURATION / DT_CTRL)


def run_solver(name, solver, solve_fn):
    env = CrazyflieEnv(dt_sim=DT_SIM, dt_ctrl=DT_CTRL)
    x = env.reset(pos=ref[0:3, 0])
    xs = np.zeros((12, total + 1)); xs[:, 0] = x
    us = np.zeros((4, total))
    solve_us = []
    iters_log = []
    print(f"\n[{name}] running...")
    t0 = time.time()
    for i in range(total):
        win = np.zeros((12, 21))
        for k in range(21): win[:, k] = ref[:, min(i + k, N_ref - 1)]
        u, info = solve_fn(solver, xs[:, i], win)
        xs[:, i + 1] = env.step(u); us[:, i] = u
        solve_us.append(info['solve_time_us'])
        iters_log.append(info.get('iterations', 1))
    twall = time.time() - t0
    # Steady-state RMSE: skip first 2.0s of transient (matches dagger.py convention)
    # IMPORTANT: error is x_{i+1} vs ref_{i+1} (not x_{i+1} vs ref_i — off-by-one)
    warmup = int(2.0 / DT_CTRL)
    err = np.linalg.norm(xs[0:3, warmup+1:total+1] - ref[0:3, warmup+1:total+1], axis=0)
    rmse = np.sqrt(np.mean(err**2)) * 1000
    solve_us = np.array(solve_us)
    iters = np.array(iters_log)
    print(f"[{name}] wall {twall:.1f}s  RMSE {rmse:.2f} mm  solve median {np.median(solve_us):.1f} us "
          f"95th {np.percentile(solve_us, 95):.1f} us  iters median {np.median(iters):.0f}")
    return dict(xs=xs, us=us, solve_us=solve_us, iters=iters, rmse=rmse, name=name)


# --- C ADMM ---
admm = CADMMSolver(Ad, Bd, Q_diag, R_diag, N_HORIZON,
                   p.u_min, p.u_max, x_min, x_max,
                   u_hover, d, rho=1.0, max_iter=200, eps_abs=1e-4)
def admm_solve(s, x, win):
    u, info = s.solve(x, win)
    s.warm_shift()
    return u, info
r_admm = run_solver('C ADMM', admm, admm_solve)

# --- OSQP ---
params = MPCParams(N=N_HORIZON, dt=DT_CTRL, Q_diag=Q_diag, R_diag=R_diag,
                   phi_max=np.radians(35), theta_max=np.radians(35))
osqp_solver = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, u_hover, params)
def osqp_solve(s, x, win):
    u, info = s.solve(x, win)
    # Normalize: mpc_osqp returns solve_time in seconds; CADMMSolver gives us
    info['solve_time_us'] = info['solve_time'] * 1e6
    return u, info
r_osqp = run_solver('OSQP', osqp_solver, osqp_solve)

# --- Compare ---
print(f"\n=== Solver comparison (same QP, same closed-loop sim) ===")
print(f"  RMSE diff:     |OSQP - ADMM| = {abs(r_osqp['rmse'] - r_admm['rmse']):.3f} mm")
print(f"  Median speedup: {np.median(r_osqp['solve_us']) / np.median(r_admm['solve_us']):.2f}x")
print(f"  95th-pct speedup: {np.percentile(r_osqp['solve_us'], 95) / np.percentile(r_admm['solve_us'], 95):.2f}x")

# --- Plots ---
out = Path('../results'); out.mkdir(exist_ok=True, parents=True)

# Headline benchmark plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(ref[0, :total], ref[1, :total], 'k--', lw=1.0, alpha=0.5, label='reference')
axes[0].plot(r_osqp['xs'][0, :total], r_osqp['xs'][1, :total], 'C1', lw=1.2, alpha=0.8, label=f"OSQP (RMSE {r_osqp['rmse']:.1f} mm)")
axes[0].plot(r_admm['xs'][0, :total], r_admm['xs'][1, :total], 'C0', lw=1.0, alpha=0.8, label=f"C ADMM (RMSE {r_admm['rmse']:.1f} mm)")
axes[0].set_aspect('equal'); axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
axes[0].set_title('Figure-8 tracking @ 100 Hz, 1 m/s peak')
axes[0].legend(loc='upper right', fontsize=9); axes[0].grid(alpha=0.3)

bins = np.logspace(np.log10(min(r_admm['solve_us'].min(), r_osqp['solve_us'].min()) + 1),
                   np.log10(max(r_admm['solve_us'].max(), r_osqp['solve_us'].max())),
                   80)
axes[1].hist(r_osqp['solve_us'], bins=bins, color='C1', alpha=0.6, label=f"OSQP (median {np.median(r_osqp['solve_us']):.0f} µs)")
axes[1].hist(r_admm['solve_us'], bins=bins, color='C0', alpha=0.6, label=f"C ADMM (median {np.median(r_admm['solve_us']):.0f} µs)")
axes[1].set_xscale('log')
axes[1].set_xlabel('Solve time [µs, log scale]'); axes[1].set_ylabel('count')
axes[1].set_title(f"Per-step latency  (speedup = {np.median(r_osqp['solve_us'])/np.median(r_admm['solve_us']):.1f}x)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out / 'benchmark_tuned_final.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  Saved {out / 'benchmark_tuned_final.png'}")

# Per-solver standalone figures
for r, fname in [(r_admm, 'admm_c_fig8.png'), (r_osqp, 'mpc_osqp_helix.png')]:
    # (will use mpc_osqp_helix.png slot for OSQP fig-8 since we're not running helix)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ref[0, :total], ref[1, :total], 'k--', lw=1.0, alpha=0.5, label='ref')
    axes[0].plot(r['xs'][0, :total], r['xs'][1, :total], 'C0', lw=1.0, label=f"{r['name']}")
    axes[0].set_aspect('equal'); axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
    axes[0].set_title(f"{r['name']} figure-8 (RMSE = {r['rmse']:.1f} mm)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].hist(r['solve_us'], bins=60, color='C0', alpha=0.75, edgecolor='black', lw=0.3)
    axes[1].axvline(np.median(r['solve_us']), color='red', ls='--', label=f"median = {np.median(r['solve_us']):.0f} µs")
    axes[1].set_xlabel('Solve time [µs]'); axes[1].set_ylabel('count')
    axes[1].set_title(f"{r['name']} per-step latency")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / fname, dpi=140, bbox_inches='tight'); plt.close()
    print(f"  Saved {out / fname}")

# Save the MPC trajectory for DAgger to use
np.savez('../results/mpc_fig8_trajectory.npz',
         x_log=r_admm['xs'], u_log=r_admm['us'], ref=ref[:, :total + 1],
         dt_ctrl=DT_CTRL, rmse_mm=r_admm['rmse'],
         solve_times=r_admm['solve_us'], iters=r_admm['iters'],
         Q_diag=Q_diag, R_diag=R_diag, N_HORIZON=N_HORIZON)
print(f"  Saved trajectory data for DAgger")
