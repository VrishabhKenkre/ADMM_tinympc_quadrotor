"""bench_tuned_solvers.py -- tuned C ADMM vs tuned OSQP on figure-8.

Tuned C ADMM (rho=3, max_iter=50, eps=1e-3) vs tuned OSQP (eps=1e-5,
max_iter=500, no polish). The original (rho=1, max_iter=200, eps=1e-4)
configuration is kept alongside to reproduce the 3.72 mm / 59 us paper plot.

Run from the repository root:
    python3 src/bench_tuned_solvers.py
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
from dagger import gen_fig8_ff

DT = 0.01
N = 20
DUR = 20.0
TOTAL = int(DUR / DT)
WARMUP = int(2.0 / DT)

p = QuadParams()
Ac, Bc = linearize_at_hover(p)
Ad, Bd = discretize_dynamics(Ac, Bc, DT)
Q_d = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
R_d = np.array([30, 1.5e3, 1.5e3, 1.5e3])
uh = np.array([p.hover_thrust, 0, 0, 0])
d = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ uh
INF = 1e10
xlo = np.array([-INF]*3 + [-INF]*3 + [-np.radians(35)]*2 + [-INF]*4)
xhi = np.array([INF]*3 + [INF]*3 + [np.radians(35)]*2 + [INF]*4)
ref = gen_fig8_ff(np.array([0.0, 0.0]), 0.5, 1.0, 4.0, DUR + N*DT + 1.0, DT)


def run_admm(rho, max_iter, eps):
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=DT)
    solver = CADMMSolver(Ad, Bd, Q_d, R_d, N, p.u_min, p.u_max, xlo, xhi,
                         uh, d, rho=rho, max_iter=max_iter, eps_abs=eps)
    x = env.reset(pos=ref[0:3, 0])
    xs = np.zeros((12, TOTAL + 1)); xs[:, 0] = x
    times, iters = [], []
    for i in range(TOTAL):
        win = np.zeros((12, N + 1))
        for k in range(N + 1): win[:, k] = ref[:, min(i + k, ref.shape[1] - 1)]
        u, info = solver.solve(xs[:, i], win); xs[:, i+1] = env.step(u)
        solver.warm_shift()
        times.append(info['solve_time_us']); iters.append(info['iterations'])
    err = np.linalg.norm(xs[0:3, WARMUP+1:TOTAL+1] - ref[0:3, WARMUP+1:TOTAL+1], axis=0)
    return dict(xs=xs, solve_us=np.array(times), iters=np.array(iters),
                rmse_mm=np.sqrt(np.mean(err**2)) * 1000)


def run_osqp(eps, max_iter):
    osqp_params = MPCParams(N=N, dt=DT, Q_diag=Q_d, R_diag=R_d,
                            phi_max=np.radians(35), theta_max=np.radians(35))
    solver = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, uh, osqp_params)
    solver.solver.update_settings(eps_abs=eps, eps_rel=eps, polish=False,
                                  max_iter=max_iter, scaling=10)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=DT)
    x = env.reset(pos=ref[0:3, 0])
    xs = np.zeros((12, TOTAL + 1)); xs[:, 0] = x
    times, iters = [], []
    for i in range(TOTAL):
        win = np.zeros((12, N + 1))
        for k in range(N + 1): win[:, k] = ref[:, min(i + k, ref.shape[1] - 1)]
        u, info = solver.solve(x, win); xs[:, i+1] = env.step(u); x = xs[:, i+1]
        times.append(info['solve_time'] * 1e6); iters.append(info.get('iterations', 0))
    err = np.linalg.norm(xs[0:3, WARMUP+1:TOTAL+1] - ref[0:3, WARMUP+1:TOTAL+1], axis=0)
    return dict(xs=xs, solve_us=np.array(times), iters=np.array(iters),
                rmse_mm=np.sqrt(np.mean(err**2)) * 1000)


print("Running configs...")
admm_orig  = run_admm(rho=1.0, max_iter=200, eps=1e-4)
admm_tuned = run_admm(rho=3.0, max_iter=50,  eps=1e-3)
admm_fast  = run_admm(rho=0.3, max_iter=50,  eps=1e-3)
osqp_orig  = run_osqp(eps=1e-3, max_iter=500)
osqp_tuned = run_osqp(eps=1e-5, max_iter=500)
print("Done.")

for name, r in [("ADMM original  (rho=1)",  admm_orig),
                ("ADMM tuned     (rho=3)",  admm_tuned),
                ("ADMM speed-opt (rho=0.3)", admm_fast),
                ("OSQP original",            osqp_orig),
                ("OSQP tuned",               osqp_tuned)]:
    print(f"  {name:<28s} RMSE {r['rmse_mm']:6.2f}mm  med {np.median(r['solve_us']):6.1f}us  "
          f"95th {np.percentile(r['solve_us'], 95):6.1f}us  iters {np.median(r['iters']):4.0f}")

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left: trajectories ---
ax = axes[0]
ax.plot(ref[0, :TOTAL], ref[1, :TOTAL], 'k--', lw=0.9, alpha=0.5, label='reference')
ax.plot(osqp_tuned['xs'][0, :TOTAL], osqp_tuned['xs'][1, :TOTAL], 'C1', lw=1.2, alpha=0.85,
        label=f"OSQP tuned ({osqp_tuned['rmse_mm']:.1f} mm)")
ax.plot(admm_tuned['xs'][0, :TOTAL], admm_tuned['xs'][1, :TOTAL], 'C0', lw=1.0, alpha=0.85,
        label=f"C ADMM tuned ({admm_tuned['rmse_mm']:.1f} mm)")
ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title(f'Figure-8 @ 100 Hz, peak ~ 0.79 m/s\n(both solvers at their speed-optimal tuned settings)')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)

# --- Right: latency histogram, log-x ---
ax = axes[1]
all_times = np.concatenate([osqp_orig['solve_us'], osqp_tuned['solve_us'],
                            admm_orig['solve_us'], admm_tuned['solve_us'], admm_fast['solve_us']])
bins = np.logspace(np.log10(max(0.5, all_times.min())), np.log10(all_times.max()), 80)
ax.hist(osqp_orig['solve_us'],  bins=bins, color='#fc8d59', alpha=0.45,
        label=f"OSQP original ({np.median(osqp_orig['solve_us']):.0f} us)")
ax.hist(osqp_tuned['solve_us'], bins=bins, color='#d7301f', alpha=0.7,
        label=f"OSQP tuned ({np.median(osqp_tuned['solve_us']):.0f} us)")
ax.hist(admm_orig['solve_us'],  bins=bins, color='#a6cee3', alpha=0.55,
        label=f"C ADMM original ({np.median(admm_orig['solve_us']):.0f} us)")
ax.hist(admm_tuned['solve_us'], bins=bins, color='#1f78b4', alpha=0.75,
        label=f"C ADMM tuned ({np.median(admm_tuned['solve_us']):.1f} us)")
ax.hist(admm_fast['solve_us'],  bins=bins, color='#33a02c', alpha=0.7,
        label=f"C ADMM speed-opt ({np.median(admm_fast['solve_us']):.1f} us)")
ax.set_xscale('log')
sp_tuned = np.median(osqp_tuned['solve_us']) / np.median(admm_tuned['solve_us'])
sp_fast  = np.median(osqp_tuned['solve_us']) / np.median(admm_fast['solve_us'])
ax.set_title(f'Solver speedup: tuned C ADMM {sp_tuned:.0f}x, speed-opt {sp_fast:.0f}x over tuned OSQP')
ax.set_xlabel('Solve time [µs, log scale]'); ax.set_ylabel('count')
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3, which='both')

plt.tight_layout()
out = Path('../results/benchmark_solver_tuning.png')
plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
print(f"\nSaved {out}")
