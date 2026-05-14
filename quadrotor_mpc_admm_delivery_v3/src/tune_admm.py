"""
tune_admm.py — sweep ADMM internal knobs to find the speed/accuracy floor.
Fixed: Q, R, dt, horizon, dynamics, trajectory.  Tunes: rho, max_iter, eps_abs.
"""
import sys, time, itertools
sys.path.insert(0, '.')
import numpy as np

from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver
from dagger import gen_fig8_ff

DT = 0.01
N = 20
DUR = 10.0
TOTAL = int(DUR / DT)
WARMUP_STEPS = int(2.0 / DT)

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


def run_admm(rho, max_iter, eps_abs):
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=DT)
    solver = CADMMSolver(Ad, Bd, Q_d, R_d, N, p.u_min, p.u_max, xlo, xhi,
                         uh, d, rho=rho, max_iter=max_iter, eps_abs=eps_abs)
    x = env.reset(pos=ref[0:3, 0])
    xs = np.zeros((12, TOTAL + 1)); xs[:, 0] = x
    times, iters = [], []
    for i in range(TOTAL):
        win = np.zeros((12, N + 1))
        for k in range(N + 1):
            win[:, k] = ref[:, min(i + k, ref.shape[1] - 1)]
        u, info = solver.solve(xs[:, i], win)
        xs[:, i + 1] = env.step(u)
        solver.warm_shift()
        times.append(info['solve_time_us'])
        iters.append(info['iterations'])
    err = np.linalg.norm(xs[0:3, WARMUP_STEPS+1:TOTAL+1] - ref[0:3, WARMUP_STEPS+1:TOTAL+1], axis=0)
    return dict(rmse_mm=np.sqrt(np.mean(err**2)) * 1000,
                solve_us=np.median(times), p95=np.percentile(times, 95),
                iters=np.median(iters))


print(f"{'rho':>6s} {'max_it':>6s} {'eps':>10s} | {'RMSE':>8s} {'solve':>8s} {'p95':>7s} {'iters':>6s}")
print("-" * 70)

# Grid sweep
rho_list = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
maxit_list = [50, 100, 200, 500]
eps_list = [1e-3, 1e-4, 1e-5]

results = []
for rho, mi, eps in itertools.product(rho_list, maxit_list, eps_list):
    try:
        r = run_admm(rho, mi, eps)
        r.update(rho=rho, max_iter=mi, eps=eps)
        results.append(r)
        marker = ""
        if r['rmse_mm'] < 1.0: marker += " sub-mm"
        if r['solve_us'] < 30: marker += " <30us"
        if r['solve_us'] < 30 and r['rmse_mm'] < 1.0: marker += " *** TARGET ***"
        print(f"{rho:6.1f} {mi:6d} {eps:10.0e} | {r['rmse_mm']:7.2f}mm {r['solve_us']:6.1f}us {r['p95']:6.1f}us {r['iters']:5.0f}{marker}")
    except Exception as e:
        print(f"{rho:6.1f} {mi:6d} {eps:10.0e} | FAILED: {e}")

# Best by Pareto
print("\nPareto frontier (min RMSE × solve product):")
results.sort(key=lambda r: r['rmse_mm'] * r['solve_us'])
for r in results[:5]:
    print(f"  rho={r['rho']:5.1f} max_it={r['max_iter']:3d} eps={r['eps']:.0e} -> "
          f"RMSE={r['rmse_mm']:.2f}mm solve={r['solve_us']:.1f}us iters={r['iters']:.0f}")
