"""tune_osqp.py -- sweep OSQP knobs to find its speed/accuracy floor.

Overrides solver settings post-setup via osqp.update_settings.

Run from the repository root:
    python3 src/tune_osqp.py
"""
import sys, time, itertools
sys.path.insert(0, '.')
import numpy as np

from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from mpc_osqp import OSQP_MPC, MPCParams
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
ref = gen_fig8_ff(np.array([0.0, 0.0]), 0.5, 1.0, 4.0, DUR + N*DT + 1.0, DT)


def run_osqp(eps_abs, eps_rel, polish, max_iter, scaling):
    osqp_params = MPCParams(N=N, dt=DT, Q_diag=Q_d, R_diag=R_d,
                            phi_max=np.radians(35), theta_max=np.radians(35))
    solver = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, uh, osqp_params)
    # Override OSQP internal settings post-setup
    solver.solver.update_settings(eps_abs=eps_abs, eps_rel=eps_rel,
                                  polish=polish, max_iter=max_iter,
                                  scaling=scaling)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=DT)
    x = env.reset(pos=ref[0:3, 0])
    xs = np.zeros((12, TOTAL + 1)); xs[:, 0] = x
    times, iters, failures = [], [], 0
    for i in range(TOTAL):
        win = np.zeros((12, N + 1))
        for k in range(N + 1):
            win[:, k] = ref[:, min(i + k, ref.shape[1] - 1)]
        u, info = solver.solve(xs[:, i], win)
        xs[:, i + 1] = env.step(u)
        times.append(info['solve_time'] * 1e6)
        iters.append(info.get('iterations', 0))
        if info['status'] not in ('solved', 'solved_inaccurate'):
            failures += 1
    err = np.linalg.norm(xs[0:3, WARMUP_STEPS+1:TOTAL+1] - ref[0:3, WARMUP_STEPS+1:TOTAL+1], axis=0)
    return dict(rmse_mm=np.sqrt(np.mean(err**2)) * 1000,
                solve_us=np.median(times), p95=np.percentile(times, 95),
                iters=np.median(iters), failures=failures)


print(f"{'eps_a':>7s} {'eps_r':>7s} {'poli':>4s} {'max_i':>5s} {'scal':>4s} | "
      f"{'RMSE':>8s} {'solve':>8s} {'p95':>7s} {'iters':>5s} {'fail':>4s}")
print("-" * 80)

# Sweep
configs = [
    # eps_abs, eps_rel, polish, max_iter, scaling
    *[(e, e, False, 500, 10) for e in [1e-2, 1e-3, 1e-4, 1e-5]],
    *[(e, e, True,  500, 10) for e in [1e-3, 1e-4, 1e-5]],
    # Lower max_iter (rely on warm-start)
    *[(1e-3, 1e-3, False, mi, 10) for mi in [25, 50, 100, 250]],
    *[(1e-2, 1e-2, False, mi, 10) for mi in [10, 25, 50]],
    # Scaling sweep
    *[(1e-3, 1e-3, False, 500, s) for s in [0, 5, 25, 50]],
]

results = []
for cfg in configs:
    eps_a, eps_r, polish, max_iter, scaling = cfg
    try:
        r = run_osqp(eps_a, eps_r, polish, max_iter, scaling)
        r.update(eps_abs=eps_a, eps_rel=eps_r, polish=polish, max_iter=max_iter, scaling=scaling)
        results.append(r)
        marker = ""
        if r['rmse_mm'] < 5.0: marker += " <5mm"
        if r['solve_us'] < 100: marker += " <100us"
        if r['failures'] > 0: marker += f" FAILS:{r['failures']}"
        print(f"{eps_a:7.0e} {eps_r:7.0e} {str(polish)[0]:>4s} {max_iter:5d} {scaling:4d} | "
              f"{r['rmse_mm']:7.2f}mm {r['solve_us']:6.1f}us {r['p95']:6.1f}us {r['iters']:5.0f} "
              f"{r['failures']:4d}{marker}")
    except Exception as e:
        print(f"FAILED {cfg}: {e}")

# Pareto: filter out failures, find min RMSE*solve
print("\nPareto frontier (min RMSE x solve product, no failures):")
clean = [r for r in results if r['failures'] == 0]
clean.sort(key=lambda r: r['rmse_mm'] * r['solve_us'])
for r in clean[:5]:
    print(f"  eps={r['eps_abs']:.0e} polish={r['polish']} max_i={r['max_iter']:3d} scal={r['scaling']:2d} "
          f"-> RMSE={r['rmse_mm']:.2f}mm solve={r['solve_us']:.1f}us iters={r['iters']:.0f}")
