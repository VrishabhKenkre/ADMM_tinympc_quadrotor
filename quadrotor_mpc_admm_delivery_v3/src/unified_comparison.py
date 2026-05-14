"""
unified_comparison.py — All 6 controllers on the same Gaussian obstacle field.

Controllers:
  (1) Linear MPC (C ADMM)              — fast, obstacle-blind
  (2) Linear MPC (OSQP)                — same QP, different solver, obstacle-blind
  (3) SE(3) NMPC                       — slow, obstacle-aware
  (4) DAgger student (on ADMM data)    — fast, obstacle-blind (inherits teacher)
  (5) DAgger+DART student (on ADMM)    — fast, obstacle-blind (inherits teacher)
  (6) BC student on NMPC data          — fast + obstacle-aware (inherits NMPC)

Same obstacle course (seed=42, 8 obstacles), same start, same goal, same dt.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco, M, G
from quad_env import CrazyflieEnv
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver
from mpc_osqp import OSQP_MPC, MPCParams
from obstacle_course import (make_obstacles, obstacle_field_value,
                              make_reference, run_nmpc, run_linear)
from dagger import PolicyNet
from train_student_on_nmpc import ObstacleAwareNet, expand_obs, ctrl_to_action

device = torch.device('cpu')
out = Path('../results'); out.mkdir(exist_ok=True)

# ─── Common setup ───
DT = 0.04
DURATION = 5.0
start = np.array([-1.5, -1.5, 1.0])
goal  = np.array([ 1.5,  1.5, 1.0])
obstacles = make_obstacles(seed=42, n=8)
ref_p, ref_v = make_reference(start, goal, DURATION, DT)
TOTAL = ref_p.shape[1]

print(f"Obstacle course: 8 Gaussian bumps, start{tuple(start)} → goal{tuple(goal)}, dt={DT}")


# ─── ADMM and OSQP (obstacle-blind, reuse functions from obstacle_course) ───
print("\n[1] Linear MPC (C ADMM, obstacle-blind)...")
r_admm = run_linear(obstacles, start, goal, duration=DURATION, dt=DT)
print(f"  done, solve median {np.median(r_admm['times'])*1e6:.0f} µs")

print("\n[2] Linear MPC (OSQP, obstacle-blind)...")
def run_osqp(obstacles, start, goal, duration=5.0, dt=DT):
    p = QuadParams()
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt)
    Q_d = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
    R_d = np.array([30, 1.5e3, 1.5e3, 1.5e3])
    uh = np.array([p.hover_thrust, 0, 0, 0])
    params = MPCParams(N=20, dt=dt, Q_diag=Q_d, R_diag=R_d,
                       phi_max=np.radians(35), theta_max=np.radians(35))
    solver = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, uh, params)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x = env.reset(pos=start)
    ref_p2, ref_v2 = make_reference(start, goal, duration, dt)
    total = ref_p2.shape[1]
    xs = [x.copy()]; times = []; field_vals = []
    for i in range(total):
        win = np.zeros((12, 21))
        for k in range(21):
            j = min(i + k, total - 1)
            win[0:3, k] = ref_p2[:, j]; win[3:6, k] = ref_v2[:, j]
        t0 = time.time()
        u, info = solver.solve(x, win)
        times.append(time.time() - t0)
        x = env.step(u); xs.append(x.copy())
        field_vals.append(obstacle_field_value(x[0:3], obstacles))
    return dict(xs=np.array(xs), times=np.array(times),
                field_vals=np.array(field_vals), ref_p=ref_p2, total=total, dt=dt)
r_osqp = run_osqp(obstacles, start, goal)
print(f"  done, solve median {np.median(r_osqp['times'])*1e6:.0f} µs")

print("\n[3] SE(3) NMPC (obstacle-aware)...")
r_nmpc = run_nmpc(obstacles, start, goal, duration=DURATION, dt=DT)
print(f"  done, solve median {np.median(r_nmpc['times'])*1e3:.0f} ms")


# ─── Student rollout helpers ───
def run_dagger_student(checkpoint_path, obstacles, start, goal, duration=5.0, dt=DT,
                       obs_dim=20):
    """Run a DAgger-trained 20-D student. It's obstacle-blind: doesn't get
       obstacle features. Output action is normalized; convert to physical via mid+half."""
    policy = PolicyNet(obs_dim=obs_dim, act_dim=4, hidden=64).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    policy.eval()
    p_ = QuadParams()
    u_mid = (p_.u_max + p_.u_min) / 2
    u_half = (p_.u_max - p_.u_min) / 2

    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x = env.reset(pos=start)
    ref_p2, ref_v2 = make_reference(start, goal, duration, dt)
    total = ref_p2.shape[1]
    xs = [x.copy()]; times = []; field_vals = []
    for i in range(total):
        j = min(i, total - 1)
        ref_state = np.concatenate([ref_p2[:, j], ref_v2[:, j],
                                    np.zeros(2), np.zeros(4)])  # 12-D ref
        tracking_err = x[0:3] - ref_state[0:3]
        obs20 = np.concatenate([x, tracking_err, ref_state[3:6], ref_state[6:8]]).astype(np.float32)
        t0 = time.time()
        with torch.no_grad():
            act = policy(torch.from_numpy(obs20).unsqueeze(0)).numpy()[0]
        times.append(time.time() - t0)
        u_phys = np.clip(u_mid + u_half * act, p_.u_min, p_.u_max)
        x = env.step(u_phys); xs.append(x.copy())
        field_vals.append(obstacle_field_value(x[0:3], obstacles))
    return dict(xs=np.array(xs), times=np.array(times),
                field_vals=np.array(field_vals), ref_p=ref_p2, total=total, dt=dt)


def run_nmpc_student(checkpoint_path, obstacles, start, goal, duration=5.0, dt=DT):
    """26-D student that SEES the obstacle gradient. Obstacle-aware by training."""
    policy = ObstacleAwareNet().to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))
    policy.eval()
    p_ = QuadParams()
    u_mid = (p_.u_max + p_.u_min) / 2
    u_half = (p_.u_max - p_.u_min) / 2

    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x = env.reset(pos=start)
    ref_p2, ref_v2 = make_reference(start, goal, duration, dt)
    total = ref_p2.shape[1]
    xs = [x.copy()]; times = []; field_vals = []
    for i in range(total):
        j = min(i, total - 1)
        ref_state = np.concatenate([ref_p2[:, j], ref_v2[:, j], np.zeros(6)])
        obs26 = expand_obs(x, ref_state, obstacles)
        t0 = time.time()
        with torch.no_grad():
            act = policy(torch.from_numpy(obs26).unsqueeze(0)).numpy()[0]
        times.append(time.time() - t0)
        u_phys = np.clip(u_mid + u_half * act, p_.u_min, p_.u_max)
        x = env.step(u_phys); xs.append(x.copy())
        field_vals.append(obstacle_field_value(x[0:3], obstacles))
    return dict(xs=np.array(xs), times=np.array(times),
                field_vals=np.array(field_vals), ref_p=ref_p2, total=total, dt=dt)


print("\n[4] DAgger student (trained on ADMM data, obstacle-blind)...")
r_dagger = run_dagger_student('../results/dagger_policy.pt', obstacles, start, goal)
print(f"  done, inference median {np.median(r_dagger['times'])*1e6:.0f} µs")

print("\n[5] DAgger+DART student (trained on ADMM, obstacle-blind)...")
r_dart = run_dagger_student('../results/policy_DAgger_p_DART_adaptive_Sh.pt',
                            obstacles, start, goal)
print(f"  done, inference median {np.median(r_dart['times'])*1e6:.0f} µs")

print("\n[6] BC student trained on NMPC data (obstacle-aware via 26-D input)...")
r_nmpc_student = run_nmpc_student('../results/policy_BC_NMPC_obstacle_v2.pt',
                                  obstacles, start, goal)
print(f"  done, inference median {np.median(r_nmpc_student['times'])*1e6:.0f} µs")

# ─── Metrics ───
def compute_metrics(r, label):
    xs = r['xs']
    ref = r['ref_p']
    path_err = np.linalg.norm(xs[1:, 0:3].T - ref, axis=0)
    warmup = int(1.0 / r['dt'])
    rmse_mm = np.sqrt(np.mean(path_err[warmup:]**2)) * 1000
    path_len = np.sum(np.linalg.norm(np.diff(xs[:, 0:3], axis=0), axis=1))
    max_field = float(np.max(r['field_vals']))
    mean_field = float(np.mean(r['field_vals']))
    final_err_mm = float(np.linalg.norm(xs[-1, 0:3] - goal) * 1000)
    solve_us = float(np.median(r['times']) * 1e6)
    return dict(label=label, rmse_mm=rmse_mm, path_len=path_len,
                max_field=max_field, mean_field=mean_field,
                final_err_mm=final_err_mm, solve_us=solve_us)


configs = [
    ('ADMM (linear, blind)',                r_admm,         'C0'),
    ('OSQP (linear, blind)',                r_osqp,         'C2'),
    ('SE(3) NMPC (obstacle-aware)',         r_nmpc,         'C3'),
    ('DAgger student (ADMM teacher)',       r_dagger,       'C4'),
    ('DAgger+DART student (ADMM teacher)',  r_dart,         'C5'),
    ('BC student (NMPC teacher)',           r_nmpc_student, 'C1'),
]

results = []
print("\n" + "="*88)
print(f"  {'Controller':<38s} | {'RMSE→ref':>8s} | {'Max field':>9s} | {'Mean fld':>8s} | {'Final err':>10s} | {'Solve':>10s}")
print("  " + "-"*100)
for label, r, _ in configs:
    m = compute_metrics(r, label)
    results.append(m)
    if m['solve_us'] > 1e5:
        t_str = f"{m['solve_us']/1000:6.0f} ms"
    elif m['solve_us'] > 100:
        t_str = f"{m['solve_us']:6.0f} µs"
    else:
        t_str = f"{m['solve_us']:6.1f} µs"
    print(f"  {label:<38s} | {m['rmse_mm']:7.1f}mm | {m['max_field']:8.3f} | {m['mean_field']:7.3f} | {m['final_err_mm']:8.1f}mm | {t_str:>10s}")

# ─── Plot ───
fig = plt.figure(figsize=(16, 8))

# Big 3D path comparison
ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
u_sph, v_sph = np.mgrid[0:2*np.pi:18j, 0:np.pi:9j]
for obs in obstacles:
    c = obs['center']; s = obs['sigma'][0]
    ax.plot_wireframe(c[0] + s * np.cos(u_sph) * np.sin(v_sph),
                      c[1] + s * np.sin(u_sph) * np.sin(v_sph),
                      c[2] + s * np.cos(v_sph),
                      color='red', alpha=0.15, lw=0.3)
for label, r, color in configs:
    ax.plot(r['xs'][:, 0], r['xs'][:, 1], r['xs'][:, 2], color=color, lw=1.6, alpha=0.85, label=label)
ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], 'k--', lw=0.8, alpha=0.4)
ax.scatter(*start, color='green', s=60, marker='o', label='start')
ax.scatter(*goal, color='blue', s=60, marker='*', label='goal')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('All 6 controllers on the same obstacle course')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(-0.18, 1.05))

# Obstacle field over time
ax = fig.add_subplot(2, 3, 2)
for label, r, color in configs:
    t = np.arange(len(r['field_vals'])) * r['dt']
    ax.plot(t, r['field_vals'], color=color, lw=1.6, label=label)
ax.set_xlabel('time [s]'); ax.set_ylabel('Obstacle field value')
ax.set_title('Obstacle exposure along path')
ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
ax.axhline(1.0, color='red', ls=':', alpha=0.5)

# Bar: max obstacle field
ax = fig.add_subplot(2, 3, 3)
labels_short = ['ADMM\n(linear,\nblind)', 'OSQP\n(linear,\nblind)', 'NMPC\n(aware)',
                'DAgger\n(ADMM)', 'DAgger+DART\n(ADMM)', 'BC\n(NMPC)']
max_fields = [m['max_field'] for m in results]
colors = [c for _, _, c in configs]
bars = ax.bar(range(6), max_fields, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
for bar, v in zip(bars, max_fields):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(6)); ax.set_xticklabels(labels_short, fontsize=8)
ax.set_ylabel('Max obstacle field encountered')
ax.set_title('Peak obstacle interaction\n(higher = closer to obstacle center)')
ax.grid(alpha=0.3, axis='y')

# Bar: solve time
ax = fig.add_subplot(2, 3, 5)
solve_times = [m['solve_us'] for m in results]
bars = ax.bar(range(6), solve_times, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
ax.set_yscale('log')
for bar, v in zip(bars, solve_times):
    if v < 100: lab = f'{v:.1f} µs'
    elif v < 1000: lab = f'{v:.0f} µs'
    elif v < 1e6: lab = f'{v/1e3:.0f} ms'
    else: lab = f'{v/1e6:.1f} s'
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.4, lab,
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(6)); ax.set_xticklabels(labels_short, fontsize=8)
ax.set_ylabel('Solve / inference latency [µs, log]')
ax.set_title('Computation cost')
ax.grid(alpha=0.3, axis='y', which='both')

# Quadrant: speed vs obstacle awareness
ax = fig.add_subplot(2, 3, 6)
for m, (label, _, color) in zip(results, configs):
    # Y axis: 1 / max_field (higher = better avoidance), capped at 50
    y = 1.0 / max(m['max_field'], 0.02)
    x = m['solve_us']
    short = label.split(' (')[0]
    ax.scatter(x, y, color=color, s=140, alpha=0.85, edgecolor='black', lw=0.7)
    ax.annotate(short, (x, y), xytext=(7, 7), textcoords='offset points', fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Solve / inference latency [µs, log]')
ax.set_ylabel('1 / (peak obstacle field)   ← higher = better avoidance')
ax.set_title('The trade-off: speed vs obstacle awareness\n(top-left corner = ideal)')
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(out / 'unified_comparison.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  Saved {out / 'unified_comparison.png'}")
