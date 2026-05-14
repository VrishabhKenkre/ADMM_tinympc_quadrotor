"""
gen_remaining_plots.py — Generate the three remaining missing PNGs.

  mpc_osqp_step.png    — OSQP step-response in xy
  mpc_osqp_helix.png   — OSQP helix tracking (3D-projected)
  cpp_deployment.png   — End-to-end inference latency comparison
                         (MPC solve, PyTorch CPU forward, compiled C forward)
"""
import sys, time, ctypes, subprocess
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from quad_env import CrazyflieEnv, generate_step_response, generate_helix_reference
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from mpc_osqp import OSQP_MPC, MPCParams
from dagger import PolicyNet

out = Path('../results'); out.mkdir(exist_ok=True, parents=True)

# === Common MPC setup matching paper ===
DT_CTRL = 0.01
N_HORIZON = 20
Q_diag = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
R_diag = np.array([30, 1.5e3, 1.5e3, 1.5e3])

p = QuadParams()
Ac, Bc = linearize_at_hover(p)
Ad, Bd = discretize_dynamics(Ac, Bc, DT_CTRL, method='expm')
u_hover = np.array([p.hover_thrust, 0, 0, 0])

# ============================================================
# 1. mpc_osqp_step.png — Step response
# ============================================================
print("Generating mpc_osqp_step.png ...")
osqp_params = MPCParams(N=N_HORIZON, dt=DT_CTRL, Q_diag=Q_diag, R_diag=R_diag,
                        phi_max=np.radians(60), theta_max=np.radians(60))
osqp = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, u_hover, osqp_params)

duration = 6.0
total = int(duration / DT_CTRL)
start_pos = np.array([0.0, 0.0, 1.0])
end_pos   = np.array([0.5, 0.3, 1.2])
ref = generate_step_response(start_pos, end_pos, step_time=1.0, duration=duration + 0.5, dt=DT_CTRL)

env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=DT_CTRL)
x = env.reset(pos=start_pos)
xs = np.zeros((12, total + 1)); xs[:, 0] = x
times = []
for i in range(total):
    win = np.zeros((12, N_HORIZON + 1))
    for k in range(N_HORIZON + 1): win[:, k] = ref[:, min(i + k, ref.shape[1] - 1)]
    u, info = osqp.solve(x, win)
    x = env.step(u); xs[:, i + 1] = x
    times.append(info['solve_time'] * 1e6)

# Find when step actually happens (1.0s) and 5% settling
settling_idx = None
final_pos = end_pos.copy()
for i in range(int(1.0/DT_CTRL), total):
    err = np.linalg.norm(xs[0:3, i] - final_pos)
    if err < 0.05 * np.linalg.norm(end_pos - start_pos):
        if settling_idx is None: settling_idx = i
    else:
        settling_idx = None
settling_time = (settling_idx * DT_CTRL - 1.0) if settling_idx else None

t = np.arange(total + 1) * DT_CTRL
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
labels = ['x [m]', 'y [m]', 'z [m]']
colors = ['C0', 'C1', 'C2']
for k, (lbl, c) in enumerate(zip(labels, colors)):
    axes[k].plot(t, ref[k, :total + 1], 'k--', lw=1, alpha=0.6, label='reference')
    axes[k].plot(t, xs[k, :], c, lw=1.2, label=f'OSQP MPC')
    axes[k].set_ylabel(lbl); axes[k].grid(alpha=0.3); axes[k].legend(loc='lower right', fontsize=9)
axes[-1].set_xlabel('time [s]')
ttl = f'Step response: ({start_pos[0]:.1f},{start_pos[1]:.1f},{start_pos[2]:.1f}) → ({end_pos[0]:.1f},{end_pos[1]:.1f},{end_pos[2]:.1f}) m'
if settling_time: ttl += f', 5% settling = {settling_time:.2f}s'
axes[0].set_title(ttl)
plt.tight_layout(); plt.savefig(out / 'mpc_osqp_step.png', dpi=140, bbox_inches='tight'); plt.close()
print(f"  Saved {out / 'mpc_osqp_step.png'}")

# ============================================================
# 2. mpc_osqp_helix.png — Helix tracking
# ============================================================
print("Generating mpc_osqp_helix.png ...")
osqp = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, u_hover, osqp_params)
duration = 10.0
total = int(duration / DT_CTRL)
ref = generate_helix_reference(np.array([0.0, 0.0]), radius=0.4,
                                z_start=0.5, z_end=1.5,
                                period=3.0, duration=duration + 0.5, dt=DT_CTRL)
env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=DT_CTRL)
x = env.reset(pos=ref[0:3, 0])
xs = np.zeros((12, total + 1)); xs[:, 0] = x
times = []
for i in range(total):
    win = np.zeros((12, N_HORIZON + 1))
    for k in range(N_HORIZON + 1): win[:, k] = ref[:, min(i + k, ref.shape[1] - 1)]
    u, info = osqp.solve(x, win)
    x = env.step(u); xs[:, i + 1] = x
    times.append(info['solve_time'] * 1e6)
warmup = int(2.0 / DT_CTRL)
err_full = np.linalg.norm(xs[0:3, 1:total+1] - ref[0:3, 1:total+1], axis=0)  # length = total
err = err_full[warmup:]  # for RMSE
rmse = np.sqrt(np.mean(err**2)) * 1000

fig = plt.figure(figsize=(11, 5))
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.plot(ref[0, :total], ref[1, :total], ref[2, :total], 'k--', lw=0.8, alpha=0.5, label='reference')
ax3d.plot(xs[0, :total], xs[1, :total], xs[2, :total], 'C1', lw=1.0, label='OSQP MPC')
ax3d.set_xlabel('x [m]'); ax3d.set_ylabel('y [m]'); ax3d.set_zlabel('z [m]')
ax3d.set_title(f'Helix tracking (RMSE = {rmse:.1f} mm)')
ax3d.legend(fontsize=9)

ax2 = fig.add_subplot(1, 2, 2)
t = np.arange(total) * DT_CTRL
ax2.plot(t, err_full * 1000, 'C1', lw=1.0)
ax2.axhline(rmse, color='red', ls='--', label=f'SS-RMSE = {rmse:.1f} mm (skip first 2s)')
ax2.axvline(2.0, color='gray', ls=':', alpha=0.5)
ax2.set_xlabel('time [s]'); ax2.set_ylabel('|position error| [mm]')
ax2.set_title('Tracking error magnitude')
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out / 'mpc_osqp_helix.png', dpi=140, bbox_inches='tight'); plt.close()
print(f"  Saved {out / 'mpc_osqp_helix.png'}")

# ============================================================
# 3. cpp_deployment.png — End-to-end inference comparison
# ============================================================
print("Generating cpp_deployment.png ...")

# Load PyTorch student
policy = PolicyNet(obs_dim=20, act_dim=4, hidden=64)
policy.load_state_dict(torch.load('../results/dagger_policy.pt', map_location='cpu', weights_only=False))
policy.eval()

# Random observation batch for benchmarking
np.random.seed(0)
N_runs = 5000
obs_batch = np.random.randn(N_runs, 20).astype(np.float32)
obs_torch = torch.from_numpy(obs_batch)

# Measure PyTorch CPU (one forward at a time, as in real deployment)
torch_times = []
with torch.no_grad():
    # warmup
    for _ in range(100): policy(obs_torch[0:1])
    for i in range(N_runs):
        t0 = time.perf_counter()
        _ = policy(obs_torch[i:i+1])
        torch_times.append((time.perf_counter() - t0) * 1e6)
torch_times = np.array(torch_times)

# Measure compiled C
lib = ctypes.CDLL('./libpolicy_inference.so')
lib.policy_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.policy_forward.restype  = None
obs_c = (ctypes.c_float * 20)()
act_c = (ctypes.c_float * 4)()

c_times = []
# warmup
for _ in range(1000): lib.policy_forward(obs_c, act_c)
for i in range(N_runs):
    for j in range(20): obs_c[j] = float(obs_batch[i, j])
    t0 = time.perf_counter()
    lib.policy_forward(obs_c, act_c)
    c_times.append((time.perf_counter() - t0) * 1e6)
c_times = np.array(c_times)

# MPC solve times from saved trajectory data
data = np.load('../results/mpc_fig8_trajectory.npz')
mpc_times = data['solve_times']

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Histogram comparison
bins = np.logspace(-0.5, 4, 100)
axes[0].hist(c_times, bins=bins, color='C2', alpha=0.7, label=f'C compiled (median {np.median(c_times):.2f} µs)')
axes[0].hist(torch_times, bins=bins, color='C3', alpha=0.6, label=f'PyTorch CPU (median {np.median(torch_times):.0f} µs)')
axes[0].hist(mpc_times, bins=bins, color='C0', alpha=0.5, label=f'C ADMM MPC (median {np.median(mpc_times):.0f} µs)')
axes[0].set_xscale('log')
axes[0].set_xlabel('Inference / solve time [µs, log scale]'); axes[0].set_ylabel('count')
axes[0].set_title('End-to-end controller latency')
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Bar chart of medians
labels = ['MPC\n(C ADMM)', 'Student\n(PyTorch CPU)', 'Student\n(compiled C)']
medians = [np.median(mpc_times), np.median(torch_times), np.median(c_times)]
colors_b = ['C0', 'C3', 'C2']
bars = axes[1].bar(labels, medians, color=colors_b, alpha=0.85)
axes[1].set_yscale('log')
axes[1].set_ylabel('Median latency [µs, log]')
axes[1].set_title(f'Speedup: MPC → C student = {medians[0]/medians[2]:.0f}x')
for bar, m in zip(bars, medians):
    h = bar.get_height()
    if m < 1: txt = f'{m:.2f} µs'
    elif m < 10: txt = f'{m:.1f} µs'
    else: txt = f'{m:.0f} µs'
    axes[1].text(bar.get_x() + bar.get_width()/2, h * 1.15, txt, ha='center', fontsize=10, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(out / 'cpp_deployment.png', dpi=140, bbox_inches='tight'); plt.close()
print(f"  Saved {out / 'cpp_deployment.png'}")

print(f"\n=== Final Latency Summary ===")
print(f"  MPC (C ADMM):       median {np.median(mpc_times):.0f} µs   95th {np.percentile(mpc_times, 95):.0f} µs")
print(f"  Student (PyTorch):  median {np.median(torch_times):.1f} µs   95th {np.percentile(torch_times, 95):.1f} µs")
print(f"  Student (C):        median {np.median(c_times):.2f} µs   95th {np.percentile(c_times, 95):.2f} µs")
print(f"  MPC -> C student speedup:    {np.median(mpc_times) / np.median(c_times):.0f}x")
