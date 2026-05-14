"""
train_student_on_nmpc.py — Behavioral cloning of NMPC trajectories.

The DAgger student in `dagger.py` was trained on the *linear* MPC. It inherits
that teacher's obstacle-blindness. To show that the student architecture isn't
the limit, here we train a BC student on NMPC rollouts through obstacle courses.

This is faster than DAgger here because each NMPC episode costs 25-30 seconds
of wall time (IPOPT) — we want to spend our compute on diverse data, not many
DAgger iterations.

Plan:
  - Generate 6 NMPC trajectories on different random obstacle courses (varied seed)
  - Each ~5 s, 25 Hz → 125 (obs, action) per traj → ~750 samples total
  - Plus 2 fig-8 trajectories from the ADMM teacher (~2000 samples) so the
    student also handles vanilla tracking
  - BC train 200 epochs
  - Architecture is the SAME 5,764-param MLP — only the training data differs

Observation expansion
---------------------
The original obs is 20-D (state + tracking error + ref vel + ref angles). To
let the student see obstacles, we extend to 26-D:
    obs[0:20]   : original 20-D obs (state + ref relative)
    obs[20:23]  : local obstacle gradient at current position (3-vector)
    obs[23:26]  : local obstacle value at +/-0.2m offsets in each axis (for free)
Total = 26-D. NMPC has access to the full field; the student gets a local
fingerprint, which is what would be measurable in practice (range sensor /
distance field).
"""
import sys, os, time, json
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from nonlinear_mpc import SE3_NMPC, rotors_to_mujoco, M, G
from quad_env import CrazyflieEnv
from obstacle_course import make_obstacles, obstacle_field_value
from dagger import gen_fig8_ff
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from solver_admm_c import CADMMSolver

device = torch.device('cpu')
torch.manual_seed(0); np.random.seed(0)

OBS_DIM = 26
ACT_DIM = 4
HIDDEN  = 64


class ObstacleAwareNet(nn.Module):
    """Same 64-64-Tanh student, expanded input to 26 dims."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
    def forward(self, x): return self.net(x)


def gradient_field_value(p, obstacles, eps=0.05):
    """Finite-difference gradient of the obstacle scalar field."""
    g = np.zeros(3)
    for i in range(3):
        pp = p.copy(); pp[i] += eps
        pm = p.copy(); pm[i] -= eps
        g[i] = (obstacle_field_value(pp, obstacles) - obstacle_field_value(pm, obstacles)) / (2 * eps)
    return g


def expand_obs(x_mj, ref_state_12, obstacles):
    """20-D base obs + 3-D obstacle gradient + 3-D 'probes' at offsets."""
    p = x_mj[0:3]
    track_err = x_mj[0:3] - ref_state_12[0:3]
    ref_v = ref_state_12[3:6]
    ref_att = ref_state_12[6:8]
    base20 = np.concatenate([x_mj, track_err, ref_v, ref_att])
    grad = gradient_field_value(p, obstacles)
    probes = np.array([
        obstacle_field_value(p + np.array([0.2, 0, 0]), obstacles)
            - obstacle_field_value(p - np.array([0.2, 0, 0]), obstacles),
        obstacle_field_value(p + np.array([0, 0.2, 0]), obstacles)
            - obstacle_field_value(p - np.array([0, 0.2, 0]), obstacles),
        obstacle_field_value(p + np.array([0, 0, 0.2]), obstacles)
            - obstacle_field_value(p - np.array([0, 0, 0.2]), obstacles),
    ])
    return np.concatenate([base20, grad, probes]).astype(np.float32)


# ─── Action normalization (matches CrazyflieEnv) ───
p_ = QuadParams()
U_MIN = p_.u_min; U_MAX = p_.u_max
U_MID = (U_MAX + U_MIN) / 2
U_HALF = (U_MAX - U_MIN) / 2

def ctrl_to_action(u):
    return np.clip((u - U_MID) / U_HALF, -1, 1).astype(np.float32)


def collect_nmpc_episode(seed, duration=5.0, dt=0.04):
    """Run NMPC on a random obstacle course; return (obs26, action4) lists."""
    rng = np.random.RandomState(seed)
    n_obs = rng.randint(5, 10)
    obstacles = make_obstacles(seed=seed, n=n_obs)
    # Vary start/goal slightly
    start = np.array([rng.uniform(-1.7, -1.3), rng.uniform(-1.7, -1.3), 1.0])
    goal  = np.array([rng.uniform( 1.3,  1.7), rng.uniform( 1.3,  1.7), 1.0])

    nmpc = SE3_NMPC(N=15, dt=dt, obstacles=obstacles,
                    q_pos=300, q_vel=10, q_quat=20, q_omega=0.1,
                    r_thrust=1e3, w_obs=800.0)
    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x_mj = env.reset(pos=start)

    N = int(duration / dt)
    ref_p_traj = start[:, None] * (1 - np.arange(N) * dt / duration)[None, :] \
               + goal[:, None] * (np.arange(N) * dt / duration)[None, :]
    ref_v_traj = ((goal - start) / duration)[:, None] * np.ones((1, N))

    obs_list, act_list = [], []
    for i in range(N):
        # NMPC state
        p, v, rpy, w = x_mj[0:3], x_mj[3:6], x_mj[6:9], x_mj[9:12]
        phi, theta, psi = rpy
        cy, sy = np.cos(psi/2), np.sin(psi/2); cp, sp = np.cos(theta/2), np.sin(theta/2)
        cr, sr = np.cos(phi/2), np.sin(phi/2)
        q = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                       cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
        x13 = np.concatenate([p, v, q, w])

        rp_win = np.zeros((3, 16)); rv_win = np.zeros((3, 16))
        for k in range(16):
            j = min(i + k, N - 1)
            rp_win[:, k] = ref_p_traj[:, j]; rv_win[:, k] = ref_v_traj[:, j]

        u_rot, info = nmpc.solve(x13, rp_win, rv_win)
        u_mj = rotors_to_mujoco(u_rot)

        # Build the student's 26-D obs (matches what it'll see at deploy time)
        ref_state_12 = np.zeros(12)
        ref_state_12[0:3] = ref_p_traj[:, i]
        ref_state_12[3:6] = ref_v_traj[:, i]
        # No ref attitude for straight line
        obs26 = expand_obs(x_mj, ref_state_12, obstacles)
        act = ctrl_to_action(u_mj)

        obs_list.append(obs26)
        act_list.append(act)

        x_mj = env.step(u_mj)

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def collect_admm_fig8_episode():
    """ADMM tracking a figure-8 (obstacles are empty here, gradient/probes = 0)."""
    p = QuadParams()
    dt = 0.01
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt)
    Q_d = np.array([300,300,300, 10,10,10, 3,3,1, 0.1,0.1,0.1])
    R_d = np.array([30, 1.5e3, 1.5e3, 1.5e3])
    uh = np.array([p.hover_thrust, 0, 0, 0])
    d = (np.eye(12) - Ad) @ np.zeros(12) - Bd @ uh
    INF = 1e10
    xlo = np.array([-INF]*3+[-INF]*3+[-np.radians(35)]*2+[-INF]*4)
    xhi = np.array([INF]*3+[INF]*3+[np.radians(35)]*2+[INF]*4)
    solver = CADMMSolver(Ad, Bd, Q_d, R_d, 20, p.u_min, p.u_max, xlo, xhi,
                         uh, d, rho=1.0, max_iter=200)
    ref = gen_fig8_ff(np.array([0.0, 0.0]), 0.5, 1.0, 4.0, 12.0, dt)

    env = CrazyflieEnv(dt_sim=0.002, dt_ctrl=dt)
    x = env.reset(pos=ref[0:3, 0])

    obstacles = []  # no obstacles in fig-8 task
    obs_list, act_list = [], []
    N = 1000
    for i in range(N):
        win = np.zeros((12, 21))
        for k in range(21): win[:, k] = ref[:, min(i+k, ref.shape[1]-1)]
        u, info = solver.solve(x, win); solver.warm_shift()
        x = env.step(u)
        ref_state = ref[:, i+1]
        obs26 = expand_obs(x, ref_state, obstacles)
        act = ctrl_to_action(u)
        obs_list.append(obs26); act_list.append(act)

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


if __name__ == '__main__':
    print("="*70)
    print("  Collecting NMPC obstacle-course rollouts...")
    print("="*70)
    obs_all, act_all = [], []
    t0 = time.time()
    for seed in [1, 2, 3, 4]:   # 4 different obstacle courses
        print(f"  Episode {seed}...", flush=True)
        ts = time.time()
        try:
            obs, act = collect_nmpc_episode(seed=seed, duration=5.0, dt=0.04)
            obs_all.append(obs); act_all.append(act)
            print(f"    Got {len(obs)} samples in {time.time()-ts:.1f}s")
        except Exception as e:
            print(f"    FAILED seed {seed}: {e}")
    print(f"  NMPC collection took {time.time()-t0:.0f}s")

    print("\n  Collecting 2 ADMM figure-8 episodes (for vanilla tracking ability)...")
    for _ in range(2):
        obs, act = collect_admm_fig8_episode()
        obs_all.append(obs); act_all.append(act)
        print(f"    Got {len(obs)} samples")

    obs_full = np.concatenate(obs_all, axis=0)
    act_full = np.concatenate(act_all, axis=0)
    print(f"\nDataset: {len(obs_full)} samples ({obs_full.shape[1]}-D obs, {act_full.shape[1]}-D act)")

    # ─── BC training ───
    print("\n" + "="*70)
    print("  Training obstacle-aware student (26->64->64->4)...")
    print("="*70)
    student = ObstacleAwareNet().to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Parameters: {n_params}")
    opt = optim.Adam(student.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.from_numpy(obs_full), torch.from_numpy(act_full))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    losses = []
    for ep in range(200):
        total = 0; n = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = student(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward(); opt.step()
            total += loss.item() * len(xb); n += len(xb)
        if (ep+1) % 50 == 0:
            print(f"  epoch {ep+1}: loss {total/n:.5f}")
        losses.append(total / n)

    out = Path('../results')
    torch.save(student.state_dict(), out / 'policy_BC_NMPC_obstacle.pt')
    print(f"\n  Saved {out / 'policy_BC_NMPC_obstacle.pt'}")
    print(f"  Final BC loss: {losses[-1]:.5f}")
