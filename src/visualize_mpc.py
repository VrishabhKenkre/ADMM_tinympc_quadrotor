"""
visualize_mpc.py — Live MuJoCo Visualization for All MPC Solvers
=================================================================
Usage:
    python3 src/visualize_mpc.py osqp fig8
    python3 src/visualize_mpc.py admm fig8
    python3 src/visualize_mpc.py admm_c fig8
    python3 src/visualize_mpc.py lqr fig8

    Trajectories: hover, fig8, helix, step

Viewer shows: green sphere = reference, red line = error, blue trail = history
Terminal shows: live error, solve time, iterations
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
from pathlib import Path
from collections import deque
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).parent))
from quad_dynamics import (QuadParams, linearize_at_hover, discretize_dynamics,
                           compute_lqr_gain)
from quad_env import (generate_figure8_reference, generate_hover_reference,
                      generate_helix_reference, generate_step_response)


def get_state(data):
    pos = data.qpos[0:3].copy()
    vel = data.qvel[0:3].copy()
    quat = data.qpos[3:7].copy()
    omega = data.qvel[3:6].copy()
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.concatenate([pos, vel, [roll, pitch, yaw], omega])


def try_add_geom(scn, geom_type, size, pos, mat, rgba):
    try:
        idx = scn.ngeom
        if idx >= scn.maxgeom - 1:
            return
        mujoco.mjv_initGeom(
            scn.geoms[idx], type=geom_type,
            size=np.array(size, dtype=np.float64),
            pos=np.array(pos, dtype=np.float64),
            mat=np.array(mat, dtype=np.float64).flatten(),
            rgba=np.array(rgba, dtype=np.float32))
        scn.ngeom += 1
    except Exception:
        pass


def draw_overlays(viewer, drone_pos, ref_pos, trail, ref_traj, ref_idx):
    scn = None
    for attr in ['user_scn', '_user_scn']:
        if hasattr(viewer, attr):
            scn = getattr(viewer, attr)
            break
    if scn is None:
        return
    
    scn.ngeom = 0
    I3 = np.eye(3)
    
    # Green sphere at reference
    try_add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                 [0.025, 0, 0], ref_pos, I3, [0, 1, 0, 0.8])
    
    # Red error line
    err_vec = ref_pos - drone_pos
    err_len = np.linalg.norm(err_vec)
    if err_len > 0.005:
        mid = (drone_pos + ref_pos) / 2
        z_ax = err_vec / err_len
        x_ax = np.cross(z_ax, [0, 0, 1]) if abs(z_ax[2]) < 0.9 else np.cross(z_ax, [1, 0, 0])
        xn = np.linalg.norm(x_ax)
        if xn > 1e-6:
            x_ax /= xn
            y_ax = np.cross(z_ax, x_ax)
            mat = np.column_stack([x_ax, y_ax, z_ax])
            try_add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE,
                         [0.004, err_len/2, 0], mid, mat, [1, 0.2, 0.2, 0.9])
    
    # Blue trail
    for i, pt in enumerate(trail):
        if scn.ngeom >= scn.maxgeom - 100:
            break
        alpha = 0.1 + 0.4 * (i / max(len(trail), 1))
        try_add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                     [0.005, 0, 0], pt, I3, [0.3, 0.5, 1.0, alpha])
    
    # Green future ref dots
    N_ref = ref_traj.shape[1]
    for k in range(ref_idx, min(N_ref, ref_idx + 100), 3):
        if scn.ngeom >= scn.maxgeom - 5:
            break
        alpha = 0.6 - 0.4 * ((k - ref_idx) / 100)
        try_add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                     [0.008, 0, 0], ref_traj[0:3, k], I3,
                     [0, 1, 0.3, max(0.05, alpha)])


def generate_figure8_with_ff(center, radius, height, period, duration, dt):
    """Figure-8 with velocity AND attitude feedforward."""
    N = int(duration / dt)
    t = np.arange(N) * dt
    omega = 2 * np.pi / period
    g = 9.81
    ref = np.zeros((12, N))
    ref[0] = center[0] + radius * np.sin(omega * t)
    ref[1] = center[1] + radius * np.sin(2 * omega * t) / 2
    ref[2] = height
    ref[3] = radius * omega * np.cos(omega * t)
    ref[4] = radius * omega * np.cos(2 * omega * t)
    ax = -radius * omega**2 * np.sin(omega * t)
    ay = -radius * omega**2 * 2 * np.sin(2 * omega * t)
    ref[6] = -ay / g    # phi = -a_y / g
    ref[7] = ax / g     # theta = a_x / g
    return ref


def build_solver(solver_type, Ad, Bd, p, dt):
    """Build the requested solver."""
    
    Q_diag = np.array([300, 300, 300, 10, 10, 10, 3, 3, 1, 0.1, 0.1, 0.1])
    R_diag = np.array([30, 1.5e3, 1.5e3, 1.5e3])
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    x_hover = np.zeros(12)
    d = (np.eye(12) - Ad) @ x_hover - Bd @ u_hover
    
    INF = 1e10
    x_min = np.array([-INF]*3 + [-INF]*3 + [-np.radians(35), -np.radians(35), -INF] + [-INF]*3)
    x_max = np.array([INF]*3 + [INF]*3 + [np.radians(35), np.radians(35), INF] + [INF]*3)
    
    N = 20
    
    if solver_type == 'lqr':
        K, P = compute_lqr_gain(Ad, Bd, Q, R)
        return {'type': 'lqr', 'K': K, 'u_hover': u_hover, 'p': p}
    
    elif solver_type == 'osqp':
        from mpc_osqp import OSQP_MPC, MPCParams
        mpc_params = MPCParams(N=N, dt=dt)
        mpc_params.Q_diag = Q_diag
        mpc_params.R_diag = R_diag
        mpc_params.phi_max = np.radians(35)
        mpc_params.theta_max = np.radians(35)
        mpc = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, u_hover, mpc_params)
        return {'type': 'osqp', 'mpc': mpc, 'N': N}
    
    elif solver_type == 'admm':
        from solver_admm import ADMMSolver, ADMMParams
        P_terminal = solve_discrete_are(Ad, Bd, Q, R)
        solver = ADMMSolver(Ad, Bd, Q, R, P_terminal, N,
                           p.u_min, p.u_max, x_min, x_max,
                           u_hover, d, ADMMParams(rho=1.0, max_iter=200))
        return {'type': 'admm', 'solver': solver, 'N': N}
    
    elif solver_type == 'admm_c':
        from solver_admm_c import CADMMSolver
        solver = CADMMSolver(Ad, Bd, Q_diag, R_diag, N,
                            p.u_min, p.u_max, x_min, x_max,
                            u_hover, d, rho=1.0, max_iter=200)
        return {'type': 'admm_c', 'solver': solver, 'N': N}
    
    else:
        raise ValueError(f"Unknown solver: {solver_type}. Use: lqr, osqp, admm, admm_c")


def solve_step(solver_dict, state, ref, ref_idx):
    """Run one step of the selected solver."""
    
    stype = solver_dict['type']
    N_ref = ref.shape[1]
    
    if stype == 'lqr':
        K = solver_dict['K']
        u_hover = solver_dict['u_hover']
        p = solver_dict['p']
        x_ref = ref[:, min(ref_idx, N_ref - 1)]
        x_err = state - x_ref
        x_err[8] = ((x_err[8] + np.pi) % (2*np.pi)) - np.pi
        
        tilt = np.sqrt(state[6]**2 + state[7]**2)
        if tilt > np.radians(60) or state[2] < 0.08:
            T = p.thrust_max * 0.95
            kp, kd = 0.005, 0.0003
            u = np.clip(np.array([T, -kp*state[6]-kd*state[9],
                                  -kp*state[7]-kd*state[10], -kd*state[11]]),
                        p.u_min, p.u_max)
        else:
            delta_u = -K @ x_err
            u = np.clip(np.array([p.hover_thrust + delta_u[0],
                                  delta_u[1], delta_u[2], delta_u[3]]),
                        p.u_min, p.u_max)
        return u, {'solve_time': 0, 'iterations': 0, 'solve_time_us': 0}
    
    else:
        N = solver_dict['N']
        ref_window = np.zeros((12, N + 1))
        for k in range(N + 1):
            ref_window[:, k] = ref[:, min(ref_idx + k, N_ref - 1)]
        
        if stype == 'osqp':
            u, info = solver_dict['mpc'].solve(state, ref_window)
            info['solve_time_us'] = info['solve_time'] * 1e6
            return u, info
        
        elif stype == 'admm':
            u, info = solver_dict['solver'].solve(state, ref_window)
            info['solve_time_us'] = info['solve_time'] * 1e6
            solver_dict['solver'].warm_start_shift()
            return u, info
        
        elif stype == 'admm_c':
            u, info = solver_dict['solver'].solve(state, ref_window)
            solver_dict['solver'].warm_shift()
            return u, info


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/visualize_mpc.py <solver> <trajectory> [speed]")
        print("  Solvers:      lqr, osqp, admm, admm_c")
        print("  Trajectories: hover, fig8, helix, step")
        print("  Speed:        1.0=real-time, 0.5=half, 0=max (default: auto)")
        print("\nExamples:")
        print("  python3 src/visualize_mpc.py osqp fig8")
        print("  python3 src/visualize_mpc.py admm fig8 0.3    # slow for Python ADMM")
        print("  python3 src/visualize_mpc.py admm_c hover")
        print("  python3 src/visualize_mpc.py lqr fig8")
        return
    
    solver_type = sys.argv[1]
    traj_mode = sys.argv[2] if len(sys.argv) > 2 else 'fig8'
    
    # Speed factor: 1.0 = real-time, 0.5 = half speed, 0 = as fast as possible
    if len(sys.argv) > 3:
        speed = float(sys.argv[3])
    else:
        # Auto-select: Python ADMM needs slower speed
        speed = 0.3 if solver_type == 'admm' else 1.0
    
    p = QuadParams()
    dt_ctrl = 0.01   # 100 Hz (tuned)
    dt_sim = 0.002
    n_substeps = int(dt_ctrl / dt_sim)
    
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt_ctrl, method='expm')
    
    # Load model
    model_path = str(Path(__file__).parent.parent /
                     "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt_sim
    
    # Generate reference
    duration = 120.0
    if traj_mode == 'hover':
        ref = generate_hover_reference(np.array([0, 0, 1.0]), duration, dt_ctrl)
    elif traj_mode == 'helix':
        ref = generate_helix_reference(
            np.array([0.0, 0.0]), 0.4, 0.5, 1.5, 3.0, duration, dt_ctrl)
    elif traj_mode == 'step':
        ref = generate_step_response(
            np.array([0, 0, 1.0]), np.array([0.5, 0.3, 1.2]),
            2.0, duration, dt_ctrl)
    else:
        ref = generate_figure8_with_ff(
            np.array([0.0, 0.0]), 0.5, 1.0, 4.0, duration, dt_ctrl)
    
    N_ref = ref.shape[1]
    
    # Build solver
    solver_dict = build_solver(solver_type, Ad, Bd, p, dt_ctrl)
    
    # Reset
    data.qpos[0:3] = ref[0:3, 0]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    data.ctrl[0] = p.hover_thrust
    mujoco.mj_forward(model, data)
    
    solver_names = {'lqr': 'LQR', 'osqp': 'OSQP MPC', 'admm': 'Python ADMM',
                    'admm_c': 'C ADMM (gcc -O3)'}
    
    print(f"\n{'='*65}")
    print(f"  {solver_names[solver_type]} — {traj_mode} — {speed:.1f}x speed")
    print(f"{'='*65}")
    print(f"  GREEN sphere = reference    RED line = error")
    print(f"  Push: double-click drone → Ctrl+right-drag")
    print(f"{'='*65}")
    print(f"\n  {'Time':>6s} │ {'Error':>8s} │ {'Thrust':>7s} │ {'Solve':>10s} │ {'Iters':>5s}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*5}")
    
    trail = deque(maxlen=300)
    sim_step = 0
    last_print = 0
    current_u = np.array([p.hover_thrust, 0, 0, 0])
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0, 0, 0.8]
        
        while viewer.is_running():
            step_start = time.time()
            
            if sim_step % n_substeps == 0:
                ctrl_idx = sim_step // n_substeps
                ref_idx = min(ctrl_idx, N_ref - 1)
                state = get_state(data)
                
                pos_err = np.linalg.norm(state[0:3] - ref[0:3, ref_idx])
                trail.append(state[0:3].copy())
                
                current_u, info = solve_step(solver_dict, state, ref, ref_idx)
                
                t_now = data.time
                if t_now - last_print > 0.5:
                    solve_us = info.get('solve_time_us', 0)
                    iters = info.get('iterations', 0)
                    
                    if solve_us > 1000:
                        solve_str = f"{solve_us/1000:7.2f} ms"
                    else:
                        solve_str = f"{solve_us:7.1f} μs"
                    
                    err_color = "\033[91m" if pos_err > 0.02 else "\033[93m" if pos_err > 0.005 else "\033[92m"
                    print(f"  {t_now:6.1f}s │ {err_color}{pos_err*1000:7.1f}mm\033[0m │ "
                          f"{current_u[0]*1000:6.1f}mN │ {solve_str} │ {iters:5d}")
                    last_print = t_now
            
            # Apply control
            data.ctrl[0] = current_u[0]
            data.ctrl[1] = -current_u[1] / 0.0069
            data.ctrl[2] = -current_u[2] / 0.0069
            data.ctrl[3] = -current_u[3] / 0.0036
            
            mujoco.mj_step(model, data)
            sim_step += 1
            
            # Render
            if sim_step % n_substeps == 0:
                ctrl_idx = sim_step // n_substeps
                ref_idx = min(ctrl_idx, N_ref - 1)
                draw_overlays(viewer, data.qpos[0:3].copy(),
                              ref[0:3, ref_idx], trail, ref, ref_idx)
                viewer.sync()
            
            elapsed = time.time() - step_start
            if speed > 0:
                sleep_time = model.opt.timestep / speed - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    print("\nViewer closed.")


if __name__ == '__main__':
    main()
