"""
plot_admm_c.py — Run C ADMM and generate plots
=================================================
Runs the C ADMM solver in closed loop, then generates the same
style of plots as mpc_osqp.py for direct comparison.

Usage:
    python3 src/plot_admm_c.py fig8 10.0
    python3 src/plot_admm_c.py hover 5.0
    python3 src/plot_admm_c.py helix 10.0
    python3 src/plot_admm_c.py step 5.0
    python3 src/plot_admm_c.py all 10.0     # run all trajectories

Author: Vrishabh Kenkre (CMU MS MechE)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solver_admm_c import CADMMSolver, NX, NU, NHORIZON
from quad_env import (CrazyflieEnv, generate_figure8_reference,
                      generate_hover_reference, generate_helix_reference,
                      generate_step_response)
from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
from scipy.linalg import solve_discrete_are


def run_and_collect(mode='fig8', duration=10.0):
    """Run C ADMM sim and collect all data for plotting."""
    
    p = QuadParams()
    dt = 0.02
    N = NHORIZON
    
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt, method='expm')
    
    Q_diag = np.array([10, 10, 10, 1, 1, 1, 5, 5, 1, 0.1, 0.1, 0.1])
    R_diag = np.array([100, 1e4, 1e4, 1e4])
    
    x_hover = np.zeros(12)
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    d = (np.eye(12) - Ad) @ x_hover - Bd @ u_hover
    
    INF = 1e10
    x_min = np.array([-INF]*3 + [-INF]*3 + [-np.radians(30), -np.radians(30), -INF] + [-INF]*3)
    x_max = np.array([INF]*3 + [INF]*3 + [np.radians(30), np.radians(30), INF] + [INF]*3)
    
    model_path = str(Path(__file__).parent.parent /
                     "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    env = CrazyflieEnv(model_path=model_path, dt_sim=0.002, dt_ctrl=dt)
    
    total_steps = int(duration / dt)
    ref_duration = duration + N * dt + 1.0
    
    if mode == 'hover':
        ref = generate_hover_reference(np.array([0, 0, 1.0]), ref_duration, dt)
        title = "Hover Stabilization"
    elif mode == 'fig8':
        ref = generate_figure8_reference(
            np.array([0.0, 0.0]), 0.5, 1.0, 4.0, ref_duration, dt)
        title = "Figure-8 Tracking"
    elif mode == 'helix':
        ref = generate_helix_reference(
            np.array([0.0, 0.0]), 0.4, 0.5, 1.5, 3.0, ref_duration, dt)
        title = "Helix Tracking"
    elif mode == 'step':
        ref = generate_step_response(
            np.array([0, 0, 1.0]), np.array([0.5, 0.3, 1.2]),
            1.0, ref_duration, dt)
        title = "Step Response"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    N_ref = ref.shape[1]
    
    solver = CADMMSolver(Ad, Bd, Q_diag, R_diag, N,
                         p.u_min, p.u_max, x_min, x_max,
                         u_hover, d, rho=1.0, max_iter=50, eps_abs=1e-4)
    
    x = env.reset(pos=ref[0:3, 0])
    x_log = np.zeros((12, total_steps + 1))
    u_log = np.zeros((4, total_steps))
    solve_times_us = np.zeros(total_steps)
    iterations = np.zeros(total_steps, dtype=int)
    x_log[:, 0] = x
    
    print(f"  Running C ADMM: {title} ({total_steps} steps)...")
    
    for i in range(total_steps):
        ref_window = np.zeros((12, N + 1))
        for k in range(N + 1):
            ref_window[:, k] = ref[:, min(i + k, N_ref - 1)]
        
        u_opt, info = solver.solve(x_log[:, i], ref_window)
        x_log[:, i + 1] = env.step(u_opt)
        u_log[:, i] = u_opt
        solve_times_us[i] = info['solve_time_us']
        iterations[i] = info['iterations']
        solver.warm_shift()
    
    tracking_err = np.linalg.norm(
        x_log[0:3, :total_steps] - ref[0:3, :total_steps], axis=0)
    
    return {
        'x_log': x_log, 'u_log': u_log, 'ref': ref,
        'tracking_err': tracking_err,
        'solve_times_us': solve_times_us,
        'iterations': iterations,
        'title': title, 'mode': mode, 'dt': dt,
        'rmse': np.sqrt(np.mean(tracking_err**2)),
        'max_err': np.max(tracking_err),
        'mean_solve_us': np.mean(solve_times_us),
        'median_solve_us': np.median(solve_times_us),
        'mean_iters': np.mean(iterations),
    }


def plot_single(data, save_path=None):
    """Generate plots for a single trajectory mode."""
    
    x_log = data['x_log']
    u_log = data['u_log']
    ref = data['ref']
    dt = data['dt']
    
    N_sim = x_log.shape[1]
    N_ref = min(ref.shape[1], N_sim)
    t = np.arange(N_sim) * dt
    t_ref = np.arange(N_ref) * dt
    t_ctrl = np.arange(u_log.shape[1]) * dt
    t_err = np.arange(len(data['tracking_err'])) * dt
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"C ADMM Solver — {data['title']}\n"
                 f"Hand-rolled ADMM, gcc -O3, N=20, 50 Hz",
                 fontsize=13, fontweight='bold')
    
    # Row 1: Position tracking
    labels = ['x', 'y', 'z']
    colors = ['b', 'r', 'g']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(t, x_log[i, :], f'{colors[i]}-', lw=1.5, label='Actual')
        ax.plot(t_ref, ref[i, :N_ref], 'k--', lw=1, label='Reference', alpha=0.7)
        ax.set_title(f'{labels[i]} position')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('[m]')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Row 2: Error, controls, angles
    ax = axes[1, 0]
    ax.plot(t_err, data['tracking_err'] * 100, 'b-', lw=1.5)
    ax.axhline(data['rmse'] * 100, color='r', ls='--',
               label=f"RMSE={data['rmse']*100:.1f}cm")
    ax.set_title('3D Tracking Error')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('[cm]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t_ctrl, u_log[0, :] * 1000, 'k-', lw=1, label='Thrust')
    ax.axhline(0.027 * 9.81 * 1000, color='r', ls='--', alpha=0.5, label='mg')
    ax.set_title('Thrust'); ax.set_xlabel('Time [s]')
    ax.set_ylabel('[mN]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(t, x_log[6, :] * 180 / np.pi, 'b-', lw=1, label='φ (roll)')
    ax.plot(t, x_log[7, :] * 180 / np.pi, 'r-', lw=1, label='θ (pitch)')
    ax.axhline(30, color='gray', ls=':', alpha=0.5, label='±30° limit')
    ax.axhline(-30, color='gray', ls=':', alpha=0.5)
    ax.set_title('Euler Angles')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('[deg]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Row 3: Solve time, iterations, XY or 3D
    ax = axes[2, 0]
    ax.plot(t_ctrl, data['solve_times_us'], 'b-', lw=0.5)
    ax.axhline(data['median_solve_us'], color='r', ls='--',
               label=f"Median={data['median_solve_us']:.0f}μs")
    ax.axhline(dt * 1e6, color='orange', ls=':',
               label=f"Budget={dt*1e6:.0f}μs")
    ax.set_title('Solve Time')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('[μs]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(500, data['median_solve_us'] * 5)])
    
    ax = axes[2, 1]
    ax.plot(t_ctrl, data['iterations'], 'b-', lw=0.5)
    ax.axhline(data['mean_iters'], color='r', ls='--',
               label=f"Mean={data['mean_iters']:.1f}")
    ax.set_title('ADMM Iterations per Solve')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Iterations')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[2, 2]
    ax.plot(x_log[0, :], x_log[1, :], 'b-', lw=1.5, label='Actual')
    ax.plot(ref[0, :N_ref], ref[1, :N_ref], 'r--', lw=1, label='Reference')
    ax.set_title('XY Plane')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    plt.show()


def plot_comparison(data_admm, data_osqp=None, save_path=None):
    """Side-by-side comparison: C ADMM vs OSQP."""
    
    if data_osqp is None:
        plot_single(data_admm, save_path)
        return
    
    dt = data_admm['dt']
    t_err_a = np.arange(len(data_admm['tracking_err'])) * dt
    t_err_o = np.arange(len(data_osqp['tracking_err'])) * dt
    t_ctrl_a = np.arange(data_admm['u_log'].shape[1]) * dt
    t_ctrl_o = np.arange(data_osqp['u_log'].shape[1]) * dt
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Solver Comparison — {data_admm['title']}\n"
                 f"C ADMM (hand-rolled) vs OSQP (library)",
                 fontsize=13, fontweight='bold')
    
    # Tracking error comparison
    ax = axes[0, 0]
    ax.plot(t_err_a, data_admm['tracking_err'] * 100, 'b-', lw=1.5,
            label=f"C ADMM (RMSE={data_admm['rmse']*100:.1f}cm)")
    ax.plot(t_err_o, data_osqp['tracking_err'] * 100, 'r-', lw=1, alpha=0.7,
            label=f"OSQP (RMSE={data_osqp['rmse']*100:.1f}cm)")
    ax.set_title('Tracking Error'); ax.set_xlabel('Time [s]')
    ax.set_ylabel('[cm]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Solve time comparison
    ax = axes[0, 1]
    ax.plot(t_ctrl_a, data_admm['solve_times_us'], 'b-', lw=0.5,
            label=f"C ADMM (med={data_admm['median_solve_us']:.0f}μs)")
    ax.plot(t_ctrl_o, np.array(data_osqp['solve_times_us']) * 1e6, 'r-', lw=0.5, alpha=0.7,
            label=f"OSQP (med={np.median(data_osqp['solve_times_us'])*1e6:.0f}μs)")
    ax.set_title('Solve Time'); ax.set_xlabel('Time [s]')
    ax.set_ylabel('[μs]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2000])
    
    # Iterations comparison
    ax = axes[0, 2]
    ax.plot(t_ctrl_a, data_admm['iterations'], 'b-', lw=0.5,
            label=f"C ADMM (mean={data_admm['mean_iters']:.1f})")
    if 'iterations' in data_osqp:
        ax.plot(t_ctrl_o, data_osqp['iterations'], 'r-', lw=0.5, alpha=0.7,
                label=f"OSQP (mean={np.mean(data_osqp['iterations']):.0f})")
    ax.set_title('Iterations per Solve'); ax.set_xlabel('Time [s]')
    ax.set_ylabel('Iterations'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # XY trajectories
    N_ref = min(data_admm['ref'].shape[1], data_admm['x_log'].shape[1])
    ax = axes[1, 0]
    ax.plot(data_admm['x_log'][0, :], data_admm['x_log'][1, :], 'b-', lw=1.5,
            label='C ADMM')
    ax.plot(data_osqp['x_log'][0, :], data_osqp['x_log'][1, :], 'r-', lw=1, alpha=0.7,
            label='OSQP')
    ax.plot(data_admm['ref'][0, :N_ref], data_admm['ref'][1, :N_ref],
            'k--', lw=0.8, label='Reference')
    ax.set_title('XY Trajectory'); ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]'); ax.set_aspect('equal')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Thrust comparison
    ax = axes[1, 1]
    ax.plot(t_ctrl_a, data_admm['u_log'][0, :] * 1000, 'b-', lw=1, label='C ADMM')
    ax.plot(t_ctrl_o, data_osqp['u_log'][0, :] * 1000, 'r-', lw=1, alpha=0.7, label='OSQP')
    ax.axhline(0.027 * 9.81 * 1000, color='gray', ls='--', alpha=0.5, label='mg')
    ax.set_title('Thrust'); ax.set_xlabel('Time [s]')
    ax.set_ylabel('[mN]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    summary = (
        f"{'Metric':<25s} {'C ADMM':>10s} {'OSQP':>10s}\n"
        f"{'─'*47}\n"
        f"{'RMSE [cm]':<25s} {data_admm['rmse']*100:>10.2f} {data_osqp['rmse']*100:>10.2f}\n"
        f"{'Max Error [cm]':<25s} {data_admm['max_err']*100:>10.2f} {data_osqp['max_err']*100:>10.2f}\n"
        f"{'Median Solve [μs]':<25s} {data_admm['median_solve_us']:>10.0f} {np.median(data_osqp['solve_times_us'])*1e6:>10.0f}\n"
        f"{'Mean Iterations':<25s} {data_admm['mean_iters']:>10.1f} {np.mean(data_osqp.get('iterations', [25])):>10.1f}\n"
        f"{'Speedup':<25s} {np.median(data_osqp['solve_times_us'])*1e6/max(data_admm['median_solve_us'],1):>9.1f}×\n"
    )
    ax.text(0.05, 0.95, summary, fontsize=10, fontfamily='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    plt.show()


def run_osqp_for_comparison(mode, duration):
    """Run OSQP to get comparison data."""
    from mpc_osqp import run_mpc_sim
    x_log, u_log, ref, mpc, tracking_err, solve_times = \
        run_mpc_sim(mode=mode, duration=duration)
    
    return {
        'x_log': x_log, 'u_log': u_log, 'ref': ref,
        'tracking_err': tracking_err,
        'solve_times_us': solve_times,  # in seconds from OSQP
        'rmse': np.sqrt(np.mean(tracking_err**2)),
        'max_err': np.max(tracking_err),
        'title': mode, 'dt': 0.02,
    }


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fig8'
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    compare = '--compare' in sys.argv or '-c' in sys.argv
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    if mode == 'all':
        for m in ['hover', 'fig8', 'helix', 'step']:
            d = 5.0 if m in ['hover', 'step'] else 10.0
            data = run_and_collect(m, d)
            print(f"  {m}: RMSE={data['rmse']*100:.2f}cm, "
                  f"solve={data['median_solve_us']:.0f}μs, "
                  f"iters={data['mean_iters']:.1f}")
            plot_single(data, str(results_dir / f"admm_c_{m}.png"))
    elif compare:
        print("\n  Running C ADMM...")
        data_admm = run_and_collect(mode, duration)
        print("\n  Running OSQP for comparison...")
        data_osqp = run_osqp_for_comparison(mode, duration)
        plot_comparison(data_admm, data_osqp,
                       str(results_dir / f"compare_admm_vs_osqp_{mode}.png"))
    else:
        data = run_and_collect(mode, duration)
        print(f"\n  RMSE: {data['rmse']*100:.2f} cm")
        print(f"  Median solve: {data['median_solve_us']:.0f} μs")
        print(f"  Mean iterations: {data['mean_iters']:.1f}")
        plot_single(data, str(results_dir / f"admm_c_{mode}.png"))
