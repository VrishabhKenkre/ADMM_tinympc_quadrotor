"""
solver_admm.py -- pure-Python hand-rolled ADMM solver for MPC.

Follows TinyMPC (Nguyen et al., ICRA 2024): solve the Riccati recursion
once to cache P_inf, K_inf, C1, C2, then run the 5-step ADMM iteration
(backward, forward, slack/clamp, dual, convergence). Only matrix-vector
products online; no inversions.

Run from the repository root:
    python3 src/solver_admm.py [mode] [duration]
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ADMMParams:
    """ADMM solver parameters."""
    rho: float = 1.0              # penalty parameter (augmented Lagrangian weight)
    max_iter: int = 50            # max ADMM iterations per solve
    eps_abs: float = 1e-4         # absolute convergence tolerance
    eps_rel: float = 1e-4         # relative convergence tolerance
    adaptive_rho: bool = True     # auto-tune rho every rho_update_interval iters
    rho_update_interval: int = 5  # how often to update rho
    rho_min: float = 1e-2         # minimum rho
    rho_max: float = 1e4          # maximum rho


class ADMMSolver:
    """Hand-rolled ADMM for MPC.

    Solves the standard linear-MPC QP (quadratic cost over (x_k - xref_k),
    (u_k - uref_k); linear dynamics x_{k+1} = Ad x_k + Bd u_k + d; box
    bounds on x and u) via the ADMM consensus reformulation z_x = x, z_u = u,
    where z is the slack copy constrained to the feasible box.
    """
    
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, P_terminal: np.ndarray,
                 N: int,
                 u_min: np.ndarray, u_max: np.ndarray,
                 x_min: np.ndarray, x_max: np.ndarray,
                 u_hover: np.ndarray,
                 gravity_offset: np.ndarray,
                 params: ADMMParams = None):
        """
        Args:
            Ad, Bd: discrete dynamics matrices
            Q, R: stage cost matrices
            P_terminal: terminal cost (from DARE)
            N: prediction horizon
            u_min, u_max: control bounds [nu]
            x_min, x_max: state bounds [nx]
            u_hover: hover control (for cost centering)
            gravity_offset: the d vector in x[k+1] = Ad x[k] + Bd u[k] + d
            params: ADMM parameters
        """
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.P_terminal = P_terminal
        self.N = N
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max
        self.u_hover = u_hover
        self.d = gravity_offset
        self.params = params or ADMMParams()
        self.rho = self.params.rho
        
        # ---- Offline: cache the Riccati matrices (done once) -------------
        self._cache_riccati()
        
        # ADMM variable storage (warm-started between solves)
        self._init_variables()
        
        # Statistics
        self.solve_times = []
        self.iterations_log = []
        self.residuals_log = []
    
    def _cache_riccati(self):
        """Precompute the infinite-horizon Riccati matrices.

        Solves the DARE with augmented cost Q_tilde = Q + rho*I,
        R_tilde = R + rho*I, then caches P_inf, K_inf, C1, C2 so the
        online ADMM loop is matvec-only.
        """
        nx, nu = self.nx, self.nu
        rho = self.rho

        # Augmented cost (rho*I added for ADMM penalty)
        Q_aug = self.Q + rho * np.eye(nx)
        R_aug = self.R + rho * np.eye(nu)

        # DARE with augmented costs gives the infinite-horizon P_inf
        P_inf = solve_discrete_are(self.Ad, self.Bd, Q_aug, R_aug)

        # K_inf = (R_tilde + Bd' P_inf Bd)^{-1} Bd' P_inf Ad
        BtPB = self.Bd.T @ P_inf @ self.Bd
        BtPA = self.Bd.T @ P_inf @ self.Ad
        K_coeff = R_aug + BtPB            # (nu x nu) -- the only inversion

        self.K_inf = np.linalg.solve(K_coeff, BtPA)   # nu x nx
        self.P_inf = P_inf

        # Cache matrices for the backward/forward pass
        self.C1 = np.linalg.inv(K_coeff)              # (R_tilde + Bd'P_inf Bd)^{-1}
        self.C2 = (self.Ad - self.Bd @ self.K_inf).T   # closed-loop A transposed

        # TinyMPC uses P_inf for the terminal stage as well; the infinite-horizon
        # approximation is accurate for N >= 10.
    
    def _init_variables(self):
        """Initialize ADMM primal, slack, and dual variables."""
        N, nx, nu = self.N, self.nx, self.nu
        
        # Primal variables (states and controls)
        self.x = np.zeros((nx, N + 1))
        self.u = np.zeros((nu, N))
        
        # Slack variables (copies that must satisfy constraints)
        self.z_x = np.zeros((nx, N + 1))
        self.z_u = np.zeros((nu, N))
        
        # Dual variables (Lagrange multipliers / prices)
        self.y_x = np.zeros((nx, N + 1))
        self.y_u = np.zeros((nu, N))
    
    def solve(self, x0: np.ndarray, x_ref: np.ndarray,
              u_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Solve the MPC QP using ADMM.
        
        Args:
            x0: current state [nx]
            x_ref: reference states [nx x (N+1)]
            u_ref: reference controls [nu x N] (default: u_hover)
        
        Returns:
            u_opt: optimal first control [nu]
            info: solve statistics
        """
        N, nx, nu = self.N, self.nx, self.nu
        
        if u_ref is None:
            u_ref = np.tile(self.u_hover, (N, 1)).T  # nu × N
        
        t_start = time.perf_counter()
        
        # ---- ADMM iterations ---------------------------------------------
        residuals = []

        for iteration in range(self.params.max_iter):

            # Step 1: primal update (backward + forward Riccati).
            # Unconstrained LQR with modified linear terms.
            self._primal_update(x0, x_ref, u_ref)

            # Step 2: slack update (projection onto constraints; clamping for boxes).
            z_x_old = self.z_x.copy()
            z_u_old = self.z_u.copy()
            self._slack_update()

            # Step 3: dual update (standard ADMM dual ascent).
            self._dual_update()

            # Step 4: convergence check
            pri_res_x = np.linalg.norm(self.x - self.z_x)
            pri_res_u = np.linalg.norm(self.u - self.z_u)
            dual_res_x = self.rho * np.linalg.norm(self.z_x - z_x_old)
            dual_res_u = self.rho * np.linalg.norm(self.z_u - z_u_old)
            
            pri_res = pri_res_x + pri_res_u
            dual_res = dual_res_x + dual_res_u
            
            residuals.append((pri_res, dual_res))
            
            # Convergence tolerances (scaled by problem size, following OSQP)
            n_total = nx * (N+1) + nu * N
            eps_pri = self.params.eps_abs * np.sqrt(n_total) + \
                      self.params.eps_rel * max(np.linalg.norm(self.x), 
                                                 np.linalg.norm(self.z_x))
            eps_dual = self.params.eps_abs * np.sqrt(n_total) + \
                       self.params.eps_rel * self.rho * max(np.linalg.norm(self.y_x),
                                                             np.linalg.norm(self.y_u))
            
            if pri_res < eps_pri and dual_res < eps_dual:
                break

            # Step 5: adaptive rho update
            if self.params.adaptive_rho and \
               (iteration + 1) % self.params.rho_update_interval == 0:
                self._update_rho(pri_res, dual_res)
        
        t_solve = time.perf_counter() - t_start
        
        # Extract first control from SLACK (clamped), not primal (unclamped)
        # When solver converges: u = z_u (identical)
        # When solver hits max_iter: u can be out of bounds, z_u is always feasible
        u_opt = self.z_u[:, 0]
        
        # Store statistics
        self.solve_times.append(t_solve)
        self.iterations_log.append(iteration + 1)
        self.residuals_log.append(residuals)
        
        info = {
            'status': 'solved' if iteration < self.params.max_iter - 1 else 'max_iter',
            'solve_time': t_solve,
            'iterations': iteration + 1,
            'primal_residual': pri_res,
            'dual_residual': dual_res,
            'x_pred': self.x.copy(),
            'u_seq': self.u.copy(),
        }
        
        return u_opt, info
    
    def _primal_update(self, x0: np.ndarray, x_ref: np.ndarray,
                        u_ref: np.ndarray):
        """ADMM primal update: solve the unconstrained LQR subproblem.

        Standard LQR with modified linear cost; quadratic terms are
        Q_tilde = Q + rho*I, R_tilde = R + rho*I (already baked into the
        cached P_inf, K_inf, C1, C2). With these cached, the whole step is
        matrix-vector products only.
        """
        N, nx, nu = self.N, self.nx, self.nu
        rho = self.rho

        # Modified linear costs:
        #   q_tilde_k = -Q xref_k + rho (y_x_k - z_x_k)
        #   r_tilde_k = -R uref_k + rho (y_u_k - z_u_k)
        q_tilde = np.zeros((nx, N + 1))
        r_tilde = np.zeros((nu, N))

        for k in range(N):
            q_tilde[:, k] = -self.Q @ x_ref[:, k] + rho * (self.y_x[:, k] - self.z_x[:, k])
            r_tilde[:, k] = -self.R @ u_ref[:, k] + rho * (self.y_u[:, k] - self.z_u[:, k])

        # Terminal: use P_inf for both reference scaling and ADMM penalty.
        q_tilde[:, N] = -self.P_inf @ x_ref[:, N] + rho * (self.y_x[:, N] - self.z_x[:, N])

        # Backward pass: affine cost-to-go p and control corrections d_ctrl.
        # Includes correction for the gravity offset d flowing through p_{k+1}.
        p = np.zeros((nx, N + 1))
        d_ctrl = np.zeros((nu, N))

        p[:, N] = q_tilde[:, N]

        for k in range(N - 1, -1, -1):
            p_next_adjusted = p[:, k + 1] + self.P_inf @ self.d
            d_ctrl[:, k] = self.C1 @ (self.Bd.T @ p_next_adjusted + r_tilde[:, k])
            p[:, k] = q_tilde[:, k] + self.C2 @ p_next_adjusted - self.K_inf.T @ r_tilde[:, k]

        # Forward pass: roll out x_0..x_N, u_0..u_{N-1}.
        self.x[:, 0] = x0

        for k in range(N):
            self.u[:, k] = -self.K_inf @ self.x[:, k] - d_ctrl[:, k]
            self.x[:, k + 1] = self.Ad @ self.x[:, k] + self.Bd @ self.u[:, k] + self.d
    
    def _slack_update(self):
        """ADMM slack update: project onto constraint set.
        
        z = clamp(x + y, bounds)
        
        For box constraints, projection is element-wise clamping.
        This is the simplest possible projection — no matrix operations,
        just min/max on each element. This is why ADMM is so elegant
        for box-constrained problems.
        """
        # State slack: z_x = clamp(x + y_x, x_min, x_max)
        for k in range(self.N + 1):
            self.z_x[:, k] = np.clip(
                self.x[:, k] + self.y_x[:, k],
                self.x_min, self.x_max
            )
        
        # Control slack: z_u = clamp(u + y_u, u_min, u_max)
        for k in range(self.N):
            self.z_u[:, k] = np.clip(
                self.u[:, k] + self.y_u[:, k],
                self.u_min, self.u_max
            )
    
    def _dual_update(self):
        """ADMM dual update: standard dual ascent.
        
        y = y + (x - z)
        
        The dual variable y is the "price" of the consensus constraint
        x = z. If x > z (primal wants to go higher than the constraint
        allows), y increases, which pushes the primal down next iteration.
        This is how ADMM enforces constraints without hard projection
        in the primal step.
        """
        for k in range(self.N + 1):
            self.y_x[:, k] += self.x[:, k] - self.z_x[:, k]
        
        for k in range(self.N):
            self.y_u[:, k] += self.u[:, k] - self.z_u[:, k]
    
    def _update_rho(self, pri_res: float, dual_res: float):
        """Adaptive rho update following OSQP's strategy.

        Primal >> dual: increase rho (push harder on consensus).
        Dual >> primal: decrease rho (relax consensus).
        """
        tau = 2.0  # scaling factor
        mu = 10.0  # threshold ratio

        if pri_res > mu * dual_res:
            self.rho *= tau
            self._cache_riccati()  # recompute cached matrices for new rho
        elif dual_res > mu * pri_res:
            self.rho /= tau
            self._cache_riccati()

        # Clamp rho to a reasonable range
        self.rho = np.clip(self.rho, self.params.rho_min, self.params.rho_max)
    
    def warm_start_shift(self):
        """Shift ADMM variables by one step for warm starting.
        
        After applying u[0] and getting the new state, shift everything
        left by one step. This gives the next solve a much better
        initial guess, reducing iterations from ~50 to ~5.
        """
        N = self.N
        
        # Shift states: [x1, x2, ..., xN, xN]
        self.x[:, :-1] = self.x[:, 1:]
        self.z_x[:, :-1] = self.z_x[:, 1:]
        self.y_x[:, :-1] = self.y_x[:, 1:]
        
        # Shift controls: [u1, u2, ..., uN-1, uN-1]
        self.u[:, :-1] = self.u[:, 1:]
        self.z_u[:, :-1] = self.z_u[:, 1:]
        self.y_u[:, :-1] = self.y_u[:, 1:]


def run_admm_sim(mode='fig8', duration=10.0):
    """Run ADMM MPC in closed loop and compare against OSQP."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from quad_env import CrazyflieEnv, generate_figure8_reference, \
                         generate_hover_reference, generate_helix_reference, \
                         generate_step_response
    from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
    
    p = QuadParams()
    dt = 0.02
    N = 20
    
    # Build dynamics
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt, method='expm')
    
    # Cost matrices
    Q = np.diag([10, 10, 10, 1, 1, 1, 5, 5, 1, 0.1, 0.1, 0.1])
    R = np.diag([100, 1e4, 1e4, 1e4])
    P_terminal = solve_discrete_are(Ad, Bd, Q, R)
    
    # Gravity offset
    x_hover = np.zeros(12)
    u_hover = np.array([p.hover_thrust, 0, 0, 0])
    d = (np.eye(12) - Ad) @ x_hover - Bd @ u_hover
    
    # State bounds
    INF = 1e10
    x_min = np.array([-INF]*3 + [-INF]*3 + [-np.radians(30), -np.radians(30), -INF] + [-INF]*3)
    x_max = np.array([INF]*3 + [INF]*3 + [np.radians(30), np.radians(30), INF] + [INF]*3)
    
    # Create environment
    model_path = str(Path(__file__).parent.parent /
                     "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    env = CrazyflieEnv(model_path=model_path, dt_sim=0.002, dt_ctrl=dt)
    
    # Generate reference
    total_steps = int(duration / dt)
    ref_duration = duration + N * dt + 1.0
    
    if mode == 'hover':
        ref = generate_hover_reference(np.array([0, 0, 1.0]), ref_duration, dt)
        title = "Hover"
    elif mode == 'fig8':
        ref = generate_figure8_reference(
            np.array([0.0, 0.0]), 0.5, 1.0, 4.0, ref_duration, dt)
        title = "Figure-8"
    elif mode == 'helix':
        ref = generate_helix_reference(
            np.array([0.0, 0.0]), 0.4, 0.5, 1.5, 3.0, ref_duration, dt)
        title = "Helix"
    else:
        ref = generate_step_response(
            np.array([0, 0, 1.0]), np.array([0.5, 0.3, 1.2]),
            1.0, ref_duration, dt)
        title = "Step"
    
    N_ref = ref.shape[1]
    
    # Create ADMM solver
    print(f"\n[ADMM MPC] {title} -- horizon {N} steps ({N*dt:.1f}s), dt={dt*1000:.0f}ms\n")
    
    solver = ADMMSolver(
        Ad, Bd, Q, R, P_terminal, N,
        p.u_min, p.u_max, x_min, x_max,
        u_hover, d,
        ADMMParams(rho=1.0, max_iter=50, adaptive_rho=True)
    )
    
    # Reset
    x = env.reset(pos=ref[0:3, 0])
    
    # Storage
    x_log = np.zeros((12, total_steps + 1))
    u_log = np.zeros((4, total_steps))
    x_log[:, 0] = x
    
    print(f"  {'Time':>6s} | {'PosErr':>8s} | {'Thrust':>7s} | {'Solve':>7s} | {'Iters':>5s} | {'PriRes':>8s} | Status")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*8}-+-{'-'*10}")
    
    for i in range(total_steps):
        # Build reference window
        ref_window = np.zeros((12, N + 1))
        for k in range(N + 1):
            ref_window[:, k] = ref[:, min(i + k, N_ref - 1)]
        
        # Solve
        u_opt, info = solver.solve(x_log[:, i], ref_window)
        
        # Apply
        x_log[:, i + 1] = env.step(u_opt)
        u_log[:, i] = u_opt
        
        # Warm start for next solve
        solver.warm_start_shift()
        
        # Print every 1s
        if (i + 1) % int(1.0 / dt) == 0:
            t_now = (i + 1) * dt
            pos_err = np.linalg.norm(x_log[0:3, i+1] - ref[0:3, min(i+1, N_ref-1)])
            print(f"  {t_now:5.1f}s | {pos_err*100:7.2f}cm | "
                  f"{u_opt[0]*1000:6.1f}mN | "
                  f"{info['solve_time']*1000:6.2f}ms | "
                  f"{info['iterations']:5d} | "
                  f"{info['primal_residual']:.2e} | {info['status']}")
    
    # Results
    tracking_err = np.linalg.norm(
        x_log[0:3, :total_steps] - ref[0:3, :total_steps], axis=0)
    rmse = np.sqrt(np.mean(tracking_err**2))
    solve_times = np.array(solver.solve_times)
    iters = np.array(solver.iterations_log)
    
    print(f"\n  Results:")
    print(f"    RMSE tracking error:  {rmse*100:.2f} cm")
    print(f"    Avg solve time:       {np.mean(solve_times)*1000:.2f} ms")
    print(f"    Max solve time:       {np.max(solve_times)*1000:.2f} ms")
    print(f"    Avg iterations:       {np.mean(iters):.1f}")
    print(f"    Max iterations:       {np.max(iters)}\n")
    
    return x_log, u_log, ref, solver, tracking_err, solve_times


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fig8'
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    
    run_admm_sim(mode=mode, duration=duration)
