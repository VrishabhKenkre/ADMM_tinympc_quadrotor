"""
mpc_osqp.py — Linear MPC with OSQP Solver
============================================
Sparse QP formulation for quadrotor trajectory tracking.

This is Day 3-4 of the project. We take the LQR baseline from Day 1-2
and replace it with a proper MPC that:
  - Looks N steps into the future (preview)
  - Respects actuator limits (thrust, torque bounds)
  - Respects state constraints (tilt angle limits)
  - Uses warm starting for real-time performance

QP Formulation (sparse):
  Decision variables:  z = [x[0], x[1], ..., x[N], u[0], ..., u[N-1]]
  
  minimize    ½ z' H z + f' z
  subject to  l ≤ A z ≤ u
  
  where:
    H = block_diag(Q, Q, ..., P, R, R, ..., R)   ← cost Hessian
    f = [-Q·xref[0], ..., -P·xref[N], 0, ..., 0]  ← linear cost (from reference)
    A encodes: dynamics equalities + initial condition + bounds

References:
  - Stellato et al. (2020), "OSQP: An Operator Splitting Solver for QPs"
  - Di Carlo et al. (IROS 2018), sparse MPC formulation for MIT Cheetah
  - Nguyen et al. (ICRA 2024), "TinyMPC" — ADMM-based MPC on Crazyflie

Author: Vrishabh Kenkre (CMU MS MechE)
"""

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MPCParams:
    """MPC configuration parameters."""
    N: int = 20              # prediction horizon
    dt: float = 0.02         # control timestep [s]  → 50 Hz
    
    # Cost weights (diagonal entries)
    # State: [px, py, pz, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]
    Q_diag: np.ndarray = None   # stage cost on states
    R_diag: np.ndarray = None   # stage cost on controls
    
    # State constraints (angle limits to keep linearization valid)
    phi_max: float = np.radians(30)     # max roll [rad]
    theta_max: float = np.radians(30)   # max pitch [rad]
    
    def __post_init__(self):
        if self.Q_diag is None:
            self.Q_diag = np.array([
                10, 10, 10,         # position: track accurately
                1,  1,  1,          # velocity: moderate
                5,  5,  1,          # euler: attitude matters, yaw less
                0.1, 0.1, 0.1      # angular rates: don't care much directly
            ])
        if self.R_diag is None:
            self.R_diag = np.array([
                100,                # thrust: moderate cost
                1e4, 1e4, 1e4       # torques: expensive (small actuators)
            ])


class OSQP_MPC:
    """Linear Model Predictive Controller using OSQP.
    
    Builds the sparse QP once, then updates only f, l, u each timestep.
    The sparsity pattern of H and A never changes — only the values
    of the reference trajectory and initial state change.
    
    Decision variable layout:
        z = [x[0], x[1], ..., x[N], u[0], u[1], ..., u[N-1]]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^
             (N+1) × nx = 252 entries     N × nu = 80 entries
             
        Total: (N+1)×nx + N×nu = 332 entries for N=20, nx=12, nu=4
    
    Constraint layout in A:
        Row block 1: Dynamics equalities     (N × nx = 240 rows)
        Row block 2: Initial condition       (nx = 12 rows)
        Row block 3: State bounds            ((N+1) × nx = 252 rows)  
        Row block 4: Control bounds          (N × nu = 80 rows)
    """
    
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 u_hover: np.ndarray,
                 params: MPCParams = None):
        """
        Args:
            Ad: discrete A matrix [nx × nx]
            Bd: discrete B matrix [nx × nu]
            u_min: control lower bounds [nu]
            u_max: control upper bounds [nu]
            u_hover: hover control (for delta-u formulation) [nu]
            params: MPC parameters
        """
        self.Ad = Ad
        self.Bd = Bd
        self.nx = Ad.shape[0]   # 12
        self.nu = Bd.shape[1]   # 4
        self.u_min = u_min
        self.u_max = u_max
        self.u_hover = u_hover
        self.params = params or MPCParams()
        self.N = self.params.N
        
        # Total decision variable size
        self.n_states = (self.N + 1) * self.nx   # 252
        self.n_controls = self.N * self.nu        # 80
        self.n_dec = self.n_states + self.n_controls  # 332
        
        # Build cost matrices
        Q = np.diag(self.params.Q_diag)
        R = np.diag(self.params.R_diag)
        
        # Terminal cost from DARE (LQR cost-to-go)
        # This makes the finite-horizon MPC approximate infinite-horizon
        self.P_terminal = solve_discrete_are(Ad, Bd, Q, R)
        self.Q = Q
        self.R = R
        
        # Build the QP matrices (done once — sparsity pattern is fixed)
        self._build_qp()
        
        # Setup OSQP solver
        self._setup_solver()
        
        # Warm start storage
        self._prev_z = None
        self._prev_y = None  # dual variables
        
        # Statistics
        self.solve_times = []
    
    def _build_qp(self):
        """Construct H, A matrices (sparse). Called once at initialization.
        
        This is the core of the MPC formulation. Every line here maps
        directly to the math in the docstring.
        """
        N, nx, nu = self.N, self.nx, self.nu
        n_dec = self.n_dec
        
        # ═══════════════════════════════════════════════
        # COST MATRIX H (block-diagonal, very sparse)
        # ═══════════════════════════════════════════════
        #
        # H = block_diag(Q, Q, ..., Q, P, R, R, ..., R)
        #     ^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
        #     (N+1) state blocks          N control blocks
        #
        # H is n_dec × n_dec but has only nx² × (N+1) + nu² × N
        # nonzero entries ≈ 12²×21 + 4²×20 = 3344 out of 332² = 110,224
        # That's 97% zeros!
        
        H_blocks = []
        
        # State cost blocks: Q for steps 0 to N-1, P for step N
        for k in range(N):
            H_blocks.append(self.Q)
        H_blocks.append(self.P_terminal)  # terminal cost
        
        # Control cost blocks: R for steps 0 to N-1
        for k in range(N):
            H_blocks.append(self.R)
        
        # Assemble block-diagonal H
        self.H = sparse.block_diag(H_blocks, format='csc')
        
        # ═══════════════════════════════════════════════
        # LINEAR COST f (depends on reference — updated each solve)
        # ═══════════════════════════════════════════════
        #
        # f = [-Q·xref[0], -Q·xref[1], ..., -P·xref[N], 0, ..., 0]
        #
        # Initialized to zero, updated in solve() with actual reference
        
        self.f = np.zeros(n_dec)
        
        # ═══════════════════════════════════════════════
        # CONSTRAINT MATRIX A
        # ═══════════════════════════════════════════════
        #
        # Four blocks of constraints, stacked vertically:
        #
        # Block 1: Dynamics — x[k+1] = Ad·x[k] + Bd·u[k]
        #   Rewritten: x[k+1] - Ad·x[k] - Bd·u[k] = 0
        #   Each row: [..., -Ad, I, ..., ..., -Bd, ...]
        #   N×nx rows, n_dec columns
        #
        # Block 2: Initial condition — x[0] = x_measured
        #   Each row: [I, 0, ..., 0, ..., 0]
        #   nx rows, n_dec columns
        #
        # Block 3: State bounds — x_min ≤ x[k] ≤ x_max
        #   Identity on state portion
        #   (N+1)×nx rows
        #
        # Block 4: Control bounds — u_min ≤ u[k] ≤ u_max
        #   Identity on control portion
        #   N×nu rows
        
        # --- Block 1: Dynamics equality constraints ---
        # For each k = 0, ..., N-1:
        #   [0...0, -Ad, I, 0...0, 0...0, -Bd, 0...0] · z = 0
        #           ^x[k] ^x[k+1]         ^u[k]
        
        # Build using triplet format (row, col, value)
        rows_dyn = []
        cols_dyn = []
        vals_dyn = []
        
        for k in range(N):
            row_offset = k * nx
            
            # -Ad block at column position for x[k]
            col_offset_xk = k * nx
            for i in range(nx):
                for j in range(nx):
                    if abs(self.Ad[i, j]) > 1e-15:
                        rows_dyn.append(row_offset + i)
                        cols_dyn.append(col_offset_xk + j)
                        vals_dyn.append(-self.Ad[i, j])
            
            # +I block at column position for x[k+1]
            col_offset_xk1 = (k + 1) * nx
            for i in range(nx):
                rows_dyn.append(row_offset + i)
                cols_dyn.append(col_offset_xk1 + i)
                vals_dyn.append(1.0)
            
            # -Bd block at column position for u[k]
            col_offset_uk = self.n_states + k * nu
            for i in range(nx):
                for j in range(nu):
                    if abs(self.Bd[i, j]) > 1e-15:
                        rows_dyn.append(row_offset + i)
                        cols_dyn.append(col_offset_uk + j)
                        vals_dyn.append(-self.Bd[i, j])
        
        n_dyn_rows = N * nx  # 240
        
        # --- Block 2: Initial condition x[0] = x_measured ---
        rows_ic = []
        cols_ic = []
        vals_ic = []
        
        for i in range(nx):
            rows_ic.append(n_dyn_rows + i)
            cols_ic.append(i)  # x[0] is at the start of z
            vals_ic.append(1.0)
        
        n_ic_rows = nx  # 12
        
        # --- Block 3: State bounds (identity on state portion) ---
        rows_xb = []
        cols_xb = []
        vals_xb = []
        
        row_start = n_dyn_rows + n_ic_rows
        for k in range(N + 1):
            for i in range(nx):
                rows_xb.append(row_start + k * nx + i)
                cols_xb.append(k * nx + i)
                vals_xb.append(1.0)
        
        n_xb_rows = (N + 1) * nx  # 252
        
        # --- Block 4: Control bounds (identity on control portion) ---
        rows_ub = []
        cols_ub = []
        vals_ub = []
        
        row_start2 = n_dyn_rows + n_ic_rows + n_xb_rows
        for k in range(N):
            for i in range(nu):
                rows_ub.append(row_start2 + k * nu + i)
                cols_ub.append(self.n_states + k * nu + i)
                vals_ub.append(1.0)
        
        n_ub_rows = N * nu  # 80
        
        # --- Assemble A ---
        total_rows = n_dyn_rows + n_ic_rows + n_xb_rows + n_ub_rows
        
        all_rows = rows_dyn + rows_ic + rows_xb + rows_ub
        all_cols = cols_dyn + cols_ic + cols_xb + cols_ub
        all_vals = vals_dyn + vals_ic + vals_xb + vals_ub
        
        self.A = sparse.csc_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(total_rows, n_dec)
        )
        
        # ═══════════════════════════════════════════════
        # BOUNDS l, u (updated each solve for initial condition)
        # ═══════════════════════════════════════════════
        
        # Block 1: Dynamics — equality: l = u = d (gravity offset)
        # The linearized dynamics in absolute coordinates is:
        #   x[k+1] = Ad·x[k] + Bd·u[k] + d
        # where d = (I - Ad)·x_hover - Bd·u_hover encodes gravity.
        # Without d, the MPC model has no gravity and won't apply thrust!
        x_hover = np.zeros(nx)  # position cancels, all other states = 0
        d = (np.eye(nx) - self.Ad) @ x_hover - self.Bd @ self.u_hover
        self._gravity_offset = d
        
        l_dyn = np.tile(d, N)
        u_dyn = np.tile(d, N)
        
        # Block 2: Initial condition — equality: l = u = x_measured
        # (filled in at solve time)
        l_ic = np.zeros(n_ic_rows)
        u_ic = np.zeros(n_ic_rows)
        
        # Block 3: State bounds
        # Position: unbounded (±∞)
        # Velocity: unbounded
        # Euler angles: φ ∈ [-30°, +30°], θ ∈ [-30°, +30°], ψ unbounded
        # Angular rates: unbounded
        INF = 1e10  # OSQP uses finite bounds, not np.inf
        x_lb = np.array([-INF, -INF, -INF,      # position
                         -INF, -INF, -INF,       # velocity
                         -self.params.phi_max,    # roll limit
                         -self.params.theta_max,  # pitch limit
                         -INF,                    # yaw (unbounded)
                         -INF, -INF, -INF])       # angular rates
        x_ub = np.array([INF, INF, INF,
                         INF, INF, INF,
                         self.params.phi_max,
                         self.params.theta_max,
                         INF,
                         INF, INF, INF])
        
        l_xb = np.tile(x_lb, N + 1)
        u_xb = np.tile(x_ub, N + 1)
        
        # Block 4: Control bounds
        l_ub = np.tile(self.u_min, N)
        u_ub = np.tile(self.u_max, N)
        
        self.l = np.concatenate([l_dyn, l_ic, l_xb, l_ub])
        self.u = np.concatenate([u_dyn, u_ic, u_xb, u_ub])
        
        # Store offsets for fast updates
        self._ic_start = n_dyn_rows  # where initial condition bounds start
        self._ic_end = n_dyn_rows + n_ic_rows
        
        # Print summary
        print(f"  QP dimensions:")
        print(f"    Decision variables: {n_dec} "
              f"({self.n_states} states + {self.n_controls} controls)")
        print(f"    Constraints: {total_rows} "
              f"({n_dyn_rows} dynamics + {n_ic_rows} init + "
              f"{n_xb_rows} state bounds + {n_ub_rows} ctrl bounds)")
        print(f"    H nonzeros: {self.H.nnz} / {n_dec**2} "
              f"({100*self.H.nnz/n_dec**2:.1f}% dense)")
        print(f"    A nonzeros: {self.A.nnz} / {total_rows*n_dec} "
              f"({100*self.A.nnz/(total_rows*n_dec):.1f}% dense)")
    
    def _setup_solver(self):
        """Initialize OSQP with the QP structure."""
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=self.H,           # cost Hessian (OSQP calls it P, not H)
            q=self.f,           # linear cost
            A=self.A,           # constraint matrix
            l=self.l,           # lower bounds
            u=self.u,           # upper bounds
            warm_starting=True, # reuse previous solution
            polish=True,        # high-accuracy solution refinement
            adaptive_rho=True,  # auto-tune ADMM penalty parameter
            max_iter=500,       # enough for convergence
            eps_abs=1e-4,       # absolute tolerance
            eps_rel=1e-4,       # relative tolerance
            verbose=False       # silent operation
        )
    
    def solve(self, x_current: np.ndarray, 
              x_ref: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Solve the MPC QP for one timestep.
        
        Args:
            x_current: current measured state [nx]
            x_ref: reference trajectory [nx × (N+1)] — columns are states
                   at times [t, t+dt, t+2dt, ..., t+N*dt]
        
        Returns:
            u_opt: optimal control for current timestep [nu]
            info: dict with solve time, status, predicted trajectory
        """
        N, nx, nu = self.N, self.nx, self.nu
        
        # ═══ Update linear cost f (reference trajectory changed) ═══
        #
        # f = [-Q·xref[0], -Q·xref[1], ..., -P·xref[N], 0, ..., 0]
        #
        # Why negative? The cost ½z'Hz + f'z has minimum at z = -H⁻¹f.
        # With H = Q and f = -Q·xref, the minimum is at z = xref. ✓
        
        for k in range(N):
            self.f[k*nx : (k+1)*nx] = -self.Q @ x_ref[:, k]
        self.f[N*nx : (N+1)*nx] = -self.P_terminal @ x_ref[:, N]
        
        # Control portion: penalize (u - u_hover), not u directly
        # Cost: (u - u_hover)' R (u - u_hover) = u'Ru - 2·u_hover'R·u + const
        # In QP form: ½z'Hz + f'z, the linear term for controls is -R·u_hover
        for k in range(N):
            ctrl_start = (N+1)*nx + k*nu
            self.f[ctrl_start : ctrl_start + nu] = -self.R @ self.u_hover
        
        # ═══ Update initial condition (x[0] = x_current) ═══
        self.l[self._ic_start : self._ic_end] = x_current
        self.u[self._ic_start : self._ic_end] = x_current
        
        # ═══ Warm start ═══
        # Shift previous solution by one step as initial guess
        if self._prev_z is not None:
            z_warm = np.zeros(self.n_dec)
            # Shift states: x[1]→x[0], x[2]→x[1], ..., x[N]→x[N-1], repeat x[N]
            for k in range(N):
                z_warm[k*nx : (k+1)*nx] = self._prev_z[(k+1)*nx : (k+2)*nx]
            z_warm[N*nx : (N+1)*nx] = self._prev_z[N*nx : (N+1)*nx]  # repeat last
            # Shift controls: u[1]→u[0], ..., u[N-1]→u[N-2], repeat u[N-1]
            state_offset = (N+1) * nx
            for k in range(N-1):
                z_warm[state_offset + k*nu : state_offset + (k+1)*nu] = \
                    self._prev_z[state_offset + (k+1)*nu : state_offset + (k+2)*nu]
            z_warm[state_offset + (N-1)*nu :] = \
                self._prev_z[state_offset + (N-1)*nu :]  # repeat last
            
            self.solver.warm_start(x=z_warm)
        
        # ═══ Update OSQP and solve ═══
        self.solver.update(q=self.f, l=self.l, u=self.u)
        
        t_start = time.perf_counter()
        result = self.solver.solve()
        t_solve = time.perf_counter() - t_start
        
        self.solve_times.append(t_solve)
        
        # ═══ Extract solution ═══
        if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
            z_opt = result.x
            self._prev_z = z_opt  # store for warm starting
            
            # First control input
            u_opt = z_opt[(N+1)*nx : (N+1)*nx + nu]
            
            # Predicted state trajectory (for visualization)
            x_pred = np.zeros((nx, N+1))
            for k in range(N+1):
                x_pred[:, k] = z_opt[k*nx : (k+1)*nx]
            
            # Full control sequence
            u_seq = np.zeros((nu, N))
            for k in range(N):
                u_seq[:, k] = z_opt[(N+1)*nx + k*nu : (N+1)*nx + (k+1)*nu]
            
            info = {
                'status': result.info.status,
                'solve_time': t_solve,
                'iterations': result.info.iter,
                'x_pred': x_pred,
                'u_seq': u_seq,
                'cost': result.info.obj_val
            }
        else:
            # Solver failed — use hover as fallback
            print(f"  ⚠ OSQP failed: {result.info.status}")
            u_opt = self.u_hover.copy()
            info = {
                'status': result.info.status,
                'solve_time': t_solve,
                'iterations': result.info.iter,
                'x_pred': None,
                'u_seq': None,
                'cost': None
            }
        
        return u_opt, info


def run_mpc_sim(mode='fig8', duration=10.0):
    """Run MPC in closed loop with MuJoCo."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from quad_env import CrazyflieEnv, generate_figure8_reference, \
                         generate_hover_reference, generate_helix_reference, \
                         generate_step_response
    from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
    
    p = QuadParams()
    mpc_params = MPCParams()
    dt = mpc_params.dt
    N = mpc_params.N
    
    # Build dynamics
    Ac, Bc = linearize_at_hover(p)
    Ad, Bd = discretize_dynamics(Ac, Bc, dt, method='expm')
    
    # Create environment
    model_path = str(Path(__file__).parent.parent /
                     "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml")
    env = CrazyflieEnv(model_path=model_path, dt_sim=0.002, dt_ctrl=dt)
    
    # Generate reference trajectory
    total_steps = int(duration / dt)
    ref_duration = duration + N * dt + 1.0  # extra for horizon lookahead
    
    if mode == 'hover':
        ref = generate_hover_reference(np.array([0, 0, 1.0]), ref_duration, dt)
        title = "Hover stabilization"
    elif mode == 'fig8':
        ref = generate_figure8_reference(
            center=np.array([0.0, 0.0]), radius=0.5,
            height=1.0, period=4.0, duration=ref_duration, dt=dt)
        title = "Figure-8 tracking"
    elif mode == 'helix':
        ref = generate_helix_reference(
            center=np.array([0.0, 0.0]), radius=0.4,
            z_start=0.5, z_end=1.5, period=3.0,
            duration=ref_duration, dt=dt)
        title = "Helix tracking"
    elif mode == 'step':
        ref = generate_step_response(
            start=np.array([0, 0, 1.0]),
            end=np.array([0.5, 0.3, 1.2]),
            step_time=1.0, duration=ref_duration, dt=dt)
        title = "Step response"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    N_ref = ref.shape[1]
    
    # Build MPC
    print(f"\n{'='*60}")
    print(f"  OSQP MPC — {title}")
    print(f"  Horizon: {N} steps ({N*dt:.1f}s), dt={dt*1000:.0f}ms")
    print(f"{'='*60}\n")
    
    mpc = OSQP_MPC(Ad, Bd, p.u_min, p.u_max, 
                    np.array([p.hover_thrust, 0, 0, 0]), mpc_params)
    
    # Reset environment
    x0 = env.reset(pos=ref[0:3, 0])
    
    # Storage
    x_log = np.zeros((12, total_steps + 1))
    u_log = np.zeros((4, total_steps))
    x_log[:, 0] = x0
    
    print(f"\n  Running {total_steps} steps...")
    print(f"  {'Time':>6s} │ {'PosErr':>8s} │ {'Thrust':>7s} │ {'Solve':>7s} │ {'Iters':>5s} │ Status")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*10}")
    
    for i in range(total_steps):
        # Build reference window: columns are x_ref at t, t+dt, ..., t+N*dt
        ref_window = np.zeros((12, N + 1))
        for k in range(N + 1):
            ref_idx = min(i + k, N_ref - 1)
            ref_window[:, k] = ref[:, ref_idx]
        
        # Solve MPC
        u_opt, info = mpc.solve(x_log[:, i], ref_window)
        
        # Apply to environment
        x_log[:, i + 1] = env.step(u_opt)
        u_log[:, i] = u_opt
        
        # Print every 1 second
        if (i + 1) % int(1.0 / dt) == 0:
            t_now = (i + 1) * dt
            pos_err = np.linalg.norm(x_log[0:3, i+1] - ref[0:3, min(i+1, N_ref-1)])
            print(f"  {t_now:5.1f}s │ {pos_err*100:7.2f}cm │ "
                  f"{u_opt[0]*1000:6.1f}mN │ "
                  f"{info['solve_time']*1000:6.2f}ms │ "
                  f"{info['iterations']:5d} │ {info['status']}")
    
    # ═══ Results ═══
    print(f"\n{'='*60}")
    print(f"  Results:")
    
    tracking_err = np.linalg.norm(
        x_log[0:3, :total_steps] - ref[0:3, :total_steps], axis=0)
    rmse = np.sqrt(np.mean(tracking_err**2))
    max_err = np.max(tracking_err)
    
    solve_times = np.array(mpc.solve_times)
    
    print(f"    RMSE tracking error:  {rmse*100:.2f} cm")
    print(f"    Max tracking error:   {max_err*100:.2f} cm")
    print(f"    Avg solve time:       {np.mean(solve_times)*1000:.2f} ms")
    print(f"    Max solve time:       {np.max(solve_times)*1000:.2f} ms")
    print(f"    Mean iterations:      {np.mean([1 for _ in mpc.solve_times]):.0f}")
    print(f"{'='*60}\n")
    
    return x_log, u_log, ref, mpc, tracking_err, solve_times


def plot_mpc_results(x_log, u_log, ref, tracking_err, solve_times, 
                     dt, save_path=None):
    """Generate comparison plots: MPC vs reference."""
    import matplotlib.pyplot as plt
    
    N_sim = x_log.shape[1]
    N_ref = min(ref.shape[1], N_sim)
    t = np.arange(N_sim) * dt
    t_ref = np.arange(N_ref) * dt
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Crazyflie MPC — OSQP Solver Results\n'
                 'Sparse QP, N=20, 50 Hz', fontsize=13, fontweight='bold')
    
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
    
    # Row 2: Tracking error, controls, angles
    ax = axes[1, 0]
    t_err = np.arange(len(tracking_err)) * dt
    ax.plot(t_err, tracking_err * 100, 'b-', lw=1.5)
    ax.axhline(np.sqrt(np.mean(tracking_err**2))*100, color='r', ls='--', 
               label=f'RMSE={np.sqrt(np.mean(tracking_err**2))*100:.1f}cm')
    ax.set_title('3D Tracking Error')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('[cm]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(t[:-1], u_log[0, :]*1000, 'k-', lw=1, label='Thrust')
    ax.axhline(0.027*9.81*1000, color='r', ls='--', alpha=0.5, label='mg')
    ax.set_title('Thrust'); ax.set_xlabel('Time [s]')
    ax.set_ylabel('[mN]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(t, x_log[6, :]*180/np.pi, 'b-', lw=1, label='φ (roll)')
    ax.plot(t, x_log[7, :]*180/np.pi, 'r-', lw=1, label='θ (pitch)')
    ax.axhline(30, color='gray', ls=':', alpha=0.5, label='±30° limit')
    ax.axhline(-30, color='gray', ls=':', alpha=0.5)
    ax.set_title('Euler Angles')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('[deg]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Row 3: Solve time, XY plane, 3D trajectory
    ax = axes[2, 0]
    ax.plot(np.arange(len(solve_times)) * dt, 
            np.array(solve_times) * 1000, 'b-', lw=0.5)
    ax.axhline(np.mean(solve_times)*1000, color='r', ls='--',
               label=f'Mean={np.mean(solve_times)*1000:.2f}ms')
    ax.axhline(dt*1000, color='orange', ls=':', label=f'Budget={dt*1000:.0f}ms')
    ax.set_title('QP Solve Time')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('[ms]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    ax.plot(x_log[0, :], x_log[1, :], 'b-', lw=1.5, label='Actual')
    ax.plot(ref[0, :N_ref], ref[1, :N_ref], 'r--', lw=1, label='Reference')
    ax.set_title('XY Plane')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[2, 2]
    ax.remove()
    ax = fig.add_subplot(3, 3, 9, projection='3d')
    ax.plot3D(x_log[0, :], x_log[1, :], x_log[2, :], 'b-', lw=1.5, label='Actual')
    ax.plot3D(ref[0, :N_ref], ref[1, :N_ref], ref[2, :N_ref], 'r--', lw=1, label='Ref')
    ax.set_title('3D Trajectory')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    plt.show()


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fig8'
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    
    x_log, u_log, ref, mpc, tracking_err, solve_times = \
        run_mpc_sim(mode=mode, duration=duration)
    
    save_path = str(Path(__file__).parent.parent / "results" / f"mpc_osqp_{mode}.png")
    plot_mpc_results(x_log, u_log, ref, tracking_err, solve_times,
                     mpc.params.dt, save_path=save_path)
