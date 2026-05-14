"""
solver_reluqp.py — ReLU-QP Solver for MPC (Condensed Formulation)
===================================================================
Uses the REX Lab's ReLU-QP (ICRA 2024) to solve the MPC QP.

KEY INSIGHT: The sparse formulation (states + controls as variables, 332 vars,
584 constraints including 240 dynamics equalities) causes ReLU-QP to diverge
because the adaptive rho oscillates between satisfying equalities and inequalities.

The fix: CONDENSED formulation. Eliminate states via prediction matrices:
    x[k] = A^k * x0 + sum(A^(k-1-j) * B * u[j]) + gravity_terms
Only controls remain as decision variables (80 vars, 80 constraints).
W matrix shrinks from 1500x1500 to 320x320. Converges in 25 warm-started
iterations instead of diverging at 8000.

On GPU, the 320x320 matmul at 25 iterations runs in ~50-100us.

References:
    Bishop et al. (ICRA 2024), "ReLU-QP: A GPU-Accelerated QP Solver for MPC"
    Section IV: uses "preconditioned condensed formulation" for MPC benchmarks

Author: Vrishabh Kenkre (CMU MS MechE)
"""

import numpy as np
import torch
import time
import sys
from scipy.linalg import solve_discrete_are
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ReLUQP-py"))
import reluqp.reluqpth as reluqp


class ReLUQP_MPC:
    """MPC using ReLU-QP with condensed (controls-only) formulation."""
    
    def __init__(self, Ad, Bd, Q_diag, R_diag, N,
                 u_min, u_max, x_min, x_max,
                 u_hover, gravity_offset,
                 device='auto', max_iter=4000, eps_abs=1e-3):
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]
        self.N = N
        self.u_hover = u_hover
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        Q = np.diag(Q_diag); R = np.diag(R_diag)
        P_t = solve_discrete_are(Ad, Bd, Q, R)
        nx, nu = self.nx, self.nu
        
        # ═══ Prediction matrices ═══
        Phi = [np.eye(nx)]
        for k in range(1, N+1):
            Phi.append(Ad @ Phi[-1])
        
        self.Phi_stack = np.zeros(((N+1)*nx, nx))
        for k in range(N+1):
            self.Phi_stack[k*nx:(k+1)*nx, :] = Phi[k]
        
        self.Gamma = np.zeros(((N+1)*nx, N*nu))
        self.gamma = np.zeros((N+1)*nx)
        for k in range(1, N+1):
            self.gamma[k*nx:(k+1)*nx] = Ad @ self.gamma[(k-1)*nx:k*nx] + gravity_offset
            for j in range(k):
                self.Gamma[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(Ad, k-1-j) @ Bd
        
        # ═══ Condensed cost ═══
        Q_stack = np.zeros(((N+1)*nx, (N+1)*nx))
        for k in range(N): Q_stack[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = Q
        Q_stack[N*nx:, N*nx:] = P_t
        
        self.R_stack = np.zeros((N*nu, N*nu))
        for k in range(N): self.R_stack[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = R
        
        self.H_c = self.Gamma.T @ Q_stack @ self.Gamma + self.R_stack
        self.GtQ = self.Gamma.T @ Q_stack
        self.u_hover_stack = np.tile(u_hover, N)
        
        # ═══ Control-bounds-only constraints ═══
        A_con = np.eye(N*nu)
        l_con = np.tile(u_min, N)
        u_con = np.tile(u_max, N)
        
        # ═══ Setup solver ═══
        self.solver = reluqp.ReLU_QP()
        self.solver.setup(
            H=torch.from_numpy(self.H_c).double(),
            g=torch.zeros(N*nu, dtype=torch.float64),
            A=torch.from_numpy(A_con).double(),
            l=torch.from_numpy(l_con).double(),
            u=torch.from_numpy(u_con).double(),
            rho=1.0, adaptive_rho=True, adaptive_rho_tolerance=5,
            max_iter=max_iter, eps_abs=eps_abs, check_interval=25,
            device=self.device, precision=torch.float64, verbose=False,
        )
        
        self.solve_times = []
        self.iterations_log = []
        
        n_var = N*nu; n_con = N*nu
        print(f"  ReLU-QP condensed: {n_var} vars, {n_con} constraints, "
              f"W={n_var+2*n_con}x{n_var+2*n_con}, device={self.device}")
    
    def solve(self, x_current, x_ref):
        """Solve MPC. x_ref: [nx x (N+1)]."""
        x_free = self.Phi_stack @ x_current + self.gamma
        x_ref_flat = x_ref.T.flatten()
        f_c = self.GtQ @ (x_free - x_ref_flat) - self.R_stack @ self.u_hover_stack
        
        self.solver.update(g=f_c)
        
        t0 = time.perf_counter()
        results = self.solver.solve()
        t_solve = time.perf_counter() - t0
        
        self.solve_times.append(t_solve)
        self.iterations_log.append(results.info.iter)
        
        u_opt = results.x.cpu().numpy()[:self.nu]
        return u_opt, {
            'status': results.info.status,
            'solve_time': t_solve,
            'iterations': results.info.iter,
        }


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))
    from quad_env import CrazyflieEnv
    from quad_dynamics import QuadParams, linearize_at_hover, discretize_dynamics
    from solver_admm_c import CADMMSolver
    
    def gen_fig8_ff(c, r, h, per, dur, dt):
        N=int(dur/dt); t=np.arange(N)*dt; w=2*np.pi/per; g=9.81
        ref=np.zeros((12,N)); ref[0]=c[0]+r*np.sin(w*t); ref[1]=c[1]+r*np.sin(2*w*t)/2; ref[2]=h
        ref[3]=r*w*np.cos(w*t); ref[4]=r*w*np.cos(2*w*t)
        ax=-r*w**2*np.sin(w*t); ay=-r*w**2*2*np.sin(2*w*t); ref[6]=-ay/g; ref[7]=ax/g
        return ref
    
    p=QuadParams(); dt=0.01; N=20
    Ac,Bc=linearize_at_hover(p); Ad,Bd=discretize_dynamics(Ac,Bc,dt)
    Q_d=np.array([300,300,300,10,10,10,3,3,1,0.1,0.1,0.1])
    R_d=np.array([30,1.5e3,1.5e3,1.5e3])
    uh=np.array([p.hover_thrust,0,0,0]); dg=(np.eye(12)-Ad)@np.zeros(12)-Bd@uh
    INF=1e10
    xlo=np.array([-INF]*3+[-INF]*3+[-np.radians(35)]*2+[-INF]*4)
    xhi=np.array([INF]*3+[INF]*3+[np.radians(35)]*2+[INF]*4)
    mp=str(Path(__file__).parent.parent/"mujoco_menagerie/bitcraze_crazyflie_2/scene.xml")
    
    dur=10.0; rd=dur+N*dt+1; ref=gen_fig8_ff(np.array([0.,0.]),0.5,1.0,4.0,rd,dt)
    Nr=ref.shape[1]; ts=int(dur/dt); skip=int(2/dt)
    
    for name, solver_cls, kwargs in [
        ("C ADMM", CADMMSolver, dict(Ad=Ad,Bd=Bd,Q_diag=Q_d,R_diag=R_d,N=N,
            u_min=p.u_min,u_max=p.u_max,x_min=xlo,x_max=xhi,u_hover=uh,
            gravity_offset=dg,rho=1.0,max_iter=200)),
        ("ReLU-QP", ReLUQP_MPC, dict(Ad=Ad,Bd=Bd,Q_diag=Q_d,R_diag=R_d,N=N,
            u_min=p.u_min,u_max=p.u_max,x_min=xlo,x_max=xhi,u_hover=uh,
            gravity_offset=dg)),
    ]:
        print(f"\n{'='*60}\n  {name} — Figure-8, 100Hz\n{'='*60}")
        s=solver_cls(**kwargs)
        env=CrazyflieEnv(model_path=mp,dt_sim=0.002,dt_ctrl=dt)
        x=env.reset(pos=ref[0:3,0]); xl=np.zeros((12,ts+1)); xl[:,0]=x
        for i in range(ts):
            rw=np.zeros((12,N+1))
            for k in range(N+1): rw[:,k]=ref[:,min(i+k,Nr-1)]
            u,info=s.solve(xl[:,i],rw); xl[:,i+1]=env.step(u)
            if hasattr(s,'warm_shift'): s.warm_shift()
        err=np.linalg.norm(xl[0:3,:ts]-ref[0:3,:ts],axis=0)
        ss=np.sqrt(np.mean(err[skip:]**2))
        st=np.median(np.array(s.solve_times))*1e6
        it=np.median(np.array(s.iterations_log))
        print(f"  SS-RMSE: {ss*1000:.1f}mm | Solve: {st:.0f}μs | Iters: {it:.0f}")
