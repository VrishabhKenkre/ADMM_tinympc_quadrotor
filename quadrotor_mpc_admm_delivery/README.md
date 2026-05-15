# quadrotor-mpc-admm

**Real-time MPC for a Crazyflie 2 quadrotor — hand-rolled C ADMM solver, OSQP / ReLU-QP benchmarks, and DAgger distillation to a 2.1 µs neural policy.**

---

## Headline results

| Controller | Tracking RMSE on figure-8 @ 1 m/s | Median solve time | Notes |
| --- | --- | --- | --- |
| C ADMM (this repo) | **3.7 mm** | **86 µs** | structure-exploiting, no deps |
| OSQP (sparse Cholesky) | 3.7 mm | 2,141 µs | **24.9× slower** at matched RMSE |
| ReLU-QP (GPU) | did not converge | — | wrong tool at n=12, see report |
| Distilled NN policy | 4.1 mm | **2.1 µs** | **41× faster** than the MPC it copies |

`results/benchmark_tuned_final.png` — full latency comparison.
`results/admm_c_fig8.png` — closed-loop tracking trace.

---

## What's in here

- **`src/admm_core.c`, `src/admm_core.h`** — ~400-line C ADMM solver with cached Riccati recursion. No dynamic allocation, no external deps, ports to STM32F4-class MCUs.
- **`src/solver_admm_c.py`** — `ctypes` Python binding for benchmarking and DAgger data collection.
- **`src/mpc_osqp.py`** — equivalent OSQP MPC for head-to-head comparison.
- **`src/solver_reluqp.py`** + **`reluqpth.py`** — ReLU-QP integration (REX Lab / Manchester group), with the patch needed for our ρ schedule.
- **`src/dagger.py`** — DAgger + DART training loop.
- **`src/policy_inference.c`, `src/policy_inference.h`** — pure-C deployment of the trained policy. 24 → 64 → 64 → 4 MLP, 6,276 params, 2.1 µs/forward-pass.
- **`src/quad_env.py`, `src/quad_dynamics.py`** — MuJoCo Menagerie Crazyflie 2 with corrected thrust/torque ranges, plus CasADi linearization (verified against autodiff, error = 0).
- **`src/visualize_live.py`** — `mujoco.viewer.launch_passive` live demo with crash recovery.
- **`report/quadrotor_mpc_admm_ieee.pdf`** — IEEE-format technical report covering everything above.

---

## Quickstart

```bash
# 1. Build the C solver
cd src && make            # produces libadmm_core.so

# 2. Run the verification + LQR baseline
python src/verify_system.py

# 3. Run the C ADMM MPC closed-loop
python src/visualize_live.py --solver admm_c

# 4. Reproduce the benchmark table
python src/benchmark.py   # writes results/benchmark_*.png

# 5. Train and export the distilled policy
python src/dagger.py --iters 5 --episodes 1500
python src/policy_inference.c   # auto-generated header from trained weights
```

Tested on Ubuntu 22.04, Python 3.10, MuJoCo 3.x. The C solver builds with any C99 compiler and has been compiled cleanly with `gcc-13` and `clang-16`.

---

## Why this stack

The Crazyflie has 192 KB of SRAM and a 168 MHz Cortex-M4. Off-the-shelf QP solvers (OSQP, qpOASES) target a different regime — sparse Cholesky on a 332-variable KKT system is overkill when the MPC has Riccati structure. **TinyMPC** (Nguyen et al., ICRA 2024 Best Paper in Automation) showed that caching the infinite-horizon LQR backward pass reduces per-iteration work to 12×12 block operations. This repo is a from-scratch C implementation of that idea, plus a head-to-head comparison with the two most-cited alternatives, plus the distillation step that removes the solver from deployment entirely.

The ReLU-QP comparison is a deliberate negative result: the GPU-parallel formulation is the right answer at n ≈ 50–100 state dimensions, not at the Crazyflie's n = 12. The repo documents the failure mode (dense weight matrix mixes thrust-cost R=30 and torque-cost R=1500 scales) and characterizes the crossover regime.

---

## Citations

If you use this code in academic work, please cite the underlying methods:

```bibtex
@inproceedings{nguyen2024tinympc,
  title={TinyMPC: Model-predictive control on resource-constrained microcontrollers},
  author={Nguyen, Anoushka and Schoedel, Sam and Alavilli, Achuthan and Plancher, Brian and Manchester, Zachary},
  booktitle={ICRA},
  year={2024}
}

@inproceedings{bishop2024reluqp,
  title={{ReLU-QP}: A {GPU}-accelerated quadratic programming solver for model-predictive control},
  author={Bishop, Arun and Br{\"u}digam, Jan and Manchester, Zachary},
  booktitle={ICRA},
  year={2024}
}

@inproceedings{ross2011dagger,
  title={A reduction of imitation learning and structured prediction to no-regret online learning},
  author={Ross, St{\'e}phane and Gordon, Geoffrey and Bagnell, Drew},
  booktitle={AISTATS},
  year={2011}
}
```

---

## License

MIT.

## Author

**Vrishabh Kenkre** — MS Mechanical Engineering, Robotics & Controls, Carnegie Mellon University. `vkenkre@andrew.cmu.edu`
