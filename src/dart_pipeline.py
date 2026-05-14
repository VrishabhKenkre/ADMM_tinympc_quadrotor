"""
dart_pipeline.py — Six-way comparison of imitation learning variants for the
quadrotor MPC distillation problem.

Variants
--------
  BC                  : behavioral cloning only (5 expert episodes)
  BC + DART(fixed)    : BC with fixed-sigma action-space noise during demonstration
  BC + DART(adapt)    : BC with adaptive Sigma_hat estimated from BC residuals
                        on expert states (Laskey et al. CoRL 2017, Alg 1)
  DAgger              : standard DAgger (5 iters x 3 episodes, student rollouts,
                        expert relabel)
  DAgger + DART(fixed): like DAgger but expert rollouts are noisy (sigma fixed)
  DAgger + DART(adapt): like DAgger but expert rollouts inject Sigma_hat that
                        is updated each iteration from current student error

Reference
---------
Laskey, M., Lee, J., Fox, R., Dragan, A., Goldberg, K. (2017). "DART: Noise
Injection for Robust Imitation Learning." CoRL 2017.

The covariance update is the maximum-likelihood estimate of the diagonal action
covariance from student-vs-expert residuals on the current dataset:

  Sigma_hat = (1 / N) * sum_i (a_expert_i - a_student_i)^2     (element-wise)

When student is good, sigma shrinks; when student is bad, sigma is large. This
moves the data distribution toward states the student will visit at test time
without requiring on-policy student rollouts (which DAgger needs).

Hyper-parameters held fixed
---------------------------
  5 expert episodes (per cycle)
  3 dagger episodes per iter (for variants that use student rollouts)
  100 BC epochs / 50 dagger epochs
  Adam lr 1e-3, batch 256
  Same env, same expert (tuned C ADMM), same evaluation
"""
import sys, os, time, json
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from dagger import (CrazyflieTrackingEnv, PolicyNet, MPCExpert,
                     train_policy, evaluate_policy)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')
results_dir = Path('../results')
results_dir.mkdir(exist_ok=True)


def collect_expert_episodes(env, expert, n_episodes, noise=None, student=None):
    """Roll the MPC expert; optionally inject action-space Gaussian noise (DART).

    Args
    ----
    noise : None | float | np.ndarray of shape (4,)
        If float, isotropic sigma on normalized action.
        If array, per-dim sigma (Sigma_hat diagonal).
        If None, no noise (vanilla expert demonstrations).
    """
    obs_list, act_list = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            ref_w = env.get_ref_window()
            u_expert = expert.get_action(env.state, ref_w)
            a_expert = env._control_to_action(u_expert).astype(np.float32)
            # DART: inject noise on the action that gets EXECUTED in env,
            # but LABEL the dataset with the clean expert action.
            if noise is not None:
                if np.isscalar(noise):
                    a_play = a_expert + np.random.randn(4).astype(np.float32) * noise
                else:
                    a_play = a_expert + np.random.randn(4).astype(np.float32) * noise.astype(np.float32)
                a_play = np.clip(a_play, -1.0, 1.0)
            else:
                a_play = a_expert
            obs_list.append(obs.copy())
            act_list.append(a_expert.copy())   # label = clean expert
            obs, _, terminated, truncated, _ = env.step(a_play)
            done = terminated or truncated
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def collect_dagger_episodes(env, expert, policy, n_episodes):
    """Roll the STUDENT; query the expert at visited states. (Standard DAgger.)"""
    obs_list, act_list = [], []
    policy.eval()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                ref_w = env.get_ref_window()
                u_expert = expert.get_action(env.state, ref_w)
                a_expert = env._control_to_action(u_expert).astype(np.float32)
                a_student = policy(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
                obs_list.append(obs.copy())
                act_list.append(a_expert.copy())
                obs, _, terminated, truncated, _ = env.step(a_student)
                done = terminated or truncated
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def estimate_sigma_hat(obs_data, act_data, policy):
    """Diagonal Sigma_hat = MLE from (a_expert - a_student)^2 over the dataset.
    Returns shape (4,)."""
    policy.eval()
    with torch.no_grad():
        a_student = policy(torch.from_numpy(obs_data)).numpy()
    diff = act_data - a_student
    return np.sqrt(np.mean(diff**2, axis=0)).astype(np.float32)  # std-dev per dim


# ─────────────────────────────────────────────────────────────────
# Runner

def make_policy():
    p = PolicyNet(obs_dim=20, act_dim=4, hidden=64).to(device)
    return p


def eval_short(env, policy, expert):
    """Return (ss_rmse_policy_mm, ss_rmse_expert_mm) over one episode."""
    p_errs, e_errs, _, _ = evaluate_policy(env, policy, expert, device, "  eval")
    skip = int(2.0 / env.dt)
    p_arr = np.array(p_errs); e_arr = np.array(e_errs)
    return (np.sqrt(np.mean(p_arr[skip:]**2)) * 1000 if len(p_arr) > skip else np.inf,
            np.sqrt(np.mean(e_arr[skip:]**2)) * 1000 if len(e_arr) > skip else np.inf)


def run_variant(name, env, expert, *,
                use_dart=False, dart_mode='fixed', sigma_init=0.1,
                use_dagger=False,
                n_expert_eps=5, n_dagger_eps=3, n_dagger_iters=5,
                bc_epochs=100, dagger_epochs=50):
    print(f"\n{'='*72}\n  Variant: {name}\n{'='*72}")
    policy = make_policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    history = {'name': name, 'rmse_per_iter': [], 'loss_per_iter': [],
               'sigmas': []}

    # ─── Phase 1: Expert demonstrations (possibly DART-noisy) ───
    sigma = sigma_init if use_dart else None
    if use_dart and dart_mode == 'adaptive':
        # Start with fixed sigma; will refine after BC
        sigma_val = sigma_init
        noise_arg = sigma_val
    elif use_dart and dart_mode == 'fixed':
        noise_arg = sigma_init
    else:
        noise_arg = None

    print(f"  Step 1: collecting {n_expert_eps} expert episodes "
          f"(noise={noise_arg if noise_arg is None else float(noise_arg) if np.isscalar(noise_arg) else 'vec'})")
    t0 = time.time()
    obs_buf, act_buf = collect_expert_episodes(env, expert, n_expert_eps, noise=noise_arg)
    print(f"  Got {len(obs_buf)} samples ({time.time()-t0:.1f}s)")

    # ─── Phase 2: Behavioral cloning ───
    print(f"  Step 2: BC for {bc_epochs} epochs ...")
    losses = train_policy(policy, opt, obs_buf, act_buf, device,
                          epochs=bc_epochs, batch_size=256)
    history['loss_per_iter'].append(float(losses[-1]))
    ss_p, ss_e = eval_short(env, policy, expert)
    history['rmse_per_iter'].append((ss_p, ss_e))
    print(f"  After BC | Policy: {ss_p:.1f}mm | Expert: {ss_e:.1f}mm | loss {losses[-1]:.4f}")

    if not use_dagger:
        # BC-only variant: optionally do a second BC round with adaptive Sigma_hat
        if use_dart and dart_mode == 'adaptive':
            sigma_hat = estimate_sigma_hat(obs_buf, act_buf, policy)
            history['sigmas'].append(sigma_hat.tolist())
            print(f"  Sigma_hat = {sigma_hat}")
            print(f"  Collecting {n_expert_eps} extra episodes with adaptive sigma ...")
            obs_extra, act_extra = collect_expert_episodes(env, expert, n_expert_eps, noise=sigma_hat)
            obs_buf = np.concatenate([obs_buf, obs_extra])
            act_buf = np.concatenate([act_buf, act_extra])
            losses = train_policy(policy, opt, obs_buf, act_buf, device,
                                  epochs=bc_epochs // 2, batch_size=256)
            history['loss_per_iter'].append(float(losses[-1]))
            ss_p, ss_e = eval_short(env, policy, expert)
            history['rmse_per_iter'].append((ss_p, ss_e))
            print(f"  After BC+DART(adapt) | Policy: {ss_p:.1f}mm | loss {losses[-1]:.4f}")
        return policy, history

    # ─── Phase 3 (DAgger): iterate student-rollout + expert-relabel ───
    for k in range(n_dagger_iters):
        print(f"\n  DAgger iter {k+1}/{n_dagger_iters}")
        if use_dart and dart_mode == 'adaptive':
            # Update Sigma_hat from current dataset
            sigma_hat = estimate_sigma_hat(obs_buf, act_buf, policy)
            history['sigmas'].append(sigma_hat.tolist())
            print(f"    Sigma_hat = {sigma_hat}")
            # Get more expert demos with noise = Sigma_hat
            obs_new, act_new = collect_expert_episodes(env, expert, n_dagger_eps, noise=sigma_hat)
        elif use_dart and dart_mode == 'fixed':
            obs_new, act_new = collect_expert_episodes(env, expert, n_dagger_eps, noise=sigma_init)
        else:
            # Plain DAgger: roll student, label with expert
            obs_new, act_new = collect_dagger_episodes(env, expert, policy, n_dagger_eps)

        obs_buf = np.concatenate([obs_buf, obs_new])
        act_buf = np.concatenate([act_buf, act_new])
        losses = train_policy(policy, opt, obs_buf, act_buf, device,
                              epochs=dagger_epochs, batch_size=256)
        history['loss_per_iter'].append(float(losses[-1]))
        ss_p, ss_e = eval_short(env, policy, expert)
        history['rmse_per_iter'].append((ss_p, ss_e))
        print(f"    Iter {k+1} | Policy: {ss_p:.1f}mm | Expert: {ss_e:.1f}mm | dataset={len(obs_buf)}")

    return policy, history


# ──────────────────────────────────────────────────────────────────
# Run all 6 variants
if __name__ == '__main__':
    env = CrazyflieTrackingEnv(dt=0.01, episode_length=10.0)
    expert = MPCExpert(dt=0.01)

    variants = [
        ('BC',                       dict(use_dagger=False, use_dart=False)),
        ('BC + DART (fixed σ=0.1)',  dict(use_dagger=False, use_dart=True, dart_mode='fixed', sigma_init=0.1)),
        ('BC + DART (adaptive Σ̂)',  dict(use_dagger=False, use_dart=True, dart_mode='adaptive', sigma_init=0.1)),
        ('DAgger',                   dict(use_dagger=True,  use_dart=False)),
        ('DAgger + DART (fixed σ=0.1)', dict(use_dagger=True, use_dart=True, dart_mode='fixed', sigma_init=0.1)),
        ('DAgger + DART (adaptive Σ̂)', dict(use_dagger=True, use_dart=True, dart_mode='adaptive', sigma_init=0.1)),
    ]

    summary = []
    histories = []
    for name, kwargs in variants:
        t0 = time.time()
        policy, hist = run_variant(name, env, expert, **kwargs)
        elapsed = time.time() - t0
        final_p, final_e = hist['rmse_per_iter'][-1]
        summary.append((name, final_p, final_e, elapsed))
        histories.append(hist)
        # Save checkpoint
        slug = name.replace(' ', '_').replace('+', 'p').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace('σ', 's').replace('Σ̂', 'Sh').replace('.', '')
        torch.save(policy.state_dict(), results_dir / f'policy_{slug}.pt')
        print(f"\n  Saved {results_dir / f'policy_{slug}.pt'}  (variant took {elapsed:.0f}s)")

    # ─── Print summary ───
    print("\n" + "="*72)
    print("  FINAL SUMMARY (steady-state RMSE after 2s warmup)")
    print("="*72)
    print(f"  {'Variant':<35s} | {'Policy':>9s} | {'Expert':>9s} | Time")
    print(f"  {'-'*35}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    for name, ssp, sse, t in summary:
        print(f"  {name:<35s} |  {ssp:6.2f}mm |  {sse:6.2f}mm | {t:5.0f}s")

    with open(results_dir / 'dart_summary.json', 'w') as f:
        json.dump({'summary': summary,
                   'histories': histories}, f, indent=2, default=float)
    print(f"\n  Saved {results_dir / 'dart_summary.json'}")
