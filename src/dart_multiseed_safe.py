"""
dart_multiseed_safe.py
----------------------
Runs all 4 DART variants across 5 seeds, persisting results to a JSON file
after EACH (seed, variant) combination. Re-runs skip combos already in the
JSON, so the script can be invoked repeatedly to resume after bash timeouts.

Usage:
    python dart_multiseed_safe.py [variant_prefix]

If variant_prefix is given (e.g. "BC" or "DAgger"), only matching variants run.
"""
import sys, os, json, time
sys.path.insert(0, '.')
import numpy as np
import torch

from dagger import CrazyflieTrackingEnv, MPCExpert, evaluate_policy
from dart_pipeline import run_variant

device = torch.device('cpu')
results_dir = "../results"
state_file = os.path.join(results_dir, "dart_multiseed_state.json")

SEEDS = [42, 7, 123, 2026, 999]
variants = [
    ('BC + DART (fixed σ=0.1)',     dict(use_dagger=False, use_dart=True,  dart_mode='fixed',    sigma_init=0.1)),
    ('BC + DART (adaptive Σ̂)',     dict(use_dagger=False, use_dart=True,  dart_mode='adaptive', sigma_init=0.1)),
    ('DAgger + DART (fixed σ=0.1)', dict(use_dagger=True,  use_dart=True,  dart_mode='fixed',    sigma_init=0.1)),
    ('DAgger + DART (adaptive Σ̂)', dict(use_dagger=True,  use_dart=True,  dart_mode='adaptive', sigma_init=0.1)),
]

# Load state
if os.path.exists(state_file):
    with open(state_file) as f:
        state = json.load(f)
    print(f"Loaded {len(state)} existing results from {state_file}")
else:
    state = []


def long_eval(env, expert, policy):
    p_errs, _, _, _ = evaluate_policy(env, policy, expert, device, "  eval")
    skip = int(2.0 / env.dt)
    p = np.array(p_errs)
    full = float(np.sqrt(np.mean(p**2)) * 1000)
    if len(p) > skip:
        return float(np.sqrt(np.mean(p[skip:]**2)) * 1000), full, True, len(p)
    return float('inf'), full, False, len(p)


def already_done(seed, variant):
    return any(r['seed'] == seed and r['variant'] == variant for r in state)


filter_prefix = sys.argv[1] if len(sys.argv) > 1 else ""

t_start = time.time()
n_runs = 0
for seed in SEEDS:
    for name, kw in variants:
        if filter_prefix and not name.startswith(filter_prefix):
            continue
        if already_done(seed, name):
            continue

        print(f"\n--- seed {seed} | {name} ---")
        np.random.seed(seed); torch.manual_seed(seed)
        env = CrazyflieTrackingEnv(dt=0.01, episode_length=10.0)
        expert = MPCExpert(dt=0.01)
        try:
            policy, _ = run_variant(name, env, expert, **kw)
            ss, full, surv, nsteps = long_eval(env, expert, policy)
            print(f"  >>> seed={seed} {name}: surv={surv} SS={ss:.2f}mm full={full:.2f}mm")
            state.append(dict(seed=int(seed), variant=name,
                              ss_mm=ss, full_mm=full, survived=bool(surv),
                              n_steps=int(nsteps)))
        except Exception as e:
            print(f"  !! FAILED seed={seed} {name}: {e}")
            state.append(dict(seed=int(seed), variant=name,
                              ss_mm=float('inf'), full_mm=float('inf'),
                              survived=False, n_steps=0, error=str(e)))

        # Persist after every variant in case of timeout
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        n_runs += 1

elapsed = time.time() - t_start
print(f"\n\n=== Completed {n_runs} new runs in {elapsed:.0f}s ===")
print(f"Total state entries: {len(state)}")

# Summary
print("\nSummary across seeds:")
print(f"  {'Variant':<35s} | {'n_surv/n':>8s} | {'mean SS':>9s} | {'min':>7s} | {'max':>7s}")
print("  " + "-"*72)
for name, _ in variants:
    if filter_prefix and not name.startswith(filter_prefix):
        continue
    rows = [r for r in state if r['variant'] == name]
    survived = [r for r in rows if r['survived']]
    if not rows: continue
    ss_vals = [r['ss_mm'] for r in survived]
    if ss_vals:
        m, mn, mx = np.mean(ss_vals), np.min(ss_vals), np.max(ss_vals)
        ss_str = f"{m:6.2f}mm"; mn_s = f"{mn:5.2f}"; mx_s = f"{mx:5.2f}"
    else:
        ss_str = "crashed"; mn_s = "-"; mx_s = "-"
    print(f"  {name:<35s} | {len(survived):>3d}/{len(rows):<3d} | {ss_str:>9s} | {mn_s:>7s} | {mx_s:>7s}")
