"""Quick variance check on vanilla BC and vanilla DAgger only — confirms
that the 'crashes' result with seed=42 is real seed-sensitivity, not a bug."""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np
import torch

from dagger import CrazyflieTrackingEnv, MPCExpert, evaluate_policy
from dart_pipeline import run_variant

device = torch.device('cpu')

SEEDS = [7, 123, 2026]   # excludes seed=42 since we already have those

variants = [
    ('BC',     dict(use_dagger=False, use_dart=False)),
    ('DAgger', dict(use_dagger=True,  use_dart=False)),
]


def long_eval(env, policy, expert):
    p_errs, _, _, _ = evaluate_policy(env, policy, expert, device, "  eval")
    skip = int(2.0 / env.dt)
    p = np.array(p_errs)
    full = np.sqrt(np.mean(p**2)) * 1000
    if len(p) > skip:
        return float(np.sqrt(np.mean(p[skip:]**2)) * 1000), float(full), True, len(p)
    return float('inf'), float(full), False, len(p)


rows = []
for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    np.random.seed(seed); torch.manual_seed(seed)
    env = CrazyflieTrackingEnv(dt=0.01, episode_length=10.0)
    expert = MPCExpert(dt=0.01)
    for name, kw in variants:
        policy, _ = run_variant(name, env, expert, **kw)
        ss, full, surv, nsteps = long_eval(env, policy, expert)
        rows.append(dict(seed=seed, variant=name, ss_mm=ss, full_mm=full,
                          survived=surv, n_steps=nsteps))
        print(f"  >>> {name} seed={seed}: survived={surv} SS={ss:.2f}mm full={full:.2f}mm steps={nsteps}")

# Combine with seed=42 (which we already had)
prev_seed42 = [
    dict(seed=42, variant='BC',     ss_mm=float('inf'), full_mm=118.1, survived=False, n_steps=None),
    dict(seed=42, variant='DAgger', ss_mm=float('inf'), full_mm=58.8,  survived=False, n_steps=None),
]
rows = prev_seed42 + rows

# Group + summarize
print("\n" + "="*72)
print("  Vanilla BC / DAgger seed sensitivity")
print("="*72)
print(f"  {'Variant':<12s} | seed | survived? | SS-RMSE | Full RMSE")
print(f"  {'-'*12}-+-{'-'*4}-+-{'-'*9}-+-{'-'*7}-+-{'-'*10}")
for r in sorted(rows, key=lambda x: (x['variant'], x['seed'])):
    ss = f"{r['ss_mm']:.1f}mm" if r['survived'] else "inf"
    print(f"  {r['variant']:<12s} | {r['seed']:>4d} | {'yes' if r['survived'] else 'NO ':>8s} | {ss:>7s} | {r['full_mm']:7.1f}mm")

# Save
with open('../results/vanilla_seed_variance.json', 'w') as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\n  Saved ../results/vanilla_seed_variance.json")
