"""Generate the DART multi-seed variance plot."""
import json, sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 4-seed vanilla BC/DAgger results from earlier
with open('../results/vanilla_seed_variance.json') as f:
    vanilla = json.load(f)
# 5-seed DART results
with open('../results/dart_multiseed_state.json') as f:
    dart_state = json.load(f)

# Group: variant -> [(seed, ss_mm, survived), ...]
groups = {}
for r in vanilla + dart_state:
    v = r['variant']; sd = r['seed']
    ss = r.get('ss_mm', float('inf'))
    if isinstance(ss, str) and ss == 'inf': ss = float('inf')
    surv = r['survived']
    groups.setdefault(v, []).append((sd, ss, surv))

order = ['BC', 'BC + DART (fixed σ=0.1)', 'BC + DART (adaptive Σ̂)',
         'DAgger', 'DAgger + DART (fixed σ=0.1)', 'DAgger + DART (adaptive Σ̂)']

short_labels = ['BC', 'BC+DART\n(fixed σ)', 'BC+DART\n(adapt Σ̂)',
                'DAgger', 'DAgger+DART\n(fixed σ)', 'DAgger+DART\n(adapt Σ̂)']

colors = ['#cccccc', '#fdcc8a', '#fc8d59', '#bbbbbb', '#fec44f', '#d7301f']

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# === Left: box plot ===
ax = axes[0]
plot_data = []
labels_keep = []
crash_rates = []
for i, name in enumerate(order):
    rows = groups.get(name, [])
    if not rows: continue
    n = len(rows)
    crashes = sum(1 for _, ss, surv in rows if not surv)
    crash_rates.append((i, crashes, n))
    surv_vals = [ss for _, ss, surv in rows if surv]
    plot_data.append((short_labels[i], surv_vals, colors[i], n - crashes, n))

xs = np.arange(len(plot_data))
for x, (lbl, vals, c, n_surv, n_total) in zip(xs, plot_data):
    if vals:
        # Scatter individual seeds
        jitter = np.random.RandomState(0).uniform(-0.07, 0.07, len(vals))
        ax.scatter(x + jitter, vals, color=c, edgecolor='black', lw=0.6, s=85,
                   alpha=0.85, zorder=3)
        # Mean+range
        mu = np.mean(vals)
        mn, mx = np.min(vals), np.max(vals)
        ax.plot([x, x], [mn, mx], color=c, lw=2, alpha=0.6, zorder=2)
        ax.plot([x - 0.16, x + 0.16], [mu, mu], color='black', lw=2, zorder=4)
        # Annotate
        if mu > 100:
            ax.text(x, mu * 1.6, f'μ={mu:.0f}mm\n[{mn:.0f}-{mx:.0f}]\nsurv {n_surv}/{n_total}',
                    ha='center', fontsize=8.5, fontweight='bold')
        else:
            ax.text(x, mu * 1.6, f'μ={mu:.2f}mm\n[{mn:.2f}-{mx:.2f}]\nsurv {n_surv}/{n_total}',
                    ha='center', fontsize=8.5, fontweight='bold')
    else:
        ax.text(x, 5, f'all crashed\n0/{n_total}', ha='center', fontsize=10,
                fontweight='bold', color='red')

ax.set_yscale('log')
ax.set_xticks(xs); ax.set_xticklabels([d[0] for d in plot_data], fontsize=9, rotation=15, ha='right')
ax.axhline(3.72, color='blue', ls='--', lw=1.4, alpha=0.7, label='MPC teacher (3.72 mm)')
ax.set_ylabel('Steady-state RMSE [mm, log]')
ax.set_title('5-seed sweep — each dot = one training run\n(black bar = mean, colored bar = range)')
ax.set_ylim(1, 5000)
ax.grid(alpha=0.3, axis='y', which='both')
ax.legend(loc='upper right')

# === Right: zoomed DAgger+DART variants ===
ax = axes[1]
zoom_variants = ['DAgger + DART (fixed σ=0.1)', 'DAgger + DART (adaptive Σ̂)']
for i, name in enumerate(zoom_variants):
    rows = groups.get(name, [])
    seeds = [sd for sd, _, surv in rows if surv]
    vals = [ss for _, ss, surv in rows if surv]
    seeds_sorted = sorted(zip(seeds, vals))
    xs_local = np.arange(len(seeds_sorted))
    seeds_arr, vals_arr = zip(*seeds_sorted)
    ax.plot(xs_local + i * 0.18 - 0.09, vals_arr, 'o-',
            color=('#fec44f' if 'fixed' in name else '#d7301f'),
            ms=10, lw=1.5, label=short_labels[order.index(name)].replace('\n', ' '),
            markeredgecolor='black')

ax.axhline(3.72, color='blue', ls='--', lw=1.4, alpha=0.7, label='MPC teacher (3.72 mm)')
n_seeds = len([sd for sd, _, surv in groups['DAgger + DART (fixed σ=0.1)'] if surv])
ax.set_xticks(np.arange(n_seeds))
seed_labels = [str(sd) for sd, _, _ in sorted([r for r in groups['DAgger + DART (fixed σ=0.1)'] if r[2]])]
ax.set_xticklabels(seed_labels)
ax.set_xlabel('seed')
ax.set_ylabel('SS-RMSE [mm]')
ax.set_title('DAgger+DART variants: matches teacher within seed-to-seed noise')
ax.grid(alpha=0.3)
ax.legend(loc='upper right')

plt.tight_layout()
out = Path('../results/dart_variance.png')
plt.savefig(out, dpi=140, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# Print final table
print("\nFinal DART multi-seed table:")
print(f"  {'Variant':<35s} | {'survival':>8s} | {'mean':>7s} | {'min':>7s} | {'max':>7s} | {'std':>6s}")
print("  " + "-"*88)
for name in order:
    rows = groups.get(name, [])
    if not rows: continue
    surv_vals = [ss for _, ss, surv in rows if surv]
    total = len(rows)
    n_surv = len(surv_vals)
    if surv_vals:
        m = np.mean(surv_vals); mn = np.min(surv_vals); mx = np.max(surv_vals); sd = np.std(surv_vals)
        print(f"  {name:<35s} | {n_surv:>3d}/{total:<3d}  | {m:5.2f}mm | {mn:5.2f}mm | {mx:5.2f}mm | {sd:4.2f}mm")
    else:
        print(f"  {name:<35s} | {n_surv:>3d}/{total:<3d}  |    crashed |        |        |       ")
