"""Generate the headline DART comparison plot."""
import json, sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

with open('../results/dart_summary.json') as f:
    seed42 = json.load(f)
with open('../results/vanilla_seed_variance.json') as f:
    vanilla = json.load(f)

# Build the table
data = {
    'BC':                          {'seeds': {}, 'order': 0, 'color': '#cccccc', 'group': 'no DART'},
    'BC + DART (fixed σ=0.1)':     {'seeds': {}, 'order': 1, 'color': '#fdcc8a', 'group': 'fixed σ'},
    'BC + DART (adaptive Σ̂)':     {'seeds': {}, 'order': 2, 'color': '#fc8d59', 'group': 'adaptive Σ̂'},
    'DAgger':                      {'seeds': {}, 'order': 3, 'color': '#bbbbbb', 'group': 'no DART'},
    'DAgger + DART (fixed σ=0.1)': {'seeds': {}, 'order': 4, 'color': '#fec44f', 'group': 'fixed σ'},
    'DAgger + DART (adaptive Σ̂)': {'seeds': {}, 'order': 5, 'color': '#d7301f', 'group': 'adaptive Σ̂'},
}

# Fill in seed=42 results from dart_summary.json
for variant_name, ssp, sse, t in seed42['summary']:
    if variant_name in data:
        data[variant_name]['seeds'][42] = ssp
# Fill multi-seed for BC and DAgger
for r in vanilla:
    if r['variant'] in data:
        data[r['variant']]['seeds'][r['seed']] = float(r['ss_mm']) if r['survived'] else float('inf')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# === Left: bar chart of variants ===
ax = axes[0]
names_ordered = sorted(data.keys(), key=lambda k: data[k]['order'])
xs = np.arange(len(names_ordered))
heights = []
colors = []
errbars_low = []
errbars_high = []
crashed_marker = []

for name in names_ordered:
    seeds_vals = list(data[name]['seeds'].values())
    survived = [v for v in seeds_vals if np.isfinite(v)]
    if survived:
        mean_v = np.mean(survived)
        heights.append(mean_v)
        errbars_low.append(mean_v - min(survived))
        errbars_high.append(max(survived) - mean_v)
    else:
        heights.append(0)
        errbars_low.append(0)
        errbars_high.append(0)
    colors.append(data[name]['color'])
    n_total = len(seeds_vals)
    n_crash = sum(1 for v in seeds_vals if not np.isfinite(v))
    crashed_marker.append(f"{n_total - n_crash}/{n_total}")

bars = ax.bar(xs, heights, color=colors, edgecolor='black', lw=0.6, alpha=0.92,
              yerr=[errbars_low, errbars_high], capsize=4, ecolor='gray')
ax.axhline(3.72, color='blue', ls='--', lw=1.2, alpha=0.7, label='MPC teacher (3.72 mm)')
ax.set_yscale('log')

short_labels = ['BC', 'BC + DART\n(fixed σ)', 'BC + DART\n(adapt Σ̂)',
                'DAgger', 'DAgger + DART\n(fixed σ)', 'DAgger + DART\n(adapt Σ̂)']
ax.set_xticks(xs); ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha='right')
ax.set_ylabel('Steady-state RMSE [mm, log]')
ax.set_title('Imitation learning variant comparison\n(mean across survived seeds; vanilla shows mean+range)')
ax.grid(alpha=0.3, axis='y', which='both')
ax.legend(loc='upper right')

# Annotate each bar
for i, name in enumerate(names_ordered):
    seeds_vals = list(data[name]['seeds'].values())
    survived = [v for v in seeds_vals if np.isfinite(v)]
    n_total = len(seeds_vals)
    n_surv = len(survived)
    if survived:
        mean_v = np.mean(survived)
        if mean_v > 100:
            label = f"{mean_v:.0f} mm\nsurv {n_surv}/{n_total}"
        else:
            label = f"{mean_v:.1f} mm\nsurv {n_surv}/{n_total}"
        ax.text(i, mean_v * (1.5 + errbars_high[i] / max(1, mean_v) * 0.5), label,
                ha='center', fontsize=9, fontweight='bold')
    else:
        ax.text(i, 5, f"all crashed\n0/{n_total}", ha='center', fontsize=10,
                fontweight='bold', color='red')

ax.set_ylim(1, 5000)

# === Right: per-seed dots for BC and DAgger ===
ax = axes[1]
v_names = ['BC', 'DAgger']
seeds = sorted(set(data['BC']['seeds'].keys()) | set(data['DAgger']['seeds'].keys()))
n_seeds = len(seeds)
for j, vname in enumerate(v_names):
    for i, sd in enumerate(seeds):
        val = data[vname]['seeds'].get(sd)
        x = j + (i - (n_seeds - 1) / 2) * 0.18
        if val is None:
            continue
        if not np.isfinite(val):
            ax.scatter(x, 0.5, marker='x', color='red', s=140, lw=2.5)
            ax.text(x, 1.0, f's{sd}\nCRASH', ha='center', fontsize=8, color='red')
        else:
            ax.scatter(x, val, marker='o', s=120, color='C0', edgecolor='black', lw=0.6)
            ax.text(x, val * 1.15, f's{sd}\n{val:.0f}mm', ha='center', fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(v_names)
ax.set_yscale('log')
ax.set_ylabel('SS-RMSE [mm, log scale]')
ax.set_title('Vanilla BC / DAgger: seed sensitivity\n(without DART)')
ax.grid(alpha=0.3, axis='y', which='both')
ax.axhline(3.72, color='blue', ls='--', lw=1.2, alpha=0.7, label='MPC teacher')
ax.legend(loc='upper right')

plt.tight_layout()
out = Path('../results/dart_comparison.png')
plt.savefig(out, dpi=140, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
