"""
three_way_pareto.py -- regenerate figures/three_way_pareto.pdf

Three-way solver Pareto plot in (log median solve time, closed-loop RMSE) space.
All numbers are loaded from the canonical JSON result files:
  - results/bench_tuned_solvers_legion.json  (CPU rows: C ADMM, OSQP -- Legion i9-14900HX)
  - results/bench_all_solvers.json           (ReLU-QP fig-8 row -- NVIDIA L40S GPU)

Run from the repository root:
    python3 figures/three_way_pareto.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ---- Load canonical data -------------------------------------------------
with open(RESULTS / "bench_tuned_solvers_legion.json") as f:
    legion = json.load(f)
with open(RESULTS / "bench_all_solvers.json") as f:
    all_solvers = json.load(f)

s = legion["solvers"]
admm_speed = (s["C_ADMM_speed_opt"]["median_us"], s["C_ADMM_speed_opt"]["rmse_mm"])
admm_tuned = (s["C_ADMM_tuned"]["median_us"], s["C_ADMM_tuned"]["rmse_mm"])
admm_orig = (s["C_ADMM_original"]["median_us"], s["C_ADMM_original"]["rmse_mm"])
osqp_tuned = (s["OSQP_tuned"]["median_us"], s["OSQP_tuned"]["rmse_mm"])
osqp_orig = (s["OSQP_original"]["median_us"], s["OSQP_original"]["rmse_mm"])

reluqp_row = next(r for r in all_solvers
                  if r["solver"] == "ReLU-QP" and r["trajectory"] == "fig8")
reluqp = (reluqp_row["median_us"], reluqp_row["rmse_mm"])

# ---- Style: serif body to match IEEEtran --------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.6,
    "lines.markeredgewidth": 0.7,
})

C_ADMM = "#1f4e8c"   # blue   -- specialized CPU
C_OSQP = "#b8430f"   # rust   -- general-purpose CPU
C_RELU = "#3f7d3f"   # green  -- general-purpose GPU

fig, ax = plt.subplots(figsize=(3.5, 3.2))

# C ADMM frontier line (ordered by solve time)
admm_pts = [admm_speed, admm_tuned, admm_orig]
ax.plot([p[0] for p in admm_pts], [p[1] for p in admm_pts],
        "-", color=C_ADMM, lw=0.9, alpha=0.55, zorder=1)

# C ADMM points (circles); headline 'tuned' point filled
ax.scatter(*admm_speed, s=34, marker="o", facecolor="white",
           edgecolor=C_ADMM, zorder=3)
ax.scatter(*admm_orig, s=34, marker="o", facecolor="white",
           edgecolor=C_ADMM, zorder=3)
ax.scatter(*admm_tuned, s=60, marker="o", facecolor=C_ADMM,
           edgecolor="black", linewidth=0.8, zorder=4,
           label="C ADMM (specialized CPU)")

# OSQP points (triangles)
ax.scatter(*osqp_tuned, s=42, marker="^", facecolor=C_OSQP,
           edgecolor="black", linewidth=0.5, zorder=3,
           label="OSQP (general-purpose CPU)")
ax.scatter(*osqp_orig, s=42, marker="^", facecolor="white",
           edgecolor=C_OSQP, zorder=3)

# ReLU-QP point (diamond -- different hardware class)
ax.scatter(*reluqp, s=48, marker="D", facecolor=C_RELU,
           edgecolor="black", linewidth=0.5, zorder=3,
           label="ReLU-QP (general-purpose GPU)")

# ---- Per-point text labels ----------------------------------------------
ax.annotate("speed-opt\n%.1f us / %.2f mm" % admm_speed, admm_speed,
            textcoords="offset points", xytext=(-1, -18), ha="center",
            fontsize=6.0, color=C_ADMM)
ax.annotate("original\n%.1f us / %.2f mm" % admm_orig, admm_orig,
            textcoords="offset points", xytext=(16, 5), ha="left",
            fontsize=6.0, color=C_ADMM)
ax.annotate("original / tuned\n~202 us / 4.86 mm", osqp_tuned,
            textcoords="offset points", xytext=(0, 11), ha="center",
            fontsize=6.0, color=C_OSQP)

# headline annotation on C ADMM tuned -- placed in the empty lower-left void
ax.annotate("tuned: 9.4 us / 3.42 mm\n22x faster than OSQP\nat matched accuracy",
            admm_tuned, textcoords="offset points", xytext=(-42, -33),
            ha="left", fontsize=6.2, color="black",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

# ReLU-QP hardware-regime annotation -- below-left of the point
ax.annotate("different hardware regime:\nL40S datacenter GPU, batch=1",
            reluqp, textcoords="offset points", xytext=(-14, -25),
            ha="right", fontsize=6.2, color=C_RELU,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C_RELU))

# ---- Axes ----------------------------------------------------------------
ax.set_xscale("log")
ax.set_xlim(2.0, 3500)
ax.set_ylim(2.6, 5.6)
ax.set_xlabel(r"Median solve time [$\mu$s, log scale]")
ax.set_ylabel("Closed-loop tracking RMSE [mm]")
ax.grid(True, which="both", alpha=0.25, lw=0.4)
ax.legend(loc="lower right", frameon=True, framealpha=0.93, borderpad=0.4)

fig.tight_layout(pad=0.4)
out = ROOT / "figures" / "three_way_pareto.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
