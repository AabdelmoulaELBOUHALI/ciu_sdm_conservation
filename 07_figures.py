"""
Step 7: Figures of the article
==============================
Builds the result figures from results/robustness/ and results/ (run
06_robustness_analysis.py with --part summarise first):

  figures/Figure_3  AUC of the nine model families over the ten species
                    (repeated spatial block CV)
  figures/Figure_4  AUC per species and model family (repeated spatial block CV)
  figures/Figure_5  single-feature faithfulness of the four explanation methods
                    (all records), with the Scott-Knott groups and the mean of
                    the random rankings; mean run time per combination
  figures/Figure_6  mean Kendall's tau between the global rankings of the four
                    methods
  figures/Figure_7  CIU ecological roles per species (CIU on all records)
  figures/Figure_8  mean faithfulness with Scott-Knott groups, and Borda rank
                    sums over faithfulness, monotonicity and run time

Each figure is written as vector PDF and EPS at its printed width (190 mm for
full-width figures, 90 mm for Figure_6), with Arial lettering and hatching in
addition to colour. Figures 1 and 2 (workflow and role grid) are drawn in LaTeX.

With --png, PNG copies at 1000 dpi are also written to figures/png/.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ROBUST = os.path.join(RESULTS, "robustness")
FIG_DIR = os.path.join(HERE, "figures")

MM = 1 / 25.4
FULL_W = 190 * MM
SINGLE_W = 90 * MM
INK, INK2, MUTED, GRID = "#0b0b0b", "#3d3c3a", "#6b6a66", "#e1e0d9"
XAI_COL = {"CIU": "#2a78d6", "SHAP": "#1baf7a", "LIME": "#eda100", "ALE": "#eb6834"}
ENS, BASE = "#2a78d6", "#9ec5f4"
ROLE_COL = {"OPTIMAL": "#0ca30c", "LIMITING_FACTOR": "#d03b3b", "LOW_INFLUENCE": "#b9b7b0"}
ROLE_HATCH = {"OPTIMAL": "", "LIMITING_FACTOR": "////", "LOW_INFLUENCE": ".."}
ROLE_LABEL = {"OPTIMAL": "Optimal driver", "LIMITING_FACTOR": "Limiting factor", "LOW_INFLUENCE": "Low influence"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MODEL_ORDER = ["VotingEnsemble", "RandomForest", "StackingEnsemble", "XGBoost", "LightGBM", "SVM",
               "BaggingEnsemble", "MLP", "GAM"]
MODEL_SHORT = {"VotingEnsemble": "Voting", "RandomForest": "RF", "StackingEnsemble": "Stacking",
               "XGBoost": "XGBoost", "LightGBM": "LightGBM", "SVM": "SVM", "BaggingEnsemble": "Bagging",
               "MLP": "MLP", "GAM": "GAM"}
ENSEMBLES = {"VotingEnsemble", "StackingEnsemble", "BaggingEnsemble"}
METHOD_ORDER = ["SHAP", "CIU", "LIME", "ALE"]

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": "#9c9b93", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}


def species_name(key):
    return key.replace("_", " ")


def clean(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name, png=False):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    if png:
        os.makedirs(os.path.join(FIG_DIR, "png"), exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "png", f"{name}.png"), dpi=1000)
    plt.close(fig)
    print("  wrote", os.path.relpath(os.path.join(FIG_DIR, name), HERE))


# ------------------------------------------------------------------ data ---
def load():
    perf = pd.read_csv(os.path.join(ROBUST, "repeated_cv_performance.csv"))
    ev = pd.read_csv(os.path.join(ROBUST, "full_data_xai_evaluation.csv"))
    quality = pd.read_csv(os.path.join(ROBUST, "full_data_xai_quality.csv")).set_index("xai_method")
    roles = pd.read_csv(os.path.join(ROBUST, "ciu_ecological_roles_all_records.csv"))
    timing = pd.read_csv(os.path.join(RESULTS, "xai_comparison_metrics.csv"))
    agree = pd.read_csv(os.path.join(RESULTS, "xai_ranking_agreement.csv"))
    assert len(perf) == 90 and len(roles) == 657 and (ev.xai_method != "RANDOM").sum() == 360
    return perf, ev, quality, roles, timing, agree


# --------------------------------------------------------------- figures ---
def auc_box(perf, ax):
    order = perf.groupby("model").AUC_mean.mean().reindex(MODEL_ORDER).sort_values(ascending=False).index
    clean(ax)
    for i, m in enumerate(order):
        vals = perf.loc[perf.model == m, "AUC_mean"].values
        ens = m in ENSEMBLES
        bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=MUTED, linewidth=0.8),
                        capprops=dict(color=MUTED, linewidth=0.8), flierprops=dict(marker=""))
        for b in bp["boxes"]:
            b.set(facecolor=ENS if ens else BASE, edgecolor=INK2, linewidth=0.5, hatch="////" if ens else "")
        jit = (np.random.RandomState(i).rand(len(vals)) - 0.5) * 0.22
        ax.scatter(np.full(len(vals), i) + jit, vals, s=7, color=INK, zorder=3, linewidths=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in order])
    ax.set_ylabel("AUC")
    ax.legend(handles=[Patch(facecolor=ENS, edgecolor=INK2, hatch="////", label="Ensemble"),
                       Patch(facecolor=BASE, edgecolor=INK2, label="Base learner")],
              loc="lower left", frameon=False, handlelength=1.4)


def vector_heatmap(fig, ax, values, vmin, vmax, label, **cb_kw):
    """Heatmap drawn as vector cells (pcolormesh) with a vector colour bar, so
    the PDF/EPS contain no bitmap."""
    n_rows, n_cols = values.shape
    im = ax.pcolormesh(np.arange(n_cols + 1) - 0.5, np.arange(n_rows + 1) - 0.5, values, cmap=SEQ_BLUE,
                       vmin=vmin, vmax=vmax, edgecolors="white", linewidth=1.5)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)          # first row at the top
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, **cb_kw)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")          # no seams between colour-bar segments
    cb.set_label(label)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)


def auc_heatmap(perf, fig, ax):
    mat = perf.pivot(index="species", columns="model", values="AUC_mean").reindex(columns=MODEL_ORDER)
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    vmin, vmax = np.nanmin(mat.values), np.nanmax(mat.values)
    vector_heatmap(fig, ax, mat.values, vmin, vmax, "Mean AUC", fraction=0.03, pad=0.015)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([species_name(s) for s in mat.index], style="italic")
    thr = vmin + 0.55 * (vmax - vmin)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5, color="white" if v > thr else INK)


def faithfulness_box(ev, quality, ax):
    """Single-feature faithfulness over the 90 combinations; the dashed line is
    the mean faithfulness of the random rankings."""
    clean(ax)
    rnd = ev[ev.xai_method == "RANDOM"].groupby(["species", "model"]).faithfulness.mean().mean()
    ax.axhline(rnd, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)
    for i, m in enumerate(METHOD_ORDER):
        vals = ev.loc[ev.xai_method == m, "faithfulness"].values
        bp = ax.boxplot([vals], positions=[i], widths=0.55, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=MUTED, linewidth=0.8),
                        capprops=dict(color=MUTED, linewidth=0.8),
                        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none",
                                        markeredgecolor=MUTED, markeredgewidth=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=XAI_COL[m], edgecolor=INK2, linewidth=0.5)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel("Faithfulness")
    ax.set_ylim(-0.8, 1.17)
    groups = quality.loc[METHOD_ORDER, "cluster"].astype(int)       # Scott-Knott groups (all records)
    for g in sorted(groups.unique()):
        idx = [i for i, m in enumerate(METHOD_ORDER) if groups[m] == g]
        ax.plot([min(idx) - 0.25, max(idx) + 0.25], [1.06, 1.06], color=MUTED, linewidth=0.6)
        ax.text(np.mean(idx), 1.08, f"group {g}", ha="center", fontsize=7.5, color=INK2)


def roles_bar(roles, ax):
    order = ["OPTIMAL", "LIMITING_FACTOR", "LOW_INFLUENCE"]
    c = roles.groupby(["species", "ecological_role"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    c = c.assign(_f=c.LIMITING_FACTOR / c.sum(axis=1)).sort_values("_f")
    clean(ax, "x")
    y = np.arange(len(c))
    left = np.zeros(len(c))
    for r in order:
        ax.barh(y, c[r].values, left=left, height=0.66, color=ROLE_COL[r], hatch=ROLE_HATCH[r],
                edgecolor="white", linewidth=0.8, label=ROLE_LABEL[r], zorder=2)
        left += c[r].values
    tot = c[order].sum(axis=1).values
    for yi, (opt, low) in enumerate(zip(c.OPTIMAL.values, c.LOW_INFLUENCE.values)):
        ax.text(opt / 2, yi, f"{opt}", ha="center", va="center", fontsize=7, color="white", fontweight="bold",
                zorder=3)
        if low >= 3:
            ax.text(tot[yi] - low / 2, yi, f"{low}", ha="center", va="center", fontsize=7, color=INK,
                    fontweight="bold", zorder=3,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=ROLE_COL["LOW_INFLUENCE"], linewidth=0))
    for yi, (lim, opt) in enumerate(zip(c.LIMITING_FACTOR.values, c.OPTIMAL.values)):
        if lim >= 2:
            ax.text(opt + lim / 2, yi, f"{lim}", ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", bbox=dict(boxstyle="square,pad=0.1", facecolor=ROLE_COL["LIMITING_FACTOR"],
                                                 linewidth=0))
    ax.set_yticks(y)
    ax.set_yticklabels([species_name(s) for s in c.index], style="italic")
    ax.set_xlabel("Number of predictor-species-model triples")
    ax.set_xlim(0, tot.max() * 1.02)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, handlelength=1.6)


def timing_bar(timing, ax):
    """Mean total run time per species-model combination of the four methods
    (the quantity used in the Borda count). The time per local explanation is
    reported in the text and in the execution-time table."""
    clean(ax)
    total = timing.groupby("xai_method").execution_time_s.mean().reindex(["LIME", "ALE", "SHAP", "CIU"])
    ax.bar(range(4), total.values, width=0.62, color=[XAI_COL[m] for m in total.index], zorder=2)
    for i, v in enumerate(total.values):
        ax.text(i, v + total.max() * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(total.index)
    ax.set_ylabel("Mean run time per combination (s)")
    ax.set_ylim(0, total.max() * 1.12)


def kendall_heatmap(agree, fig, ax):
    methods = ["CIU", "SHAP", "LIME", "ALE"]
    mat = np.eye(4)
    for (a, b), t in agree.groupby(["method_a", "method_b"]).kendall_tau.mean().items():
        mat[methods.index(a), methods.index(b)] = mat[methods.index(b), methods.index(a)] = t
    vector_heatmap(fig, ax, mat, 0, 1, "Mean Kendall's tau", fraction=0.046, pad=0.03)
    ax.set_aspect("equal")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(methods); ax.set_yticklabels(methods)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] > 0.55 else INK)


def article_figures(perf, ev, quality, roles, timing, agree, png=False):
    fig, ax = plt.subplots(figsize=(FULL_W, 85 * MM)); auc_box(perf, ax); save(fig, "Figure_3", png)
    fig, ax = plt.subplots(figsize=(FULL_W, 85 * MM)); auc_heatmap(perf, fig, ax); save(fig, "Figure_4", png)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 75 * MM), gridspec_kw=dict(wspace=0.3))
    faithfulness_box(ev, quality, a1); timing_bar(timing, a2); save(fig, "Figure_5", png)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 75 * MM)); kendall_heatmap(agree, fig, ax); save(fig, "Figure_6", png)
    fig, ax = plt.subplots(figsize=(FULL_W, 85 * MM)); roles_bar(roles, ax); save(fig, "Figure_7", png)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 70 * MM), gridspec_kw=dict(wspace=0.35))
    q = quality.loc[METHOD_ORDER]
    clean(a1)
    a1.bar(range(4), q.faithfulness_mean.values, width=0.62, color=[XAI_COL[m] for m in METHOD_ORDER], zorder=2)
    for i, v in enumerate(q.faithfulness_mean.values):
        a1.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a1.set_xticks(range(4)); a1.set_xticklabels(METHOD_ORDER); a1.set_ylabel("Mean faithfulness")
    a1.set_ylim(0, 1)
    clean(a2, "x")
    bo = q.borda_sum.sort_values()
    yy = np.arange(len(bo))[::-1]
    a2.barh(yy, bo.values, height=0.62, color=[XAI_COL[m] for m in bo.index], zorder=2)
    for y, v in zip(yy, bo.values):
        a2.text(v + 0.12, y, f"{v:.0f}", ha="left", va="center", fontsize=7.5, color=INK)
    a2.set_yticks(yy); a2.set_yticklabels(list(bo.index))
    a2.set_xlabel("Borda rank sum (lower is better)"); a2.set_xlim(0, 10)
    save(fig, "Figure_8", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also write 1000-dpi PNG copies to figures/png/")
    a = ap.parse_args()
    plt.rcParams.update(STYLE)
    perf, ev, quality, roles, timing, agree = load()
    article_figures(perf, ev, quality, roles, timing, agree, png=a.png)


if __name__ == "__main__":
    main()
