"""
Step 5: Evaluate & Compare XAI Methods
=========================================
Computes quantitative metrics following Hakkoum et al. (2024):
1. Faithfulness: Does removing "important" features degrade predictions more?
2. Monotonicity: Does degradation increase monotonically with feature removal?
3. Execution time: How fast is each method?
4. Ranking agreement: Kendall's tau between methods
5. CIU-specific: Ecological limiting factor analysis (CI vs CU)

Output:
- results/xai_comparison_metrics.csv
- results/xai_ranking_agreement.csv
- results/ciu_ecological_analysis.csv
- results/ciu_threshold_sensitivity.csv   (threshold-robustness of the role split)
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SPECIES, DATA_DIR, MODELS_DIR, RESULTS_DIR, MODELS, METRICS, XAI
from utils.metrics import (
    compute_faithfulness,
    compute_monotonicity,
    compute_feature_ranking_agreement,
)
from utils.statistical_tests import scott_knott, borda_count


def main():
    print("=" * 70)
    print("STEP 5: EVALUATE & COMPARE XAI METHODS")
    print("CIU for species distribution models")
    print("=" * 70)

    all_metrics = []
    all_rankings = []
    all_ciu_ecology = []

    for sp_key, sp_info in SPECIES.items():
        # Load dataset
        data_path = os.path.join(DATA_DIR, f"{sp_key}_dataset.npz")
        if not os.path.exists(data_path):
            continue

        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        feature_names = list(data["feature_names"])

        for model_name in MODELS.keys():
            print(f"\n{'='*60}")
            print(f"{sp_info['gbif_name']} × {model_name}")
            print(f"{'='*60}")

            # Load model
            model_path = os.path.join(MODELS_DIR, f"{sp_key}_{model_name}.pkl")
            if not os.path.exists(model_path):
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load global importances
            global_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_xai_global.csv")
            if not os.path.exists(global_path):
                print(f"  FAIL XAI results not found. Run step 4 first.")
                continue

            global_df = pd.read_csv(global_path)

            # Load timing
            timing_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_xai_timing.csv")
            timing_df = pd.read_csv(timing_path) if os.path.exists(timing_path) else None

            # Build importance dict
            importances = {
                "CIU": global_df["CIU_CI"].values,
                "SHAP": global_df["SHAP_mean_abs"].values,
                "LIME": global_df["LIME_mean_abs"].values,
                "ALE": global_df["ALE_range"].values,
            }

            # ---- 5.1 Faithfulness & Monotonicity ----
            print("\n  Computing faithfulness and monotonicity...")

            # Stratified random sample (50 instances balanced
            # across presence/absence). Falls back to a plain random sample when
            # one class is too small to stratify.
            rng = np.random.RandomState(42)
            n_eval_target = XAI["ciu"]["n_explain"]
            n_eval = min(n_eval_target, X.shape[0])
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = min(n_eval // 2, len(pos_idx))
            n_neg = min(n_eval - n_pos, len(neg_idx))
            if n_pos > 0 and n_neg > 0:
                eval_indices = np.concatenate([
                    rng.choice(pos_idx, size=n_pos, replace=False),
                    rng.choice(neg_idx, size=n_neg, replace=False),
                ])
            else:
                eval_indices = rng.choice(X.shape[0], size=n_eval, replace=False)
            rng.shuffle(eval_indices)
            X_eval = X[eval_indices]

            for method_name, imp_values in importances.items():
                faith = compute_faithfulness(
                    model, X_eval, imp_values,
                    n_steps=METRICS["n_faithfulness_steps"],
                    X_train=X,
                )
                mono = compute_monotonicity(
                    model, X_eval, imp_values,
                    X_train=X,
                )

                timing_row = timing_df[timing_df["method"] == method_name] if timing_df is not None else None
                exec_time = timing_row["time_seconds"].values[0] if timing_row is not None and len(timing_row) > 0 else np.nan
                time_per_inst = timing_row["time_per_instance"].values[0] if timing_row is not None and len(timing_row) > 0 else np.nan

                metric_row = {
                    "species": sp_key,
                    "model": model_name,
                    "xai_method": method_name,
                    "faithfulness": faith,
                    "monotonicity": mono,
                    "execution_time_s": exec_time,
                    "time_per_instance_s": time_per_inst,
                }
                all_metrics.append(metric_row)

                print(f"    {method_name}: faithfulness={faith:.4f}, monotonicity={mono:.4f}, time={exec_time:.1f}s")

            # ---- 5.2 Ranking Agreement (Kendall's tau) ----
            print("\n  Computing ranking agreement (Kendall's tau)...")

            methods = list(importances.keys())
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    tau = compute_feature_ranking_agreement(
                        importances[methods[i]],
                        importances[methods[j]],
                        method="kendall",
                    )
                    ranking_row = {
                        "species": sp_key,
                        "model": model_name,
                        "method_a": methods[i],
                        "method_b": methods[j],
                        "kendall_tau": tau,
                    }
                    all_rankings.append(ranking_row)
                    print(f"    {methods[i]} vs {methods[j]}: tau = {tau:.4f}")

            # ---- 5.3 CIU Ecological Analysis ----
            print("\n  CIU Ecological Limiting Factor Analysis...")

            ciu_detail_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_ciu_details.csv")
            if os.path.exists(ciu_detail_path):
                ciu_df = pd.read_csv(ciu_detail_path)

                # Compute mean CI and CU per feature
                ciu_summary = ciu_df.groupby("feature").agg(
                    mean_CI=("CI", "mean"),
                    std_CI=("CI", "std"),
                    mean_CU=("CU", "mean"),
                    std_CU=("CU", "std"),
                ).reset_index()

                # Classify each feature into ecological categories.
                # The three conditions below are exhaustive over the (CI, CU)
                # plane, so exactly three roles ever occur (OPTIMAL /
                # LIMITING_FACTOR / LOW_INFLUENCE) and the "MODERATE" fallback is
                # unreachable and kept only as a defensive default. The
                # cut-offs (0.3, 0.5) match
                # classify_at() used in the threshold-sensitivity block below.
                def classify_feature(row):
                    ci_threshold = 0.3
                    cu_threshold = 0.5

                    if row["mean_CI"] >= ci_threshold and row["mean_CU"] < cu_threshold:
                        return "LIMITING_FACTOR"  # High importance, unfavorable value
                    elif row["mean_CI"] >= ci_threshold and row["mean_CU"] >= cu_threshold:
                        return "OPTIMAL"  # High importance, already favorable
                    elif row["mean_CI"] < ci_threshold:
                        return "LOW_INFLUENCE"  # Low importance regardless
                    else:
                        return "MODERATE"  # unreachable; defensive default only

                ciu_summary["ecological_role"] = ciu_summary.apply(classify_feature, axis=1)
                ciu_summary["species"] = sp_key
                ciu_summary["model"] = model_name

                all_ciu_ecology.append(ciu_summary)

                print(f"\n    Feature Ecological Classification (CIU):")
                for _, row in ciu_summary.iterrows():
                    print(f"      {row['feature']}: CI={row['mean_CI']:.3f}, "
                          f"CU={row['mean_CU']:.3f} -> {row['ecological_role']}")

    # ---- Save all results ----
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(RESULTS_DIR, "xai_comparison_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nOK Metrics saved: {metrics_path}")

        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY: XAI METHOD COMPARISON")
        print(f"{'='*70}")
        summary = metrics_df.groupby("xai_method").agg(
            mean_faithfulness=("faithfulness", "mean"),
            mean_monotonicity=("monotonicity", "mean"),
            mean_time=("execution_time_s", "mean"),
        ).round(4)
        print(summary.to_string())

        # ---- Ranking of the explanation methods: Scott-Knott on faithfulness + Borda count ----
        faith_groups = {
            method: sub["faithfulness"].dropna().tolist()
            for method, sub in metrics_df.groupby("xai_method")
            if sub["faithfulness"].dropna().size > 0
        }
        if faith_groups:
            sk_faith = scott_knott(faith_groups, alpha=0.05, higher_is_better=True)
            sk_faith_path = os.path.join(
                RESULTS_DIR, "xai_ranking_scott_knott.csv"
            )
            sk_faith.to_csv(sk_faith_path, index=False)
            print(f"OK Scott-Knott clustering of XAI methods saved: {sk_faith_path}")

        # Borda aggregates faithfulness (higher better), monotonicity (higher
        # better), and execution time per instance (lower better).
        borda_input = summary.rename(
            columns={
                "mean_faithfulness": "faithfulness",
                "mean_monotonicity": "monotonicity",
                "mean_time": "time",
            }
        )
        borda_methods = borda_count(
            borda_input,
            higher_is_better={
                "faithfulness": True,
                "monotonicity": True,
                "time": False,
            },
        )
        borda_methods_path = os.path.join(RESULTS_DIR, "xai_ranking_borda.csv")
        borda_methods.to_csv(borda_methods_path, index=False)
        print(f"OK Borda-Count ranking of XAI methods saved: {borda_methods_path}")

    if all_rankings:
        rankings_df = pd.DataFrame(all_rankings)
        rankings_path = os.path.join(RESULTS_DIR, "xai_ranking_agreement.csv")
        rankings_df.to_csv(rankings_path, index=False)
        print(f"\nOK Rankings saved: {rankings_path}")

    if all_ciu_ecology:
        ecology_df = pd.concat(all_ciu_ecology, ignore_index=True)
        ecology_path = os.path.join(RESULTS_DIR, "ciu_ecological_analysis.csv")
        ecology_df.to_csv(ecology_path, index=False)
        print(f"\nOK CIU ecological analysis saved: {ecology_path}")

        # ---- 5.4 Threshold-sensitivity of the role classification ----
        # The (CI >= 0.3, CU >= 0.5) cut-offs are conventional (Framling 1996;
        # Knapic et al. 2021), not data-driven. To document how much the role
        # proportions depend on that choice, the same classification rule is
        # re-applied to the same (mean_CI, mean_CU) values over a small threshold grid.
        # This is a post-hoc re-labelling: no model is retrained and no CIU value
        # is recomputed.
        ci_grid = [0.25, 0.30, 0.35]
        cu_grid = [0.40, 0.50, 0.60]
        baseline = (0.30, 0.50)  # rule used everywhere else in the pipeline

        def classify_at(mean_ci, mean_cu, ci_t, cu_t):
            """Same three-way rule as classify_feature(), parameterised by the
            two cut-offs. Exhaustive over the (CI, CU) plane: every triple maps
            to exactly one of OPTIMAL / LIMITING_FACTOR / LOW_INFLUENCE."""
            if mean_ci >= ci_t and mean_cu >= cu_t:
                return "OPTIMAL"
            elif mean_ci >= ci_t:  # mean_cu < cu_t
                return "LIMITING_FACTOR"
            else:                  # mean_ci < ci_t
                return "LOW_INFLUENCE"

        n_total = len(ecology_df)
        sensitivity_rows = []
        for ci_t in ci_grid:
            for cu_t in cu_grid:
                roles = ecology_df.apply(
                    lambda r: classify_at(r["mean_CI"], r["mean_CU"], ci_t, cu_t),
                    axis=1,
                )
                counts = roles.value_counts()
                n_opt = int(counts.get("OPTIMAL", 0))
                n_lim = int(counts.get("LIMITING_FACTOR", 0))
                n_low = int(counts.get("LOW_INFLUENCE", 0))

                # Species carrying the largest LIMITING_FACTOR fraction at this
                # threshold pair (reported in the threshold-sensitivity table).
                lim_mask = roles == "LIMITING_FACTOR"
                per_species = (
                    ecology_df.assign(is_lim=lim_mask)
                    .groupby("species")["is_lim"]
                    .mean()
                )
                top_species = (
                    per_species.idxmax() if per_species.max() > 0 else "(none)"
                )

                sensitivity_rows.append({
                    "ci_threshold": ci_t,
                    "cu_threshold": cu_t,
                    "is_baseline": (ci_t, cu_t) == baseline,
                    "n_total": n_total,
                    "n_optimal": n_opt,
                    "pct_optimal": round(100 * n_opt / n_total, 1),
                    "n_limiting": n_lim,
                    "pct_limiting": round(100 * n_lim / n_total, 1),
                    "n_low_influence": n_low,
                    "pct_low_influence": round(100 * n_low / n_total, 1),
                    "most_constrained_species": top_species,
                })

        sensitivity_df = pd.DataFrame(sensitivity_rows)
        sensitivity_path = os.path.join(
            RESULTS_DIR, "ciu_threshold_sensitivity.csv"
        )
        sensitivity_df.to_csv(sensitivity_path, index=False)
        print(f"OK CIU threshold-sensitivity analysis saved: {sensitivity_path}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
