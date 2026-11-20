"""
Step 8: Export of the summary tables to one Excel workbook
==========================================================
Collects the summary tables written by steps 3-6 into
results/exports/results_summary.xlsx, with an index sheet describing each
table. Per-combination files (xai_global, xai_local, xai_timing, ciu_details)
stay in results/ as CSV/NPZ files.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "exports", "results_summary.xlsx")

# (sheet name <= 31 characters, file relative to results/, description)
SHEETS = [
    ("cv_single_run", "model_performance.csv",
     "Spatial block CV, original fold assignment (seed 42): mean and SD of six metrics over five folds"),
    ("cv_single_run_scottknott", "model_ranking_scott_knott.csv",
     "Scott-Knott groups of the per-fold AUC per species (original fold assignment)"),
    ("cv_single_run_borda", "model_ranking_borda.csv",
     "Borda count of the six metrics per species (original fold assignment)"),
    ("cv_repeated", "robustness/repeated_cv_performance.csv",
     "Spatial block CV repeated with six seeds, scaler fitted in each training fold: mean and SD over 30 folds"),
    ("cv_repeated_panel", "robustness/repeated_cv_panel.csv",
     "Repeated CV: means over the ten species per model family"),
    ("cv_repeated_scottknott", "robustness/repeated_cv_scott_knott.csv",
     "Repeated CV: Scott-Knott groups of the 30 per-fold AUC values per species"),
    ("cv_repeated_borda", "robustness/repeated_cv_borda.csv",
     "Repeated CV: Borda count of the six metrics per species"),
    ("cv_repeated_ranks", "robustness/repeated_cv_ranks_by_seed.csv",
     "Repeated CV: AUC and TSS rank of each model family for each seed"),
    ("surrogate_trees", "global_surrogate_trees.csv",
     "Global surrogate trees: fidelity, depth, leaves, Gini importances"),
    ("whitebox_trees", "whitebox_trees.csv",
     "White-box trees fitted on the observed labels"),
    ("xai_metrics_50_instances", "xai_comparison_metrics.csv",
     "Deletion-curve faithfulness and monotonicity on the 50-instance evaluation sample; execution times"),
    ("xai_kendall_tau", "xai_ranking_agreement.csv",
     "Kendall's tau between the global rankings of each pair of methods"),
    ("xai_metrics_all_records", "robustness/full_data_xai_evaluation.csv",
     "Single-feature faithfulness, monotonicity and deletion-curve faithfulness on every record, "
     "including 50 random rankings per combination"),
    ("xai_quality_all_records", "robustness/full_data_xai_quality.csv",
     "Method means on every record, Scott-Knott groups, Borda rank sum, mean total run time"),
    ("ciu_roles_50_instances", "ciu_ecological_analysis.csv",
     "CIU mean CI/CU and ecological role per triple (50 instances)"),
    ("ciu_roles_all_records", "robustness/ciu_ecological_roles_all_records.csv",
     "CIU mean CI/CU and ecological role per triple (every record, two Monte Carlo seeds)"),
    ("ciu_threshold_sensitivity", "robustness/ciu_threshold_sensitivity_all_records.csv",
     "Role counts and most constrained species over the CI/CU threshold grid (every record)"),
    ("sampling_eval_means", "robustness/sampling_eval_method_means.csv",
     "Faithfulness/monotonicity means for six 50-instance evaluation samples"),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    index = []
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="index", index=False)
        for sheet, rel, desc in SHEETS:
            assert len(sheet) <= 31, sheet
            df = pd.read_csv(os.path.join(RESULTS, rel))
            df.to_excel(writer, sheet_name=sheet, index=False)
            index.append({"sheet": sheet, "source": "results/" + rel, "rows": len(df), "description": desc})
        pd.DataFrame(index).to_excel(writer, sheet_name="index", index=False)
    print("wrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
