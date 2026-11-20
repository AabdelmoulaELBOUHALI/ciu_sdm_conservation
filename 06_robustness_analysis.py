"""
Step 6: Robustness analysis
===========================
Reads the outputs of steps 3-5 (results/, models/) and writes to
results/robustness/. Files produced by steps 3-5 are never modified.

Parts (run in this order; settings in config.ROBUSTNESS):

  reproduce   Re-runs the stochastic steps with the original seed (42) and
              compares them with results/*.csv (cross-validated metrics,
              50-instance CIU, faithfulness/monotonicity).

  repeated-cv Spatial block cross-validation repeated with six seeds. Each
              seed changes the block-to-fold assignment and the model seeds.
              The scaler is fitted inside each training fold (unscaled
              predictors from data/*_dataset.npz).

  sampling    Sensitivity of the 50-instance explanations to the sampling
              seed: CIU (instance sample and Monte Carlo draws) and the
              stratified evaluation sample of faithfulness/monotonicity.

  full-data-eval
              Deletion-curve faithfulness and monotonicity of the four global
              rankings on every record, with a random-ranking baseline.

  single-feature-eval
              Single-feature faithfulness (Alvarez-Melis & Jaakkola, 2018) of
              the four global rankings on every record, with the same
              random-ranking baseline.

  full-data-ciu
              CIU on every record (two Monte Carlo seeds) for the ecological
              role classification.

  summarise   Builds the summary tables used in the article from the files
              written by the other parts.

Example:
    python 06_robustness_analysis.py --part repeated-cv --workers 6
"""

import argparse
import copy
import importlib
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import (DATA_DIR, MODELS_DIR, RESULTS_DIR, MODELS, XAI, METRICS,  # noqa: E402
                    SPATIAL_CV, RANDOM_SEED, ROBUSTNESS)
from utils.ciu_custom import CIUExplainer  # noqa: E402
from utils.metrics import (compute_faithfulness, compute_monotonicity,  # noqa: E402
                           single_feature_effects, faithfulness_correlation)
from utils.spatial_cv import spatial_block_cv  # noqa: E402
from utils.statistical_tests import scott_knott, borda_count  # noqa: E402

train = importlib.import_module("03_train_sdms")       # create_model, evaluate_model

OUT_DIR = os.path.join(RESULTS_DIR, "robustness")
ORIGINAL_SEED = RANDOM_SEED
assert ORIGINAL_SEED == 42 == SPATIAL_CV["random_seed"]
METRIC_NAMES = ["AUC", "TSS", "Kappa", "Accuracy", "Sensitivity", "Specificity"]
XAI_METHODS = ["CIU", "SHAP", "LIME", "ALE"]
KEY = ["species", "model", "feature"]


# ---------------------------------------------------------------- data ------
def analysed_species():
    """The ten species of results/model_performance.csv."""
    sp = sorted(pd.read_csv(os.path.join(RESULTS_DIR, "model_performance.csv")).species.unique())
    assert len(sp) == 10, sp
    return sp


def combos():
    return [(sp, m) for sp in analysed_species() for m in MODELS]


def load_species(sp_key):
    d = np.load(os.path.join(DATA_DIR, f"{sp_key}_dataset.npz"), allow_pickle=True)
    X, X_raw, y = d["X"], d["X_unscaled"], d["y"]
    assert X.shape == X_raw.shape and X.shape[0] == y.shape[0] and set(np.unique(y)) == {0, 1}
    coords = pd.DataFrame({"longitude": d["coords_lon"], "latitude": d["coords_lat"]})
    return X, X_raw, y, coords, [str(f) for f in d["feature_names"]]


def load_model(sp_key, model_name):
    with open(os.path.join(MODELS_DIR, f"{sp_key}_{model_name}.pkl"), "rb") as f:
        return pickle.load(f)


def single_thread(model):
    """Run nested estimators on one thread when predicting in parallel workers.
    Predicted values are unchanged."""
    if hasattr(model, "get_params"):
        keys = [k for k in model.get_params(deep=True) if k == "n_jobs" or k.endswith("__n_jobs")]
        if keys:
            model.set_params(**{k: 1 for k in keys})
    return model


def stored_folds(sp_key):
    d = np.load(os.path.join(DATA_DIR, f"{sp_key}_spatial_folds.npz"))
    return [(d[f"train_{i}"], d[f"test_{i}"]) for i in range(SPATIAL_CV["n_folds"])]


def folds_for_seed(coords, seed):
    return spatial_block_cv(coords, n_folds=SPATIAL_CV["n_folds"],
                            block_size_km=SPATIAL_CV["block_size_km"], random_seed=seed)


def classify_role(mean_ci, mean_cu, ci_cut, cu_cut):
    """Rule of 05_evaluate_metrics.classify_feature."""
    if mean_ci < ci_cut:
        return "LOW_INFLUENCE"
    return "OPTIMAL" if mean_cu >= cu_cut else "LIMITING_FACTOR"


def seeded_model_params(seed):
    """config.MODELS with every random_state set to `seed`. 03_train_sdms builds
    ensemble members from its module-level MODELS and the bagging base tree from
    RANDOM_SEED, so both are set in this worker process (config.py unchanged)."""
    params = copy.deepcopy(MODELS)
    for p in params.values():
        if "random_state" in p:
            p["random_state"] = seed
    train.MODELS = params
    train.RANDOM_SEED = seed
    return params


def global_importances(sp_key, model_name):
    g = pd.read_csv(os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_xai_global.csv"))
    assert list(g["feature"]) == load_species(sp_key)[4]          # same column order as X
    return {"CIU": g["CIU_CI"].values, "SHAP": g["SHAP_mean_abs"].values,
            "LIME": g["LIME_mean_abs"].values, "ALE": g["ALE_range"].values}


def eval_indices(y, n_eval, seed):
    """Stratified evaluation sample of 05_evaluate_metrics."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = min(n_eval // 2, len(pos))
    n_neg = min(n_eval - n_pos, len(neg))
    assert n_pos > 0 and n_neg > 0
    idx = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                          rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(idx)
    return idx


def ciu_means(model, X, feats, n_explain, seed):
    def predictor(x):                       # as in 04_apply_xai.apply_ciu
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return model.predict_proba(x)

    expl = CIUExplainer(predictor=predictor, feature_names=feats, data=X,
                        nsamples=XAI["ciu"]["nsamples"], output_index=1)
    res = expl.explain_dataset(X, n_explain=n_explain, random_seed=seed)
    return res["local_CI"].mean(axis=0), res["local_CU"].mean(axis=0)


# ---------------------------------------------------------------- tasks -----
def task_cv(args):
    """Five-fold spatial block CV of one species-model pair for one seed."""
    sp_key, model_name, seed, in_fold_scaling = args
    X, X_raw, y, coords, _ = load_species(sp_key)
    params = copy.deepcopy(MODELS) if seed == ORIGINAL_SEED else seeded_model_params(seed)
    folds = stored_folds(sp_key) if seed == ORIGINAL_SEED else folds_for_seed(coords, seed)
    rows = []
    for k, (tr, te) in enumerate(folds):
        if in_fold_scaling:
            model = Pipeline([("scaler", StandardScaler()),
                              ("model", train.create_model(model_name, params[model_name]))])
            model.fit(X_raw[tr], y[tr])
            m = train.evaluate_model(model, X_raw[te], y[te])
        else:                               # pipeline of step 3 (scaled once on the full dataset)
            model = train.create_model(model_name, params[model_name])
            model.fit(X[tr], y[tr])
            m = train.evaluate_model(model, X[te], y[te])
        rows.append({"species": sp_key, "model": model_name, "seed": seed, "fold": k, **m})
    return rows


def task_ciu(args):
    sp_key, model_name, seed, all_records = args
    X, _, _, _, feats = load_species(sp_key)
    model = single_thread(load_model(sp_key, model_name))
    n_explain = X.shape[0] if all_records else XAI["ciu"]["n_explain"]
    ci, cu = ciu_means(model, X, feats, n_explain, seed)
    return [{"species": sp_key, "model": model_name, "seed": seed, "feature": f,
             "mean_CI": ci[j], "mean_CU": cu[j], "n_instances": n_explain}
            for j, f in enumerate(feats)]


def task_eval(args):
    """Faithfulness/monotonicity of the four global rankings (and, on all
    records, of random rankings)."""
    sp_key, model_name, seed, all_records = args
    X, _, y, _, _ = load_species(sp_key)
    model = single_thread(load_model(sp_key, model_name))
    X_eval = X if all_records else X[eval_indices(y, min(XAI["ciu"]["n_explain"], X.shape[0]), seed)]
    rows = [{"species": sp_key, "model": model_name, "seed": seed, "xai_method": m,
             "faithfulness": compute_faithfulness(model, X_eval, imp, n_steps=METRICS["n_faithfulness_steps"],
                                                  X_train=X),
             "monotonicity": compute_monotonicity(model, X_eval, imp, X_train=X)}
            for m, imp in global_importances(sp_key, model_name).items()]
    if all_records:
        rng = np.random.RandomState(ORIGINAL_SEED)
        for r in range(ROBUSTNESS["n_random_rankings"]):
            imp = rng.permutation(X.shape[1]).astype(float)       # a random feature ranking
            rows.append({"species": sp_key, "model": model_name, "seed": r, "xai_method": "RANDOM",
                         "faithfulness": compute_faithfulness(model, X_eval, imp,
                                                              n_steps=METRICS["n_faithfulness_steps"], X_train=X),
                         "monotonicity": compute_monotonicity(model, X_eval, imp, X_train=X)})
    return rows


def task_single_feature(args):
    """Single-feature faithfulness of the four global rankings and of random
    rankings, on every record. CIU is also scored with the mean CI of the other
    fifty-instance samples (CIU_SAMPLE) and of every record (CIU_ALL_RECORDS)."""
    sp_key, model_name = args
    X, _, _, _, feats = load_species(sp_key)
    model = single_thread(load_model(sp_key, model_name))
    effects = single_feature_effects(model, X, X.mean(axis=0))     # baseline: training mean
    rows = [{"species": sp_key, "model": model_name, "seed": ORIGINAL_SEED, "xai_method": m,
             "faithfulness": faithfulness_correlation(imp, effects)}
            for m, imp in global_importances(sp_key, model_name).items()]
    rng = np.random.RandomState(ORIGINAL_SEED)                      # same rankings as task_eval
    for r in range(ROBUSTNESS["n_random_rankings"]):
        imp = rng.permutation(X.shape[1]).astype(float)
        rows.append({"species": sp_key, "model": model_name, "seed": r, "xai_method": "RANDOM",
                     "faithfulness": faithfulness_correlation(imp, effects)})
    smp = read("sampling_ciu.csv")
    smp = smp[(smp.species == sp_key) & (smp.model == model_name)]
    for seed, sub in smp.groupby("seed"):
        rows.append({"species": sp_key, "model": model_name, "seed": seed, "xai_method": "CIU_SAMPLE",
                     "faithfulness": faithfulness_correlation(sub.set_index("feature").loc[feats, "mean_CI"].values,
                                                              effects)})
    fd = read("full_data_ciu_by_seed.csv")
    ci = fd[(fd.species == sp_key) & (fd.model == model_name)].groupby("feature").mean_CI.mean()
    rows.append({"species": sp_key, "model": model_name, "seed": ORIGINAL_SEED, "xai_method": "CIU_ALL_RECORDS",
                 "faithfulness": faithfulness_correlation(ci.loc[feats].values, effects)})
    return rows


def run(task, args_list, workers, label):
    t0 = time.time()
    out = []
    with Pool(workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(task, args_list), 1):
            out.extend(rows)
            if i % 10 == 0 or i == len(args_list):
                print(f"  [{label}] {i}/{len(args_list)} tasks, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out)


def write(df, name):
    df.to_csv(os.path.join(OUT_DIR, name), index=False)
    print("  wrote", os.path.join("results", "robustness", name), flush=True)


def read(name):
    return pd.read_csv(os.path.join(OUT_DIR, name))


# ---------------------------------------------------------------- parts -----
def part_reproduce(workers):
    s = ORIGINAL_SEED
    rep = {}
    for sp in analysed_species():                                  # fold generator = stored folds
        _, _, _, coords, _ = load_species(sp)
        for (_, a), (_, b) in zip(folds_for_seed(coords, s), stored_folds(sp)):
            assert np.array_equal(np.sort(a), np.sort(b)), sp
    cv = run(task_cv, [(sp, m, s, False) for sp, m in combos()], workers, "reproduce-cv")
    write(cv, "reproduce_cv_folds.csv")
    agg = cv.groupby(["species", "model"])[METRIC_NAMES].mean().add_suffix("_mean").reset_index()
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "model_performance.csv")).merge(agg, on=["species", "model"],
                                                                                  suffixes=("", "_rerun"))
    assert len(ref) == 90
    rep["cv_max_abs_diff"] = {k: float((ref[f"{k}_mean"] - ref[f"{k}_mean_rerun"]).abs().max()) for k in METRIC_NAMES}

    ciu = run(task_ciu, [(sp, m, s, False) for sp, m in combos()], workers, "reproduce-ciu")
    write(ciu, "reproduce_ciu.csv")
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "ciu_ecological_analysis.csv")).merge(ciu, on=KEY,
                                                                                         suffixes=("", "_rerun"))
    assert len(ref) == 657
    rep["ciu_max_abs_diff"] = float(max((ref.mean_CI - ref.mean_CI_rerun).abs().max(),
                                        (ref.mean_CU - ref.mean_CU_rerun).abs().max()))
    roles = [classify_role(a, b, ROBUSTNESS["ci_cut"], ROBUSTNESS["cu_cut"])
             for a, b in zip(ref.mean_CI_rerun, ref.mean_CU_rerun)]
    rep["ciu_role_agreement"] = float((ref.ecological_role == pd.Series(roles)).mean())

    ev = run(task_eval, [(sp, m, s, False) for sp, m in combos()], workers, "reproduce-eval")
    write(ev, "reproduce_eval.csv")
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "xai_comparison_metrics.csv")).merge(
        ev, on=["species", "model", "xai_method"], suffixes=("", "_rerun"))
    assert len(ref) == 360
    rep["eval_max_abs_diff"] = float(max((ref.faithfulness - ref.faithfulness_rerun).abs().max(),
                                         (ref.monotonicity - ref.monotonicity_rerun).abs().max()))
    return rep


def part_repeated_cv(workers):
    seeds = ROBUSTNESS["cv_seeds"]
    df = run(task_cv, [(sp, m, s, True) for s in seeds for sp, m in combos()], workers, "repeated-cv")
    write(df, "repeated_cv_folds.csv")
    return {"n_rows": len(df), "seeds": seeds}


def part_sampling(workers):
    seeds = ROBUSTNESS["sampling_seeds"]
    assert ORIGINAL_SEED not in seeds
    write(run(task_ciu, [(sp, m, s, False) for s in seeds for sp, m in combos()], workers, "sampling-ciu"),
          "sampling_ciu.csv")
    write(run(task_eval, [(sp, m, s, False) for s in seeds for sp, m in combos()], workers, "sampling-eval"),
          "sampling_eval.csv")
    return {"seeds": seeds}


def part_full_data_eval(workers):
    write(run(task_eval, [(sp, m, ORIGINAL_SEED, True) for sp, m in combos()], workers, "full-data-eval"),
          "full_data_eval.csv")
    return {"n_random_rankings": ROBUSTNESS["n_random_rankings"]}


def part_single_feature_eval(workers):
    write(run(task_single_feature, combos(), workers, "single-feature-eval"), "full_data_single_feature_faithfulness.csv")
    return {"n_random_rankings": ROBUSTNESS["n_random_rankings"]}


def part_full_data_ciu(workers):
    seeds = ROBUSTNESS["full_data_mc_seeds"]
    assert len(seeds) >= 2
    write(run(task_ciu, [(sp, m, s, True) for s in seeds for sp, m in combos()], workers, "full-data-ciu"),
          "full_data_ciu_by_seed.csv")
    return {"mc_seeds": seeds}


def role_shares(t, role_col):
    share = t.assign(l=t[role_col] == "LIMITING_FACTOR").groupby("species").l.mean().sort_values(ascending=False)
    return share


def part_summarise():
    rep = {}
    timing = pd.read_csv(os.path.join(RESULTS_DIR, "xai_comparison_metrics.csv")).groupby(
        "xai_method").execution_time_s.mean()

    # ---- repeated CV ----
    cv = read("repeated_cv_folds.csv")
    perf = cv.groupby(["species", "model"])[METRIC_NAMES].agg(["mean", "std"])
    perf.columns = [f"{a}_{b}" for a, b in perf.columns]
    perf = perf.reset_index()
    write(perf, "repeated_cv_performance.csv")                       # same layout as model_performance.csv
    panel = perf.groupby("model")[[f"{k}_mean" for k in METRIC_NAMES]].mean()
    write(panel.sort_values("AUC_mean", ascending=False).reset_index(), "repeated_cv_panel.csv")
    by_seed = cv.groupby(["seed", "species", "model"])[METRIC_NAMES].mean().groupby(["seed", "model"]).mean()
    ranks = by_seed.AUC.unstack("seed").rank(ascending=False)
    tss_ranks = by_seed.TSS.unstack("seed").rank(ascending=False)
    write(pd.concat({"AUC_rank": ranks, "TSS_rank": tss_ranks}, axis=1).reset_index(), "repeated_cv_ranks_by_seed.csv")
    sk_rows, borda_rows = [], []
    for sp, sub in cv.groupby("species"):
        groups = {m: g.AUC.tolist() for m, g in sub.groupby("model")}
        skd = scott_knott(groups, alpha=0.05, higher_is_better=True).assign(species=sp)
        sk_rows.append(skd)
        bd = borda_count(perf[perf.species == sp].set_index("model")[[f"{k}_mean" for k in METRIC_NAMES]],
                         higher_is_better=True).assign(species=sp)
        borda_rows.append(bd)
    write(pd.concat(sk_rows), "repeated_cv_scott_knott.csv")
    write(pd.concat(borda_rows), "repeated_cv_borda.csv")

    # ---- explanation quality on every record ----
    # faithfulness: single-feature definition; monotonicity: deletion curve
    ev = read("full_data_eval.csv")                                   # deletion curve
    sf_all = read("full_data_single_feature_faithfulness.csv")        # single feature
    ciu_var = sf_all[sf_all.xai_method.isin(["CIU_SAMPLE", "CIU_ALL_RECORDS"])]
    rep["ciu_faithfulness_other_samples"] = {int(k): float(v) for k, v in
                                             ciu_var[ciu_var.xai_method == "CIU_SAMPLE"].groupby("seed").faithfulness.mean().items()}
    rep["ciu_faithfulness_all_records_ci"] = float(ciu_var[ciu_var.xai_method == "CIU_ALL_RECORDS"].faithfulness.mean())
    sf = sf_all[~sf_all.index.isin(ciu_var.index)]
    key = ["species", "model", "xai_method", "seed"]
    assert len(ev) == len(sf) == 90 * (len(XAI_METHODS) + ROBUSTNESS["n_random_rankings"])
    q = sf.merge(ev[key + ["monotonicity"]], on=key, validate="1:1")
    q = q.merge(ev[key + ["faithfulness"]].rename(columns={"faithfulness": "deletion_faithfulness"}),
                on=key, validate="1:1")
    write(q, "full_data_xai_evaluation.csv")
    rnd, q = q[q.xai_method == "RANDOM"], q[q.xai_method != "RANDOM"]
    crit = ["faithfulness", "monotonicity", "deletion_faithfulness"]
    means = q.groupby("xai_method")[crit].agg(["mean", "std"])
    means.columns = [f"{a}_{b}" for a, b in means.columns]
    skd = scott_knott({m: q[q.xai_method == m].faithfulness.tolist() for m in XAI_METHODS},
                      alpha=0.05, higher_is_better=True)
    sk_del = scott_knott({m: q[q.xai_method == m].deletion_faithfulness.tolist() for m in XAI_METHODS},
                         alpha=0.05, higher_is_better=True)
    mm = q.groupby("xai_method")[["faithfulness", "monotonicity"]].mean()
    mm["time"] = timing
    bd = borda_count(mm, higher_is_better={"faithfulness": True, "monotonicity": True, "time": False})
    xq = means.join(skd.set_index("treatment")[["cluster"]])
    xq = xq.join(sk_del.set_index("treatment")[["cluster"]].rename(columns={"cluster": "cluster_deletion"}))
    xq = xq.join(bd.set_index("candidate")[["borda_sum"]])
    xq = xq.join(mm[["time"]].rename(columns={"time": "mean_total_time_s"}))
    write(xq.reset_index().rename(columns={"index": "xai_method"}), "full_data_xai_quality.csv")
    base = rnd.groupby(["species", "model"])[crit].mean()
    for c in crit:
        rep[f"random_baseline_{c}"] = float(base[c].mean())
    wide = q.pivot_table(index=["species", "model"], columns="xai_method", values="faithfulness")
    rep["share_above_random_faithfulness"] = {m: float((wide[m] > base.faithfulness.reindex(wide.index)).mean())
                                              for m in XAI_METHODS}
    write(q.groupby("model")[crit].mean().reset_index(), "full_data_faithfulness_by_model.csv")

    # ---- CIU roles on every record ----
    fd = read("full_data_ciu_by_seed.csv")
    fd["role"] = [classify_role(a, b, ROBUSTNESS["ci_cut"], ROBUSTNESS["cu_cut"]) for a, b in zip(fd.mean_CI, fd.mean_CU)]
    wide = fd.pivot_table(index=KEY, columns="seed", values="role", aggfunc="first")
    assert len(wide) == 657 and not wide.isna().any().any()
    rep["full_data_mc_seed_agreement"] = float((wide.nunique(axis=1) == 1).mean())
    roles = fd.groupby(KEY)[["mean_CI", "mean_CU"]].mean().reset_index()
    roles["ecological_role"] = [classify_role(a, b, ROBUSTNESS["ci_cut"], ROBUSTNESS["cu_cut"])
                                for a, b in zip(roles.mean_CI, roles.mean_CU)]
    orig = pd.read_csv(os.path.join(RESULTS_DIR, "ciu_ecological_analysis.csv"))
    roles = roles.merge(orig[KEY + ["ecological_role", "mean_CU", "mean_CI"]].rename(
        columns={"ecological_role": "role_50_instances", "mean_CU": "mean_CU_50_instances",
                 "mean_CI": "mean_CI_50_instances"}), on=KEY)
    write(roles, "ciu_ecological_roles_all_records.csv")
    rep["agreement_with_50_instance_roles"] = float((roles.ecological_role == roles.role_50_instances).mean())
    changed = roles[roles.ecological_role != roles.role_50_instances]
    near = ((changed.mean_CU_50_instances - ROBUSTNESS["cu_cut"]).abs() <= 0.1) | \
           ((changed.mean_CI_50_instances - ROBUSTNESS["ci_cut"]).abs() <= 0.05)
    rep["changed_labels"] = int(len(changed))
    rep["changed_labels_near_cutoff"] = int(near.sum())
    sens = []
    for ci_cut in ROBUSTNESS["ci_cut_grid"]:
        for cu_cut in ROBUSTNESS["cu_cut_grid"]:
            r = roles.assign(rr=[classify_role(a, b, ci_cut, cu_cut) for a, b in zip(roles.mean_CI, roles.mean_CU)])
            sh = role_shares(r, "rr")
            c = r.rr.value_counts()
            sens.append({"ci_cut": ci_cut, "cu_cut": cu_cut, "OPTIMAL": int(c.get("OPTIMAL", 0)),
                         "LIMITING_FACTOR": int(c.get("LIMITING_FACTOR", 0)),
                         "LOW_INFLUENCE": int(c.get("LOW_INFLUENCE", 0)),
                         "most_constrained": sh.index[0], "second": sh.index[1]})
    write(pd.DataFrame(sens), "ciu_threshold_sensitivity_all_records.csv")

    # ---- sensitivity of the 50-instance labels to the sampling seed ----
    smp = read("sampling_ciu.csv")
    allr = pd.concat([orig[KEY + ["mean_CI", "mean_CU"]].assign(seed=ORIGINAL_SEED), smp[KEY + ["mean_CI", "mean_CU", "seed"]]])
    allr["role"] = [classify_role(a, b, ROBUSTNESS["ci_cut"], ROBUSTNESS["cu_cut"]) for a, b in zip(allr.mean_CI, allr.mean_CU)]
    w = allr.pivot_table(index=KEY, columns="seed", values="role", aggfunc="first")
    rep["sampling_agreement_with_seed42"] = {int(s): float((w[s] == w[ORIGINAL_SEED]).mean())
                                             for s in w.columns if s != ORIGINAL_SEED}
    rep["sampling_limiting_counts"] = {int(s): int((w[s] == "LIMITING_FACTOR").sum()) for s in w.columns}
    rep["sampling_most_constrained"] = {int(s): role_shares(allr[allr.seed == s], "role").index[0]
                                        for s in w.columns}
    sev = read("sampling_eval.csv")
    sev = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, "xai_comparison_metrics.csv"))
                     [["species", "model", "xai_method", "faithfulness", "monotonicity"]].assign(seed=ORIGINAL_SEED), sev])
    write(sev.groupby(["seed", "xai_method"])[["faithfulness", "monotonicity"]].mean().reset_index(),
          "sampling_eval_method_means.csv")
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf8") as f:
        json.dump(rep, f, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["reproduce", "repeated-cv", "sampling", "full-data-eval", "single-feature-eval",
                             "full-data-ciu", "summarise"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = {"reproduce": lambda: part_reproduce(a.workers), "repeated-cv": lambda: part_repeated_cv(a.workers),
             "sampling": lambda: part_sampling(a.workers),
             "full-data-eval": lambda: part_full_data_eval(a.workers),
             "single-feature-eval": lambda: part_single_feature_eval(a.workers),
             "full-data-ciu": lambda: part_full_data_ciu(a.workers),
             "summarise": part_summarise}
    rep = parts[a.part]()
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
