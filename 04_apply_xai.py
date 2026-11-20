"""
Step 4: Apply XAI Methods (CIU, SHAP, LIME, ALE)
===================================================
Applies the four explanation methods to each trained SDM, stores global and
local feature importance, and measures execution time. CIU returns two values
per feature and instance: contextual importance (CI, how far the feature can
move the output) and contextual utility (CU, how favourable its current value
is). Global surrogate trees and white-box trees are also fitted here.

Output:
- results/{species}_{model}_xai_global.csv   (global feature importance per method)
- results/{species}_{model}_xai_local.npz    (local explanations per instance)
- results/{species}_{model}_xai_timing.csv   (execution times)
- results/{species}_{model}_ciu_details.csv   (CI + CU per feature per instance)
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SPECIES, DATA_DIR, MODELS_DIR, RESULTS_DIR, MODELS, XAI
from utils.surrogate_tree import fit_global_surrogate, fit_whitebox_tree


# ============================================================
# CIU: Contextual Importance and Utility
# ============================================================
def apply_ciu(model, X, feature_names, n_explain=200, nsamples=1000):
    """
    Apply CIU (contextual importance and utility; Framling 1996, 2022)
    with the implementation in utils/ciu_custom.py.
    """
    from utils.ciu_custom import CIUExplainer

    print("    Applying CIU (Contextual Importance & Utility)...")
    start_time = time.time()
    n_explain = min(n_explain, X.shape[0])

    def predictor(x):
        if x.ndim == 1: x = x.reshape(1, -1)
        return model.predict_proba(x)

    ciu_explainer = CIUExplainer(
        predictor=predictor,
        feature_names=list(feature_names),
        data=X,
        nsamples=nsamples,
        output_index=1,
    )
    result = ciu_explainer.explain_dataset(X, n_explain=n_explain)
    elapsed = time.time() - start_time

    print(f"      Done in {elapsed:.1f}s")
    return {
        "global_importance": result["global_CI"],
        "local_ci": result["local_CI"],
        "local_cu": result["local_CU"],
        "explain_indices": result["explain_indices"],
        "time": elapsed,
    }


# ============================================================
# SHAP: SHapley Additive exPlanations
# ============================================================
def apply_shap(model, X, feature_names, n_explain=200, n_background=100):
    """
    Apply SHAP (TreeExplainer for tree-based models, KernelSHAP for the others).

    SHAP provides a single value per feature per instance:
    - Positive SHAP = feature pushes prediction toward presence
    - Negative SHAP = feature pushes prediction toward absence

    Global importance = mean(|SHAP values|)

    LIMITATION vs CIU:
    SHAP cannot distinguish between "feature is important and at good value"
    vs "feature is important and at bad value" in a single metric.
    CIU's CI+CU decomposition provides this separation.
    """
    import shap

    print("    Applying SHAP...")
    start_time = time.time()

    n_explain = min(n_explain, X.shape[0])

    # Background dataset for KernelSHAP
    np.random.seed(42)
    bg_indices = np.random.choice(X.shape[0], size=min(n_background, X.shape[0]), replace=False)
    X_background = X[bg_indices]

    # Instances to explain
    explain_indices = np.random.choice(X.shape[0], size=n_explain, replace=False)
    X_explain = X[explain_indices]

    # Create SHAP explainer
    # Use TreeExplainer for tree-based models (faster), KernelSHAP for others
    model_type = type(model).__name__

    if model_type in ["RandomForestClassifier", "XGBClassifier", "LGBMClassifier"]:
        print(f"      Using TreeExplainer for {model_type}")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        # For binary classification, take class 1 SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        print(f"      Using KernelSHAP for {model_type}")

        def predict_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, X_background)
        shap_values = explainer.shap_values(X_explain, nsamples=200)

    elapsed = time.time() - start_time

    # Robust handling of different shap_values shapes across SHAP versions
    sv = shap_values
    sv_arr = np.asarray(sv)
    n_features = len(feature_names)

    # Report the array layout returned by the installed SHAP version
    print(f"      SHAP values type={type(sv)} shape={getattr(sv_arr, 'shape', None)}")

    # If shap returned a list (multi-output), try to select class 1
    if isinstance(sv, list):
        try:
            sv = sv[1]
            sv_arr = np.asarray(sv)
            print(f"      SHAP: selected class-1 shap_values, new shape={sv_arr.shape}")
        except Exception:
            # fallback: take mean across list elements
            sv_arr = np.mean(np.abs(np.asarray(sv)), axis=0)

    # If 3D, try to detect which axis corresponds to features and which to outputs
    if sv_arr.ndim == 3:
        # Possible shapes:
        # - (n_explain, n_outputs, n_features)
        # - (n_explain, n_features, n_outputs)
        # - (n_outputs, n_explain, n_features)
        # The feature axis is the one whose length equals n_features.
        f_axis = None
        for axis_idx, dim in enumerate(sv_arr.shape):
            if dim == n_features:
                f_axis = axis_idx
                break

        if f_axis is None:
            raise ValueError(f"Cannot detect feature axis in shap_values with shape={sv_arr.shape}")

        # Move feature axis to last to get shape (..., n_features)
        if f_axis != 2:
            sv_arr = np.moveaxis(sv_arr, f_axis, 2)

        # Now sv_arr has shape (.., n_features). We expect first dim(s) to include n_explain and possibly n_outputs.
        # If shape is (n_explain, n_outputs, n_features) after moveaxis, select output index 1 if available.
        if sv_arr.shape[1] > 1:
            out_idx = 1 if sv_arr.shape[1] > 1 else 0
            # select class output index
            sv_arr = sv_arr[:, out_idx, :]
        else:
            sv_arr = sv_arr[:, 0, :]

    # If transposed (n_features, n_explain) convert to (n_explain, n_features)
    if sv_arr.ndim == 2 and sv_arr.shape[0] == n_features and sv_arr.shape[1] != n_features:
        sv_arr = sv_arr.T

    # At this point we expect sv_arr to be 2D: (n_explain, n_features)
    if sv_arr.ndim == 2 and sv_arr.shape[1] == n_features:
        global_importance = np.mean(np.abs(sv_arr), axis=0)
    elif sv_arr.ndim == 1 and sv_arr.shape[0] == n_features:
        global_importance = np.abs(sv_arr)
    else:
        # Otherwise: flatten, and if the length is a multiple of n_features, reshape and average
        flat = sv_arr.ravel()
        if flat.size == n_features:
            global_importance = np.abs(flat)
        elif flat.size % n_features == 0:
            flat = flat.reshape(-1, n_features)
            global_importance = np.mean(np.abs(flat), axis=0)
            print(f"      SHAP: reshaped flattened array to {flat.shape} and averaged to features")
        else:
            raise ValueError(
                f"Unable to align SHAP values with features: shap_values.shape={sv_arr.shape}, n_features={n_features}"
            )

    print(f"      OK SHAP done in {elapsed:.1f}s")
    print(f"      Top features: {dict(zip(feature_names, np.round(global_importance, 3)))}")

    return {
        "global_importance": global_importance,
        "local_values": sv_arr,
        "explain_indices": explain_indices,
        "time": elapsed,
    }


# ============================================================
# LIME: Local Interpretable Model-agnostic Explanations
# ============================================================
def apply_lime(model, X, feature_names, n_explain=200, n_samples=5000):
    """
    Apply LIME to explain SDM predictions.

    LIME creates a local linear model around each instance.
    Feature importance = |coefficient| of the local linear model.

    LIMITATION: LIME explanations can be unstable (different runs
    may give different results for the same instance).
    """
    import lime
    import lime.lime_tabular

    print("    Applying LIME...")
    start_time = time.time()

    n_explain = min(n_explain, X.shape[0])

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=list(feature_names),
        class_names=["absence", "presence"],
        mode="classification",
        random_state=42,
    )

    # Explain instances
    np.random.seed(42)
    explain_indices = np.random.choice(X.shape[0], size=n_explain, replace=False)

    n_features = X.shape[1]
    all_lime_importances = np.zeros((n_explain, n_features))

    for i, idx in enumerate(explain_indices):
        exp = explainer.explain_instance(
            X[idx],
            model.predict_proba,
            num_features=n_features,
            num_samples=n_samples,
        )

        # Extract feature weights for class 1 (presence)
        feature_weights = dict(exp.as_list(label=1))
        for j, fname in enumerate(feature_names):
            # LIME returns feature names with conditions; match by name
            for lime_feat, weight in feature_weights.items():
                if fname in lime_feat:
                    all_lime_importances[i, j] = weight
                    break

        if (i + 1) % 50 == 0:
            print(f"      Explained {i+1}/{n_explain} instances...")

    elapsed = time.time() - start_time

    # Global importance = mean |LIME coefficient|
    global_importance = np.mean(np.abs(all_lime_importances), axis=0)

    print(f"      OK LIME done in {elapsed:.1f}s")
    print(f"      Top features: {dict(zip(feature_names, np.round(global_importance, 3)))}")

    return {
        "global_importance": global_importance,
        "local_values": all_lime_importances,
        "explain_indices": explain_indices,
        "time": elapsed,
    }


# ============================================================
# ALE: Accumulated Local Effects
# ============================================================
def apply_ale(model, X, feature_names, n_bins=20):
    """
    Apply ALE (Accumulated Local Effects) plots.

    ALE shows the average effect of a feature on the prediction,
    accounting for correlations between features (unlike PDP).

    Returns global importance as the range of ALE values (max - min).
    """
    print("    Applying ALE...")
    start_time = time.time()

    n_features = X.shape[1]
    ale_ranges = np.zeros(n_features)
    ale_curves = {}

    def predict_fn(x):
        return model.predict_proba(x)[:, 1]

    for j in range(n_features):
        try:
            # First-order ALE computed on quantile-based bins
            feature_values = X[:, j]
            # Create quantile-based bins
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.unique(np.percentile(feature_values, percentiles))

            if len(bin_edges) < 3:
                ale_ranges[j] = 0
                ale_curves[feature_names[j]] = {"edges": bin_edges, "effects": np.array([0])}
                continue

            # Compute local effects for each bin
            effects = []
            for k in range(len(bin_edges) - 1):
                # Find instances in this bin
                mask = (feature_values >= bin_edges[k]) & (feature_values < bin_edges[k + 1])
                if k == len(bin_edges) - 2:  # Include right edge for last bin
                    mask = (feature_values >= bin_edges[k]) & (feature_values <= bin_edges[k + 1])

                if np.sum(mask) == 0:
                    effects.append(0)
                    continue

                X_bin = X[mask].copy()

                # Predict at upper and lower bin edges
                X_upper = X_bin.copy()
                X_upper[:, j] = bin_edges[k + 1]
                X_lower = X_bin.copy()
                X_lower[:, j] = bin_edges[k]

                pred_upper = predict_fn(X_upper)
                pred_lower = predict_fn(X_lower)

                local_effect = np.mean(pred_upper - pred_lower)
                effects.append(local_effect)

            # Accumulate effects
            effects = np.array(effects)
            accumulated = np.cumsum(effects)
            # Center: subtract mean
            accumulated = accumulated - np.mean(accumulated)

            ale_ranges[j] = np.max(accumulated) - np.min(accumulated) if len(accumulated) > 0 else 0
            ale_curves[feature_names[j]] = {
                "edges": bin_edges,
                "effects": accumulated,
            }

        except Exception as e:
            print(f"      WARN ALE error for {feature_names[j]}: {e}")
            ale_ranges[j] = 0

    elapsed = time.time() - start_time

    print(f"      OK ALE done in {elapsed:.1f}s")
    print(f"      Top features by ALE range: {dict(zip(feature_names, np.round(ale_ranges, 3)))}")

    return {
        "global_importance": ale_ranges,
        "ale_curves": ale_curves,
        "time": elapsed,
    }


# ============================================================
# MAIN: Apply all methods to all models × all species
# ============================================================
def main():
    print("=" * 70)
    print("STEP 4: APPLY XAI METHODS (CIU, SHAP, LIME, ALE)")
    print("CIU for species distribution models")
    print("=" * 70)

    # Local copy of the XAI settings (the optional quick mode below edits the copy)
    xai_cfg = dict(XAI) if isinstance(XAI, dict) else XAI

    # Quick test mode (environment variable XAI_QUICK): smaller settings for a
    # fast functional check of the pipeline. Not used for the reported results.
    quick = os.environ.get("XAI_QUICK")
    if quick:
        print("\n    XAI_QUICK is set: reduced n_explain/nsamples (functional check only)")
        # Smaller settings are applied to the local copy only; config.py is not changed
        if isinstance(xai_cfg, dict):
            xai_cfg["ciu"] = xai_cfg.get("ciu", {}).copy()
            xai_cfg["shap"] = xai_cfg.get("shap", {}).copy()
            xai_cfg["lime"] = xai_cfg.get("lime", {}).copy()
            xai_cfg["ale"] = xai_cfg.get("ale", {}).copy()
            xai_cfg["ciu"]["n_explain"] = min(20, xai_cfg["ciu"].get("n_explain", 200))
            xai_cfg["ciu"]["nsamples"] = min(50, xai_cfg["ciu"].get("nsamples", 1000))
            xai_cfg["shap"]["n_background_samples"] = min(20, xai_cfg["shap"].get("n_background_samples", 100))
            xai_cfg["lime"]["n_samples"] = min(200, xai_cfg["lime"].get("n_samples", 5000))
            xai_cfg["ale"]["n_bins"] = min(10, xai_cfg["ale"].get("n_bins", 20))

    # Global-interpretability outputs (surrogate and white-box trees).
    all_surrogate_rows = []
    all_whitebox_rows = []

    for sp_key, sp_info in SPECIES.items():
        print(f"\n{'='*60}")
        print(f"Species: {sp_info['gbif_name']} ({sp_info['common_name']})")
        print(f"{'='*60}")

        # Load dataset
        data_path = os.path.join(DATA_DIR, f"{sp_key}_dataset.npz")
        if not os.path.exists(data_path):
            print(f"  FAIL Dataset not found. Skipping.")
            continue

        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        feature_names = list(data["feature_names"])

        # ---- White-box decision tree on the observed labels (one per species) ----
        print("\n  Fitting white-box decision tree on ground-truth labels...")
        whitebox_info = fit_whitebox_tree(X, y, feature_names, max_depth=4)
        whitebox_row = {
            "species": sp_key,
            "depth": whitebox_info["depth"],
            "n_leaves": whitebox_info["n_leaves"],
            "training_accuracy": whitebox_info["training_accuracy"],
        }
        for fname, imp in whitebox_info["feature_importance"].items():
            whitebox_row[f"imp_{fname}"] = imp
        all_whitebox_rows.append(whitebox_row)
        print(
            f"    Depth={whitebox_info['depth']}, leaves={whitebox_info['n_leaves']}, "
            f"training acc={whitebox_info['training_accuracy']:.3f}"
        )

        for model_name in MODELS.keys():
            print(f"\n  --- Model: {model_name} ---")

            # Load trained model
            model_path = os.path.join(MODELS_DIR, f"{sp_key}_{model_name}.pkl")
            if not os.path.exists(model_path):
                print(f"    FAIL Model not found. Skipping.")
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            n_explain = xai_cfg["ciu"]["n_explain"]

            # ---- Apply CIU ----
            ciu_results = apply_ciu(
                model, X, feature_names,
                n_explain=n_explain,
                nsamples=xai_cfg["ciu"]["nsamples"],
            )

            # ---- Apply SHAP ----
            shap_results = apply_shap(
                model, X, feature_names,
                n_explain=n_explain,
                n_background=xai_cfg["shap"]["n_background_samples"],
            )

            # ---- Apply LIME ----
            lime_results = apply_lime(
                model, X, feature_names,
                n_explain=min(n_explain, 100),  # at most 100 instances (computational cost of LIME)
                n_samples=xai_cfg["lime"]["n_samples"],
            )

            # ---- Apply ALE ----
            ale_results = apply_ale(
                model, X, feature_names,
                n_bins=xai_cfg["ale"]["n_bins"],
            )

            # ---- Global surrogate decision tree ----
            # Use 80% of the data to train the surrogate and 20% as oracle
            # for fidelity evaluation.
            print("    Fitting global surrogate decision tree...")
            rng = np.random.RandomState(42)
            perm = rng.permutation(X.shape[0])
            split = int(0.8 * X.shape[0])
            X_surr_train = X[perm[:split]]
            X_surr_oracle = X[perm[split:]]
            surrogate_info = fit_global_surrogate(
                model,
                X_surr_train,
                X_surr_oracle,
                feature_names,
                max_depth=4,
            )
            print(
                f"      Fidelity={surrogate_info['fidelity']:.3f}, "
                f"depth={surrogate_info['depth']}, "
                f"leaves={surrogate_info['n_leaves']}"
            )
            surrogate_row = {
                "species": sp_key,
                "model": model_name,
                "fidelity": surrogate_info["fidelity"],
                "depth": surrogate_info["depth"],
                "n_leaves": surrogate_info["n_leaves"],
            }
            for fname, imp in surrogate_info["feature_importance"].items():
                surrogate_row[f"imp_{fname}"] = imp
            all_surrogate_rows.append(surrogate_row)

            # ---- Save global importance comparison ----
            # Defensive: ensure each "global_importance" is a 1D array/list
            def _to_1d_array(val, names, var_name):
                """Return a 1D numpy array aligned with `names`.

                Accepts:
                - numpy arrays (any shape that can be flattened to length n)
                - lists of scalars length n
                - dict mapping feature_name -> scalar

                Raises a ValueError that describes the input if the conversion fails.
                """
                # dict -> ordered list by names
                if isinstance(val, dict):
                    try:
                        arr = np.array([val[n] for n in names], dtype=float)
                    except KeyError as e:
                        raise ValueError(f"{var_name} dict missing feature: {e}")
                    return arr

                arr = np.asarray(val)
                # A scalar is rejected
                if arr.ndim == 0:
                    raise ValueError(f"{var_name} is a scalar (not 1D): value={arr}")

                # Flatten if necessary (e.g., shape (n,1) or (1,n))
                arr = arr.reshape(-1)

                if arr.shape[0] != len(names):
                    raise ValueError(
                        f"{var_name} length {arr.shape[0]} does not match number of features {len(names)}"
                    )
                return arr

            ciu_g = _to_1d_array(ciu_results.get("global_importance"), feature_names, "CIU_CI")
            shap_g = _to_1d_array(shap_results.get("global_importance"), feature_names, "SHAP_mean_abs")
            lime_g = _to_1d_array(lime_results.get("global_importance"), feature_names, "LIME_mean_abs")
            ale_g = _to_1d_array(ale_results.get("global_importance"), feature_names, "ALE_range")
            surrogate_g = _to_1d_array(
                surrogate_info["feature_importance"], feature_names, "Surrogate_DT"
            )
            whitebox_g = _to_1d_array(
                whitebox_info["feature_importance"], feature_names, "Whitebox_DT"
            )

            global_df = pd.DataFrame({
                "feature": feature_names,
                "CIU_CI": ciu_g,
                "SHAP_mean_abs": shap_g,
                "LIME_mean_abs": lime_g,
                "ALE_range": ale_g,
                "Surrogate_DT": surrogate_g,
                "Whitebox_DT": whitebox_g,
            })

            # Normalize each method to [0, 1] for comparison
            for col in [
                "CIU_CI",
                "SHAP_mean_abs",
                "LIME_mean_abs",
                "ALE_range",
                "Surrogate_DT",
                "Whitebox_DT",
            ]:
                max_val = global_df[col].max()
                if max_val > 0:
                    global_df[f"{col}_norm"] = global_df[col] / max_val
                else:
                    global_df[f"{col}_norm"] = 0

            global_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_xai_global.csv")
            global_df.to_csv(global_path, index=False)
            print(f"\n    OK Global importance saved: {global_path}")

            # ---- Save CIU-specific details (CI + CU) ----
            ciu_detail_rows = []
            for i in range(ciu_results["local_ci"].shape[0]):
                for j, fname in enumerate(feature_names):
                    ciu_detail_rows.append({
                        "instance_idx": ciu_results["explain_indices"][i],
                        "feature": fname,
                        "CI": ciu_results["local_ci"][i, j],
                        "CU": ciu_results["local_cu"][i, j],
                    })

            ciu_detail_df = pd.DataFrame(ciu_detail_rows)
            ciu_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_ciu_details.csv")
            ciu_detail_df.to_csv(ciu_path, index=False)
            print(f"    OK CIU details (CI+CU) saved: {ciu_path}")

            # ---- Save local explanations ----
            local_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_xai_local.npz")
            np.savez(
                local_path,
                ciu_ci=ciu_results["local_ci"],
                ciu_cu=ciu_results["local_cu"],
                ciu_indices=ciu_results["explain_indices"],
                shap_values=shap_results["local_values"],
                shap_indices=shap_results["explain_indices"],
                lime_values=lime_results["local_values"],
                lime_indices=lime_results["explain_indices"],
            )
            print(f"    OK Local explanations saved: {local_path}")

            # ---- Save timing comparison ----
            timing_df = pd.DataFrame({
                "method": ["CIU", "SHAP", "LIME", "ALE"],
                "time_seconds": [
                    ciu_results["time"],
                    shap_results["time"],
                    lime_results["time"],
                    ale_results["time"],
                ],
                "n_instances_explained": [
                    n_explain,
                    n_explain,
                    min(n_explain, 100),
                    X.shape[0],  # ALE: one global computation over all records
                ],
            })
            # For CIU, SHAP and LIME this is the time per local explanation. ALE
            # produces no local explanation: its value is the cost per record of
            # the single global computation, not comparable with the other three.
            timing_df["time_per_instance"] = (
                timing_df["time_seconds"] / timing_df["n_instances_explained"]
            )

            timing_path = os.path.join(RESULTS_DIR, f"{sp_key}_{model_name}_xai_timing.csv")
            timing_df.to_csv(timing_path, index=False)
            print(f"    OK Timing comparison saved: {timing_path}")

            # Print summary
            print(f"\n    TIMING SUMMARY:")
            for _, row in timing_df.iterrows():
                print(f"      {row['method']}: {row['time_seconds']:.1f}s total, "
                      f"{row['time_per_instance']:.3f}s/instance")

    # ---- Save surrogate and white-box tree summaries ----
    if all_surrogate_rows:
        surrogate_df = pd.DataFrame(all_surrogate_rows)
        surrogate_path = os.path.join(RESULTS_DIR, "global_surrogate_trees.csv")
        surrogate_df.to_csv(surrogate_path, index=False)
        print(f"\nOK Global surrogate tree summary saved: {surrogate_path}")

    if all_whitebox_rows:
        whitebox_df = pd.DataFrame(all_whitebox_rows)
        whitebox_path = os.path.join(RESULTS_DIR, "whitebox_trees.csv")
        whitebox_df.to_csv(whitebox_path, index=False)
        print(f"OK White-box tree summary saved: {whitebox_path}")

    print(f"\n{'='*70}")
    print("XAI APPLICATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
