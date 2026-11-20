"""
Step 3: Train Species Distribution Models
============================================
Trains SDMs using spatial block cross-validation.
Evaluates with AUC, TSS (True Skill Statistic), Kappa, accuracy,
sensitivity and specificity.

Models: random forest, XGBoost, LightGBM, multi-layer perceptron, GAM, SVM,
and three ensembles (soft voting, stacking, bagging).

Output:
- models/{species}_{model_name}.pkl
- results/model_performance.csv
- results/model_ranking_scott_knott.csv, results/model_ranking_borda.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SPECIES, DATA_DIR, MODELS_DIR, RESULTS_DIR, MODELS, RANDOM_SEED
from utils.statistical_tests import scott_knott, borda_count


def make_base_model(model_name, params):
    """Create a non-ensemble base model instance from name and parameters."""
    if model_name == "RandomForest":
        return RandomForestClassifier(**params)
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(**params)
    elif model_name == "MLP":
        return MLPClassifier(**params)
    elif model_name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)
    elif model_name == "GAM":
        from utils.gam_wrapper import GAMClassifierWrapper
        return GAMClassifierWrapper(**params)
    elif model_name == "SVM":
        return SVC(**params)
    raise ValueError(f"Unknown base model: {model_name}")


def compute_sensitivity_specificity(y_true, y_pred_binary):
    """
    Compute Sensitivity (True Positive Rate) and Specificity (True Negative Rate).
    Returned separately so that both are recorded as metrics.
    """
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity


def compute_tss(y_true, y_pred_binary):
    """
    True Skill Statistic (TSS) = Sensitivity + Specificity - 1
    Standard SDM evaluation metric (Allouche et al. 2006).
    """
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred_binary)
    return sensitivity + specificity - 1


def create_model(model_name, params):
    """Create a model instance from name and parameters."""
    if model_name in {"RandomForest", "XGBoost", "MLP", "LightGBM", "GAM", "SVM"}:
        return make_base_model(model_name, params)
    elif model_name == "BaggingEnsemble":
        # Homogeneous ensemble: bagging of decision trees.
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_SEED),
            **params,
        )
    elif model_name == "VotingEnsemble":
        estimators = [
            (base_name, make_base_model(base_name, MODELS[base_name]))
            for base_name in params["estimators"]
        ]
        return VotingClassifier(
            estimators=estimators,
            voting=params.get("voting", "soft"),
            weights=params.get("weights"),
            n_jobs=params.get("n_jobs", None),
        )
    elif model_name == "StackingEnsemble":
        estimators = [
            (base_name, make_base_model(base_name, MODELS[base_name]))
            for base_name in params["estimators"]
        ]
        final_estimator_name = params.get("final_estimator", "MLP")
        if final_estimator_name == "MLP":
            final_estimator = MLPClassifier(**MODELS["MLP"])
        else:
            final_estimator = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            stack_method=params.get("stack_method", "predict_proba"),
            passthrough=params.get("passthrough", False),
            cv=params.get("cv", 5),
            n_jobs=params.get("n_jobs", None),
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, X_test, y_test):
    """Compute the six evaluation metrics:
    AUC, TSS, Kappa, Accuracy, Sensitivity, Specificity.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    sensitivity, specificity = compute_sensitivity_specificity(y_test, y_pred)
    tss = sensitivity + specificity - 1
    kappa = cohen_kappa_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return {
        "AUC": auc,
        "TSS": tss,
        "Kappa": kappa,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def main():
    print("=" * 70)
    print("STEP 3: TRAIN SPECIES DISTRIBUTION MODELS")
    print("CIU for species distribution models")
    print("=" * 70)

    all_results = []
    # Per-fold AUC values per (species, model), used by the Scott-Knott test.
    fold_auc_by_species = {}

    for sp_key, sp_info in SPECIES.items():
        print(f"\n{'='*60}")
        print(f"Training models for: {sp_info['gbif_name']} ({sp_info['common_name']})")
        print(f"{'='*60}")

        # Load dataset
        data_path = os.path.join(DATA_DIR, f"{sp_key}_dataset.npz")
        folds_path = os.path.join(DATA_DIR, f"{sp_key}_spatial_folds.npz")

        if not os.path.exists(data_path):
            print(f"  FAIL Dataset not found. Run steps 1-2 first.")
            continue

        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        feature_names = list(data["feature_names"])
        n_folds = int(data["n_folds"])

        print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Features: {feature_names}")
        print(f"  Class distribution: {np.sum(y==1)} presences, {np.sum(y==0)} absences")

        # Load spatial CV folds
        folds_data = np.load(folds_path, allow_pickle=True)
        spatial_folds = []
        for i in range(n_folds):
            train_idx = folds_data[f"train_{i}"]
            test_idx = folds_data[f"test_{i}"]
            spatial_folds.append((train_idx, test_idx))

        # ---- Train each model type ----
        for model_name, model_params in MODELS.items():
            print(f"\n  --- {model_name} ---")

            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(spatial_folds):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Create and train model
                model = create_model(model_name, model_params)
                model.fit(X_train, y_train)

                # Evaluate
                metrics = evaluate_model(model, X_test, y_test)
                fold_metrics.append(metrics)

                print(f"    Fold {fold_idx+1}: AUC={metrics['AUC']:.3f}, "
                      f"TSS={metrics['TSS']:.3f}, Kappa={metrics['Kappa']:.3f}")

            # Average metrics across folds
            avg_metrics = {}
            for metric in fold_metrics[0].keys():
                values = [fm[metric] for fm in fold_metrics]
                avg_metrics[f"{metric}_mean"] = np.mean(values)
                avg_metrics[f"{metric}_std"] = np.std(values)

            # Store per-fold AUC values for Scott-Knott clustering downstream.
            fold_auc_by_species.setdefault(sp_key, {})[model_name] = [
                fm["AUC"] for fm in fold_metrics
            ]

            print(f"\n    AVERAGE: AUC={avg_metrics['AUC_mean']:.3f}±{avg_metrics['AUC_std']:.3f}, "
                  f"TSS={avg_metrics['TSS_mean']:.3f}±{avg_metrics['TSS_std']:.3f}")

            # Train final model on ALL data for XAI analysis
            print(f"    Training final model on full dataset...")
            final_model = create_model(model_name, model_params)
            final_model.fit(X, y)

            # Save model
            model_path = os.path.join(MODELS_DIR, f"{sp_key}_{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(final_model, f)
            print(f"    OK Saved: {model_path}")

            # Record results
            result_row = {
                "species": sp_key,
                "model": model_name,
                **avg_metrics,
            }
            all_results.append(result_row)

    # ---- Save performance summary ----
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(RESULTS_DIR, "model_performance.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*70}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"\nPerformance Summary:")
        print(results_df.to_string(index=False))
        print(f"\nSaved to: {results_path}")

        # ---- Phase 1 ranking: Scott-Knott (on per-fold AUC) + Borda Count ----
        # Scott-Knott groups the models per species on their per-fold AUC;
        # the Borda count aggregates the six metrics with equal weights.
        sk_rows = []
        borda_rows = []
        ranking_metrics = [
            "AUC_mean",
            "TSS_mean",
            "Kappa_mean",
            "Accuracy_mean",
            "Sensitivity_mean",
            "Specificity_mean",
        ]
        for sp_key, model_to_folds in fold_auc_by_species.items():
            sk_df = scott_knott(model_to_folds, alpha=0.05, higher_is_better=True)
            sk_df.insert(0, "species", sp_key)
            sk_rows.append(sk_df)

            sp_mask = results_df["species"] == sp_key
            if sp_mask.any():
                sp_scores = (
                    results_df.loc[sp_mask, ["model"] + ranking_metrics]
                    .set_index("model")
                )
                borda_df = borda_count(sp_scores, higher_is_better=True)
                borda_df.insert(0, "species", sp_key)
                borda_rows.append(borda_df)

        if sk_rows:
            sk_out = pd.concat(sk_rows, ignore_index=True)
            sk_path = os.path.join(RESULTS_DIR, "model_ranking_scott_knott.csv")
            sk_out.to_csv(sk_path, index=False)
            print(f"OK Scott-Knott clustering saved: {sk_path}")

        if borda_rows:
            borda_out = pd.concat(borda_rows, ignore_index=True)
            borda_path = os.path.join(RESULTS_DIR, "model_ranking_borda.csv")
            borda_out.to_csv(borda_path, index=False)
            print(f"OK Borda-Count ranking saved: {borda_path}")
    else:
        print("\nFAIL No models were trained. Check data availability.")


if __name__ == "__main__":
    main()
