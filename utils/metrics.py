"""
XAI Evaluation Metrics
========================
Quantitative metrics for comparing interpretability methods,
following Hakkoum, Idri & Abnane (2024, EAAI).

Metrics:
1. Faithfulness: Does removing "important" features degrade prediction more?
   compute_faithfulness: correlation between deletion step and cumulative change
   (deletion curve); faithfulness_correlation: correlation between importance and
   single-feature change (Alvarez-Melis & Jaakkola, 2018).
2. Monotonicity: Does prediction degrade monotonically as features are removed in order?
3. Execution Time: How long does each method take?
4. Feature Ranking Agreement: Kendall's tau between methods
"""

import numpy as np
import time
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


def compute_faithfulness(
    model,
    X: np.ndarray,
    feature_importances: np.ndarray,
    n_steps: int = 10,
    baseline: str = "mean",
    X_train: np.ndarray = None,
) -> float:
    """
    Compute faithfulness metric (also called "deletion metric").

    Idea: If feature importance ranking is faithful, removing the most important
    features first should degrade model performance faster than removing
    less important features.

    Faithfulness = Pearson correlation between the deletion step and the
    mean absolute change in prediction when the most important features
    are replaced by the baseline.

    Parameters
    ----------
    model : fitted model
        Must have predict_proba method
    X : np.ndarray
        Test instances (n_samples, n_features)
    feature_importances : np.ndarray
        Importance scores (n_features,): higher = more important
    n_steps : int
        Number of features to remove incrementally
    baseline : str
        What to replace features with: "mean" or "zero" (other values use the mean)
    X_train : np.ndarray, optional
        Training data for computing mean baseline

    Returns
    -------
    float
        Faithfulness score (higher = more faithful, range roughly -1 to 1)
    """
    n_features = X.shape[1]
    n_steps = min(n_steps, n_features)

    # Sort features by importance (most important first)
    importance_order = np.argsort(feature_importances)[::-1]

    # Compute baseline values
    if baseline == "mean":
        if X_train is not None:
            baseline_values = np.mean(X_train, axis=0)
        else:
            baseline_values = np.mean(X, axis=0)
    elif baseline == "zero":
        baseline_values = np.zeros(n_features)
    else:
        baseline_values = np.mean(X, axis=0)

    # Get original predictions
    original_probs = _get_proba(model, X)

    # Progressively mask features (most important first)
    performance_drops = []
    for step in range(1, n_steps + 1):
        n_to_mask = max(1, int(step * n_features / n_steps))
        features_to_mask = importance_order[:n_to_mask]

        # Create masked input
        X_masked = X.copy()
        X_masked[:, features_to_mask] = baseline_values[features_to_mask]

        # Compute performance on masked input
        masked_probs = _get_proba(model, X_masked)

        # Performance drop = mean absolute change in prediction
        drop = np.mean(np.abs(original_probs - masked_probs))
        performance_drops.append(drop)

    # Faithfulness = Pearson correlation between step number and performance drop
    # Faithful explanations should show increasing drop (positive correlation)
    steps = np.arange(1, len(performance_drops) + 1)
    if np.std(performance_drops) == 0:
        return 0.0

    faithfulness = np.corrcoef(steps, performance_drops)[0, 1]
    return faithfulness


def compute_monotonicity(
    model,
    X: np.ndarray,
    feature_importances: np.ndarray,
    X_train: np.ndarray = None,
) -> float:
    """
    Compute monotonicity metric.

    Measures whether the prediction changes monotonically as features are
    removed one-by-one in order of decreasing importance.

    Monotonicity = fraction of steps at which the change in prediction
    does not decrease (drop at step i+1 >= drop at step i).

    Parameters
    ----------
    model : fitted model
    X : np.ndarray
        Test instances
    feature_importances : np.ndarray
        Importance scores (higher = more important)
    X_train : np.ndarray, optional
        For computing mean baseline

    Returns
    -------
    float
        Monotonicity score (0 to 1, higher = more monotonic)
    """
    n_features = X.shape[1]

    # Sort features by importance (most important first)
    importance_order = np.argsort(feature_importances)[::-1]

    # Baseline
    if X_train is not None:
        baseline_values = np.mean(X_train, axis=0)
    else:
        baseline_values = np.mean(X, axis=0)

    # Original predictions
    original_probs = _get_proba(model, X)

    # Remove features one by one
    drops = []
    for i in range(n_features):
        features_to_mask = importance_order[: i + 1]
        X_masked = X.copy()
        X_masked[:, features_to_mask] = baseline_values[features_to_mask]
        masked_probs = _get_proba(model, X_masked)
        drop = np.mean(np.abs(original_probs - masked_probs))
        drops.append(drop)

    # Count monotonic increases
    n_monotonic = 0
    for i in range(1, len(drops)):
        if drops[i] >= drops[i - 1] - 1e-10:  # Small tolerance
            n_monotonic += 1

    monotonicity = n_monotonic / max(1, (len(drops) - 1))
    return monotonicity


def compute_feature_ranking_agreement(
    importances_a: np.ndarray,
    importances_b: np.ndarray,
    method: str = "kendall",
) -> float:
    """
    Compute agreement between two feature importance rankings.

    Parameters
    ----------
    importances_a, importances_b : np.ndarray
        Feature importance scores from two different XAI methods
    method : str
        "kendall" for Kendall's tau, "spearman" for Spearman's rho

    Returns
    -------
    float
        Correlation coefficient (-1 to 1)
    """
    if method == "kendall":
        tau, p_value = kendalltau(importances_a, importances_b)
        return tau
    elif method == "spearman":
        rho, p_value = spearmanr(importances_a, importances_b)
        return rho
    else:
        raise ValueError(f"Unknown method: {method}")


def measure_execution_time(func, *args, n_repetitions=3, **kwargs):
    """
    Measure execution time of a function.

    Parameters
    ----------
    func : callable
        Function to time
    *args, **kwargs
        Arguments to pass to func
    n_repetitions : int
        Number of repetitions for averaging

    Returns
    -------
    tuple
        (result, mean_time_seconds, std_time_seconds)
    """
    times = []
    result = None

    for i in range(n_repetitions):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return result, mean_time, std_time


def _get_proba(model, X):
    """Get probability predictions from a model."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2:
            return probs[:, 1]  # Probability of positive class
        return probs
    else:
        return model.predict(X)


def compute_all_metrics(
    model,
    X_test: np.ndarray,
    X_train: np.ndarray,
    feature_importances: dict,
    n_faithfulness_steps: int = 10,
) -> dict:
    """
    Compute all evaluation metrics for all XAI methods.

    Parameters
    ----------
    model : fitted model
    X_test : np.ndarray
    X_train : np.ndarray
    feature_importances : dict
        {method_name: importance_array} for each XAI method
    n_faithfulness_steps : int

    Returns
    -------
    dict
        Nested dict: {method: {metric: value}}
    """
    results = {}

    for method_name, importances in feature_importances.items():
        print(f"  Computing metrics for {method_name}...")

        faith = compute_faithfulness(
            model, X_test, importances,
            n_steps=n_faithfulness_steps,
            X_train=X_train,
        )

        mono = compute_monotonicity(
            model, X_test, importances,
            X_train=X_train,
        )

        results[method_name] = {
            "faithfulness": faith,
            "monotonicity": mono,
        }

        print(f"    Faithfulness: {faith:.4f}")
        print(f"    Monotonicity: {mono:.4f}")

    # Compute pairwise ranking agreement
    method_names = list(feature_importances.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            m_a, m_b = method_names[i], method_names[j]
            tau = compute_feature_ranking_agreement(
                feature_importances[m_a],
                feature_importances[m_b],
                method="kendall",
            )
            results.setdefault("ranking_agreement", {})[f"{m_a}_vs_{m_b}"] = tau
            print(f"    Kendall's τ ({m_a} vs {m_b}): {tau:.4f}")

    return results


# ----------------------------------------------------------------------------
# Single-feature faithfulness
# ----------------------------------------------------------------------------
# Global-ranking version of the faithfulness metric of Alvarez-Melis &
# Jaakkola (2018): correlation between the importance of each feature and the
# change in prediction when that feature alone is set to its training mean.

def single_feature_effects(model, X: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    """Mean absolute change in predicted probability over the rows of X when
    feature j alone is set to its baseline value, for every feature j.
    Returns an array of shape (n_features,)."""
    assert baseline_values.shape == (X.shape[1],)
    p = _get_proba(model, X)
    effects = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        X_j = X.copy()
        X_j[:, j] = baseline_values[j]
        effects[j] = np.mean(np.abs(p - _get_proba(model, X_j)))
    return effects


def faithfulness_correlation(feature_importances: np.ndarray, effects: np.ndarray) -> float:
    """Pearson correlation between global importance scores and the
    single-feature effects of `single_feature_effects`."""
    assert feature_importances.shape == effects.shape
    assert np.std(feature_importances) > 0 and np.std(effects) > 0
    return float(np.corrcoef(feature_importances, effects)[0, 1])

