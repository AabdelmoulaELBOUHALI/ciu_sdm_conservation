"""
Global Surrogate Decision Tree
===============================
Fits a compact decision tree that approximates a black-box SDM and measures:

- Fidelity: accuracy of the surrogate predictions vs. the black-box predictions
  on a held-out oracle dataset (i.e. how faithfully the tree mimics the model).
- Comprehensibility: tree depth and number of leaves (smaller = easier to read).
- Feature importance: the surrogate's Gini importances, comparable in rank with
  SHAP / LIME / ALE / CIU global importances.

Also provides a white-box decision tree fitted directly on the observed
labels, used as a transparent reference.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def _predict_classes(model, X: np.ndarray) -> np.ndarray:
    """Return hard predictions from ``model``, using predict_proba when available."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return (probs[:, 1] >= 0.5).astype(int)
    return np.asarray(model.predict(X)).astype(int)


def fit_global_surrogate(
    black_box_model,
    X_train: np.ndarray,
    X_oracle: np.ndarray,
    feature_names,
    max_depth: int = 4,
    random_state: int = 42,
):
    """Fit a decision tree that imitates ``black_box_model`` predictions.

    Parameters
    ----------
    black_box_model : fitted classifier
        Must expose predict / predict_proba.
    X_train : np.ndarray
        Inputs used to query the black-box and train the surrogate.
    X_oracle : np.ndarray
        Held-out inputs used only for fidelity evaluation.
    feature_names : sequence of str
        Names for reporting feature importance.
    max_depth : int
        Cap on surrogate tree depth for comprehensibility.

    Returns
    -------
    dict with keys:
        ``surrogate``: fitted DecisionTreeClassifier,
        ``fidelity``: accuracy against black-box on ``X_oracle``,
        ``depth``: tree depth,
        ``n_leaves``: leaf count,
        ``feature_importance``: dict of feature -> Gini importance.
    """
    feature_names = list(feature_names)
    y_train_hat = _predict_classes(black_box_model, X_train)
    y_oracle_hat = _predict_classes(black_box_model, X_oracle)

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    # Fall back to a constant prediction when the black-box is degenerate
    # (only one class produced on the training pool) so downstream reporting
    # still returns a valid surrogate object.
    unique = np.unique(y_train_hat)
    if unique.size < 2:
        constant_cls = int(unique[0]) if unique.size == 1 else 0
        surrogate.fit(X_train, np.full(len(X_train), constant_cls))
    else:
        surrogate.fit(X_train, y_train_hat)

    surrogate_pred = surrogate.predict(X_oracle)
    fidelity = accuracy_score(y_oracle_hat, surrogate_pred)

    importance = surrogate.feature_importances_
    feature_importance = {
        feature_names[i]: float(importance[i]) for i in range(len(feature_names))
    }

    return {
        "surrogate": surrogate,
        "fidelity": float(fidelity),
        "depth": int(surrogate.get_depth()),
        "n_leaves": int(surrogate.get_n_leaves()),
        "feature_importance": feature_importance,
    }


def fit_whitebox_tree(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    max_depth: int = 4,
    random_state: int = 42,
):
    """Fit a decision tree directly on ground-truth labels.

    Serves as a transparent reference for the surrogate and the global
    rankings of the explanation methods.
    """
    feature_names = list(feature_names)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    importance = tree.feature_importances_
    feature_importance = {
        feature_names[i]: float(importance[i]) for i in range(len(feature_names))
    }
    return {
        "tree": tree,
        "depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "feature_importance": feature_importance,
        "training_accuracy": float(accuracy_score(y, tree.predict(X))),
    }
