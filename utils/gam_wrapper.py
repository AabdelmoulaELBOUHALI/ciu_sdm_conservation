"""
sklearn-compatible GAM Classifier Wrapper
==========================================
Wraps pygam.LogisticGAM to expose the standard sklearn interface:
    .fit(X, y)
    .predict(X)         -> binary labels
    .predict_proba(X)   -> (n_samples, 2) probability array

This wrapper enables GAM to be used identically to RF/XGBoost/MLP
throughout the pipeline (training, XAI, evaluation).

References:
- pygam: https://pygam.readthedocs.io
- Hastie & Tibshirani (1990). Generalized Additive Models.
"""

import numpy as np
import pickle


class GAMClassifierWrapper:
    """
    Sklearn-compatible wrapper for pygam LogisticGAM.

    Parameters
    ----------
    n_splines : int
        Number of spline basis functions per feature (default 20).
    lam : float
        Regularisation strength (lambda). Larger = smoother.
    """

    def __init__(self, n_splines: int = 20, lam: float = 0.6):
        self.n_splines = n_splines
        self.lam = lam
        self._model = None
        self.classes_ = np.array([0, 1])

    # ------------------------------------------------------------------
    # Core sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit LogisticGAM to training data.

        If optimisation diverges at the specified lam, automatically retries
        with progressively stronger regularisation (lam * 10, lam * 100).
        """
        try:
            from pygam import LogisticGAM, s
        except ImportError as e:
            raise ImportError(
                "pygam is required for GAM models. Install with: pip install pygam"
            ) from e

        n_features = X.shape[1]

        def _build_and_fit(lam_val):
            terms = s(0, n_splines=self.n_splines, lam=lam_val)
            for i in range(1, n_features):
                terms = terms + s(i, n_splines=self.n_splines, lam=lam_val)
            mdl = LogisticGAM(terms)
            mdl.fit(X, y)
            return mdl

        lam_candidates = [self.lam, self.lam * 10, self.lam * 100, self.lam * 1000]
        last_error = None
        for lam_val in lam_candidates:
            try:
                self._model = _build_and_fit(lam_val)
                if lam_val != self.lam:
                    print(f"      GAM: converged with lam={lam_val:.1f} (original={self.lam})")
                return self
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"GAM optimisation diverged even with lam={lam_candidates[-1]}. "
            f"Last error: {last_error}"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities as (n_samples, 2) array.
        Column 0 = P(absence), Column 1 = P(presence).
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proba_class1 = self._model.predict_proba(X)
        proba_class0 = 1.0 - proba_class1
        return np.column_stack([proba_class0, proba_class1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class predictions (threshold at 0.5)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Pickle support (required for saving models with pickle.dump)
    # ------------------------------------------------------------------

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"GAMClassifierWrapper(n_splines={self.n_splines}, lam={self.lam})"
