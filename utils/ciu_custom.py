"""
Custom CIU (Contextual Importance and Utility) Implementation
===============================================================
Implements CIU following Främling (1996, 2022) theory.
This avoids pandas version incompatibility with the py-ciu package.

CIU Theory:
-----------
For an instance x, feature i, and output j:

- cmin_j(i) = min output when varying feature i (other features fixed)
- cmax_j(i) = max output when varying feature i (other features fixed)
- absmin_j = minimum of output j (estimated from the predictions on the training data)
- absmax_j = maximum of output j (estimated from the predictions on the training data)

Contextual Importance (CI):
    CI_j(i) = (cmax_j(i) - cmin_j(i)) / (absmax_j - absmin_j)
    Range [0, 1]. How much could feature i change the output?

Contextual Utility (CU):
    CU_j(i) = (output_j(x) - cmin_j(i)) / (cmax_j(i) - cmin_j(i))
    Range [0, 1]. Is feature i currently at a favourable value?

References:
- Främling, K. (1996). Explaining Results of Neural Networks by Contextual Importance and Utility.
- Främling, K. (2022). Contextual Importance and Utility: A Theoretical Foundation.
"""

import numpy as np
from typing import Callable, List, Optional


class CIUExplainer:
    """
    Contextual Importance and Utility (CIU) explainer.

    Parameters
    ----------
    predictor : callable
        Function that takes input array (n_samples, n_features)
        and returns probabilities (n_samples,) or (n_samples, n_classes)
    feature_names : list of str
        Names of input features
    data : np.ndarray
        Training data (n_samples, n_features) for deriving min/max ranges
    nsamples : int
        Number of Monte Carlo samples for estimating cmin/cmax
    output_index : int
        Index of the output class to explain (default=1 for binary classification)
    """

    def __init__(
        self,
        predictor: Callable,
        feature_names: List[str],
        data: np.ndarray,
        nsamples: int = 1000,
        output_index: int = 1,
    ):
        self.predictor = predictor
        self.feature_names = feature_names
        self.data = data
        self.nsamples = nsamples
        self.output_index = output_index
        self.n_features = data.shape[1]

        # Compute feature min/max from training data
        self.feature_mins = data.min(axis=0)
        self.feature_maxs = data.max(axis=0)

        # Estimate absolute output min/max
        preds = self._predict(data)
        self.absmin = preds.min()
        self.absmax = preds.max()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for the target output class."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        result = self.predictor(X)
        if result.ndim == 2:
            return result[:, self.output_index]
        return result

    def explain_instance(
        self,
        instance: np.ndarray,
        nsamples: Optional[int] = None,
    ) -> dict:
        """
        Compute CIU values (CI and CU) for all features of one instance.

        Parameters
        ----------
        instance : np.ndarray, shape (n_features,)
        nsamples : int, optional
            Override default nsamples

        Returns
        -------
        dict with keys:
            'CI': np.ndarray (n_features,): Contextual Importance
            'CU': np.ndarray (n_features,): Contextual Utility
            'cmin': np.ndarray (n_features,): Contextual minimum
            'cmax': np.ndarray (n_features,): Contextual maximum
            'output': float: Current prediction for this instance
        """
        if nsamples is None:
            nsamples = self.nsamples

        instance = np.asarray(instance).ravel()
        current_output = self._predict(instance)[0]

        CI = np.zeros(self.n_features)
        CU = np.zeros(self.n_features)
        cmin_vals = np.zeros(self.n_features)
        cmax_vals = np.zeros(self.n_features)

        for i in range(self.n_features):
            cmin, cmax = self._estimate_contextual_minmax(instance, i, nsamples)
            cmin_vals[i] = cmin
            cmax_vals[i] = cmax

            # Contextual Importance
            output_range = self.absmax - self.absmin
            if output_range > 1e-10:
                CI[i] = (cmax - cmin) / output_range
            else:
                CI[i] = 0

            # Contextual Utility
            context_range = cmax - cmin
            if context_range > 1e-10:
                CU[i] = (current_output - cmin) / context_range
            else:
                CU[i] = 0.5  # Neutral if feature has no effect

            # Clip to [0, 1]
            CI[i] = np.clip(CI[i], 0, 1)
            CU[i] = np.clip(CU[i], 0, 1)

        return {
            "CI": CI,
            "CU": CU,
            "cmin": cmin_vals,
            "cmax": cmax_vals,
            "output": current_output,
        }

    def _estimate_contextual_minmax(
        self, instance: np.ndarray, feature_idx: int, nsamples: int
    ) -> tuple:
        """
        Estimate cmin and cmax for a given feature by Monte Carlo sampling.

        Fix all features except feature_idx, then sample feature_idx
        uniformly across its range to find min/max output.
        """
        # Create sample matrix: repeat instance nsamples times
        samples = np.tile(instance, (nsamples, 1))

        # Vary the target feature uniformly across its range
        f_min = self.feature_mins[feature_idx]
        f_max = self.feature_maxs[feature_idx]
        samples[:, feature_idx] = np.random.uniform(f_min, f_max, nsamples)

        # Predict for all samples
        preds = self._predict(samples)

        return preds.min(), preds.max()

    def explain_dataset(
        self,
        X: np.ndarray,
        n_explain: int = 200,
        random_seed: int = 42,
    ) -> dict:
        """
        Explain multiple instances and return aggregated results.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
        n_explain : int
        random_seed : int

        Returns
        -------
        dict with keys:
            'global_CI': np.ndarray (n_features,): Mean CI
            'global_CU': np.ndarray (n_features,): Mean CU
            'local_CI': np.ndarray (n_explain, n_features)
            'local_CU': np.ndarray (n_explain, n_features)
            'explain_indices': np.ndarray (n_explain,)
        """
        np.random.seed(random_seed)
        n_explain = min(n_explain, X.shape[0])
        explain_indices = np.random.choice(X.shape[0], size=n_explain, replace=False)

        all_CI = np.zeros((n_explain, self.n_features))
        all_CU = np.zeros((n_explain, self.n_features))

        for i, idx in enumerate(explain_indices):
            result = self.explain_instance(X[idx])
            all_CI[i] = result["CI"]
            all_CU[i] = result["CU"]

            if (i + 1) % 50 == 0:
                print(f"      CIU: Explained {i+1}/{n_explain} instances...")

        return {
            "global_CI": np.mean(all_CI, axis=0),
            "global_CU": np.mean(all_CU, axis=0),
            "local_CI": all_CI,
            "local_CU": all_CU,
            "explain_indices": explain_indices,
        }
