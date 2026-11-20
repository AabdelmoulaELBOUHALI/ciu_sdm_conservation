"""
Statistical ranking tools: Scott-Knott clustering and Borda count.

Scott-Knott (Scott and Knott 1974; Jelihovschi et al. 2014) partitions a set
of treatment means into non-overlapping homogeneous groups. The sorted means
are split at the point that maximises the between-group sum of squares B0,
and the split is kept when

    lambda = pi / (2 (pi - 2)) * B0 / sigma0^2

exceeds the chi-square quantile with k / (pi - 2) degrees of freedom, where
k is the number of means in the group being split and

    sigma0^2 = (sum_i (ybar_i - ybar)^2 + v * s2_mean) / (k + v),

with s2_mean = MSE / r the error variance of a treatment mean (MSE and its v
degrees of freedom come from the one-way ANOVA of all observations, r is the
number of observations per treatment). The procedure is applied recursively
to both sides of every accepted split.

Borda count aggregates several rankings by summing the ranks of each
candidate (rank 1 = best); the smallest sum is the best aggregate rank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def scott_knott(
    groups: dict,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Scott-Knott clustering of treatment means.

    Parameters
    ----------
    groups : dict[str, array-like]
        Treatment name -> observations (e.g. per-fold AUC values). Every
        treatment must have the same number (>= 2) of observations.
    alpha : float
        Significance level of the split test.
    higher_is_better : bool
        When True, cluster 1 contains the highest means.

    Returns
    -------
    pandas.DataFrame
        Columns ``treatment``, ``mean``, ``n``, ``rank``, ``cluster``, sorted
        from best to worst. Treatments sharing a cluster are not significantly
        different.
    """
    treatments = list(groups.keys())
    values = [np.asarray(groups[t], dtype=float) for t in treatments]
    n_obs = np.array([len(v) for v in values])
    if len(treatments) == 0:
        return pd.DataFrame(columns=["treatment", "mean", "n", "rank", "cluster"])
    if not np.all(n_obs == n_obs[0]) or n_obs[0] < 2:
        raise ValueError("Scott-Knott requires the same number (>= 2) of observations per treatment")

    means = np.array([v.mean() for v in values])
    n_total, k_total = int(n_obs.sum()), len(treatments)
    v_df = n_total - k_total                                     # error degrees of freedom
    mse = sum(((v - v.mean()) ** 2).sum() for v in values) / v_df
    s2_mean = mse / n_obs[0]                                     # error variance of a mean

    order = np.argsort(-means if higher_is_better else means, kind="stable")
    sorted_means = means[order]
    clusters = np.zeros(k_total, dtype=int)

    def recurse(lo: int, hi: int, cluster_id: int) -> int:
        k = hi - lo
        if k == 1:
            clusters[lo:hi] = cluster_id
            return cluster_id + 1
        sub = sorted_means[lo:hi]
        grand = sub.mean()
        best_b0, best_split = -np.inf, None
        for split in range(1, k):
            b0 = split * (sub[:split].mean() - grand) ** 2 + (k - split) * (sub[split:].mean() - grand) ** 2
            if b0 > best_b0:
                best_b0, best_split = b0, split
        sigma0_sq = (((sub - grand) ** 2).sum() + v_df * s2_mean) / (k + v_df)
        lam = np.pi / (2.0 * (np.pi - 2.0)) * best_b0 / sigma0_sq
        critical = stats.chi2.ppf(1.0 - alpha, df=k / (np.pi - 2.0))
        if lam <= critical:
            clusters[lo:hi] = cluster_id
            return cluster_id + 1
        next_id = recurse(lo, lo + best_split, cluster_id)
        return recurse(lo + best_split, hi, next_id)

    recurse(0, k_total, 1)
    return pd.DataFrame({
        "treatment": [treatments[i] for i in order],
        "mean": sorted_means.astype(float),
        "n": n_obs[order].astype(int),
        "rank": np.arange(1, k_total + 1),
        "cluster": clusters,
    })


def borda_count(
    score_matrix: pd.DataFrame,
    higher_is_better=True,
) -> pd.DataFrame:
    """Borda-count rank aggregation across criteria.

    Parameters
    ----------
    score_matrix : pandas.DataFrame
        Rows are candidates (models or explanation methods), columns are
        criteria. Ties receive averaged ranks.
    higher_is_better : bool or dict[str, bool]
        Direction of each criterion (a dict allows mixed directions, e.g.
        execution time where lower is better).

    Returns
    -------
    pandas.DataFrame
        Columns ``candidate``, per-criterion ranks, ``borda_sum`` and
        ``borda_rank``, sorted from best (smallest sum) to worst.
    """
    if score_matrix.empty:
        return pd.DataFrame()

    if isinstance(higher_is_better, bool):
        direction = {col: higher_is_better for col in score_matrix.columns}
    else:
        direction = {col: bool(higher_is_better[col]) for col in score_matrix.columns}

    ranks = pd.DataFrame(index=score_matrix.index)
    for col in score_matrix.columns:
        ranks[col] = score_matrix[col].astype(float).rank(ascending=not direction[col], method="average")

    out = ranks.copy()
    out["borda_sum"] = ranks.sum(axis=1)
    out["borda_rank"] = out["borda_sum"].rank(ascending=True, method="min").astype(int)
    out = out.sort_values("borda_sum").reset_index()
    return out.rename(columns={out.columns[0]: "candidate"})
