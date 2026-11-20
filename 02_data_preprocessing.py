"""
Step 2: Data Preprocessing
============================
1. Filter extreme environmental outliers among the presences (IQR method)
2. Generate pseudo-absences (distance-constrained method)
3. Remove collinear variables (VIF filtering)
4. Standardize features
5. Create spatial block cross-validation folds

Output:
- data/{species}_dataset.npz (X, X_unscaled, y, coordinates, feature_names, scaler)
- data/{species}_spatial_folds.npz (train and test indices of each fold)
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SPECIES, DATA_DIR, PSEUDO_ABSENCE, SPATIAL_CV,
    BIOCLIM_VARS, MAX_VIF_THRESHOLD, RANDOM_SEED, PREPROCESSING
)
from utils.pseudo_absence import generate_pseudo_absences, combine_presence_absence
from utils.spatial_cv import spatial_block_cv


def compute_vif(X, feature_names):
    """
    Compute Variance Inflation Factor for each feature.
    VIF > 10 indicates problematic multicollinearity.
    """
    from numpy.linalg import LinAlgError

    vif_values = {}
    for i, name in enumerate(feature_names):
        # VIF = 1 / (1 - R²) where R² is from regressing feature i on all others
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)

        # Add intercept
        X_others_int = np.column_stack([np.ones(len(y)), X_others])

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_others_int, y, rcond=None)[0]
            y_pred = X_others_int @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                vif_values[name] = float("inf")
            else:
                r_squared = 1 - ss_res / ss_tot
                vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
                vif_values[name] = vif
        except (LinAlgError, ValueError):
            vif_values[name] = float("inf")

    return vif_values


def stepwise_vif_selection(X, feature_names, max_vif=10):
    """
    Iteratively remove features with highest VIF until all VIF < threshold.
    """
    print(f"\nVIF-based feature selection (threshold={max_vif})...")
    current_features = list(feature_names)
    current_X = X.copy()
    removed = []

    while True:
        vif_values = compute_vif(current_X, current_features)

        # Find max VIF
        max_feature = max(vif_values, key=vif_values.get)
        max_val = vif_values[max_feature]

        if max_val <= max_vif:
            break

        # Remove feature with highest VIF
        idx = current_features.index(max_feature)
        current_features.pop(idx)
        current_X = np.delete(current_X, idx, axis=1)
        removed.append((max_feature, max_val))
        print(f"  Removed {max_feature} (VIF={max_val:.1f})")

    print(f"  OK Kept {len(current_features)} features: {current_features}")
    if removed:
        print(f"  Removed {len(removed)} features: {[r[0] for r in removed]}")

    # Print final VIF values
    final_vif = compute_vif(current_X, current_features)
    print(f"\n  Final VIF values:")
    for name, vif in sorted(final_vif.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {vif:.2f}")

    return current_X, current_features


def apply_iqr_filter(df, feature_cols, multiplier=1.5):
    """
    Remove multivariate outliers with the IQR rule.

    A record is kept only if every column lies within
    [Q1 - multiplier * IQR, Q3 + multiplier * IQR].
    """
    if df.empty or not feature_cols:
        return df

    mask = np.ones(len(df), dtype=bool)
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask &= df[col].between(lower, upper)

    filtered = df.loc[mask].reset_index(drop=True)
    print(f"  IQR filter kept {len(filtered)}/{len(df)} records")
    return filtered


def main():
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("CIU for species distribution models")
    print("=" * 70)

    # Load background grid
    bg_path = os.path.join(DATA_DIR, "background_grid.csv")
    if not os.path.exists(bg_path):
        print("FAIL Background grid not found. Run 01_data_collection.py first.")
        return

    background_df = pd.read_csv(bg_path)
    bioclim_cols = [c for c in background_df.columns if c.startswith("bio")]
    print(f"Background grid: {len(background_df)} cells, {len(bioclim_cols)} bioclim vars")

    for sp_key, sp_info in SPECIES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {sp_info['gbif_name']} ({sp_info['common_name']})")
        print(f"{'='*60}")

        # Load clean occurrences
        occ_path = os.path.join(DATA_DIR, f"{sp_key}_clean_occurrences.csv")
        if not os.path.exists(occ_path):
            print(f"  FAIL No data file found. Skipping.")
            continue

        presence_df = pd.read_csv(occ_path)
        print(f"  Presence records: {len(presence_df)}")

        # Remove extreme environmental outliers among the presence records
        # (all 19 bioclimatic variables) before pseudo-absences are drawn.
        if PREPROCESSING["apply_iqr_filter"]:
            presence_df = apply_iqr_filter(
                presence_df,
                [c for c in BIOCLIM_VARS if c in presence_df.columns],
                multiplier=PREPROCESSING["iqr_multiplier"],
            )
            if len(presence_df) < 30:
                print("  FAIL Too few presence records after IQR filtering. Skipping.")
                continue

        # ---- 2.1 Generate pseudo-absences ----
        n_pseudo = int(len(presence_df) * PSEUDO_ABSENCE["n_pseudo_absences_ratio"])
        absence_df = generate_pseudo_absences(
            presence_df=presence_df,
            background_df=background_df,
            method=PSEUDO_ABSENCE["method"],
            n_pseudo=n_pseudo,
            min_distance_km=PSEUDO_ABSENCE["min_distance_km"],
            max_distance_km=PSEUDO_ABSENCE["max_distance_km"],
            random_seed=PSEUDO_ABSENCE["random_seed"],
        )

        # ---- 2.2 Combine into training dataset ----
        X, y, coords, feature_names = combine_presence_absence(
            presence_df, absence_df, bioclim_cols
        )

        # ---- 2.3 VIF-based collinearity filtering ----
        X_filtered, filtered_names = stepwise_vif_selection(
            X, feature_names, max_vif=MAX_VIF_THRESHOLD
        )

        # ---- 2.4 Standardize features ----
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        print(f"\n  Features standardized (mean=0, std=1)")

        # ---- 2.5 Spatial block cross-validation ----
        spatial_folds = spatial_block_cv(
            coords=coords,
            n_folds=SPATIAL_CV["n_folds"],
            block_size_km=SPATIAL_CV["block_size_km"],
            random_seed=SPATIAL_CV["random_seed"],
        )

        # ---- 2.6 Save processed dataset ----
        save_path = os.path.join(DATA_DIR, f"{sp_key}_dataset.npz")

        # Convert folds to saveable format
        fold_train = [f[0] for f in spatial_folds]
        fold_test = [f[1] for f in spatial_folds]

        np.savez(
            save_path,
            X=X_scaled,
            X_unscaled=X_filtered,
            y=y,
            coords_lon=coords["longitude"].values,
            coords_lat=coords["latitude"].values,
            feature_names=np.array(filtered_names),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            # Save folds as object arrays
            n_folds=len(spatial_folds),
        )

        # Save folds separately (variable length arrays)
        folds_path = os.path.join(DATA_DIR, f"{sp_key}_spatial_folds.npz")
        fold_dict = {}
        for i, (train_idx, test_idx) in enumerate(spatial_folds):
            fold_dict[f"train_{i}"] = train_idx
            fold_dict[f"test_{i}"] = test_idx
        np.savez(folds_path, **fold_dict)

        print(f"\n  OK Saved dataset: {save_path}")
        print(f"    Shape: X={X_scaled.shape}, y={y.shape}")
        print(f"    Features: {filtered_names}")
        print(f"    Presences: {np.sum(y==1)}, Pseudo-absences: {np.sum(y==0)}")
        print(f"    Spatial CV folds: {len(spatial_folds)}")

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
