"""
Pseudo-Absence Generation for SDMs
====================================
Implements multiple pseudo-absence generation strategies:
1. Random background sampling
2. Distance-constrained sampling
3. Target-group sampling (not implemented: falls back to distance-constrained sampling)

Following best practices from:
- Barbet-Massin et al. (2012)
- Broussin et al. (2024) - EcoPA
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def generate_pseudo_absences(
    presence_df: pd.DataFrame,
    background_df: pd.DataFrame,
    method: str = "distance_constrained",
    n_pseudo: int = None,
    ratio: float = 1.0,
    min_distance_km: float = 10,
    max_distance_km: float = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate pseudo-absence points for SDM training.

    Parameters
    ----------
    presence_df : pd.DataFrame
        Presence records with 'longitude', 'latitude', and bioclim columns
    background_df : pd.DataFrame
        Background grid points with 'longitude', 'latitude', and bioclim columns
    method : str
        One of: "random", "distance_constrained", "target_group"
    n_pseudo : int, optional
        Exact number of pseudo-absences. If None, uses ratio.
    ratio : float
        Ratio of pseudo-absences to presences (1.0 = equal)
    min_distance_km : float
        Minimum distance from any presence point (for distance_constrained)
    max_distance_km : float
        Maximum distance from any presence point (for distance_constrained)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Pseudo-absence points with same columns as presence_df
    """
    np.random.seed(random_seed)

    if n_pseudo is None:
        n_pseudo = int(len(presence_df) * ratio)

    print(f"\nGenerating {n_pseudo} pseudo-absences using '{method}' method...")

    if method == "random":
        return _random_background(background_df, n_pseudo, random_seed)
    elif method == "distance_constrained":
        return _distance_constrained(
            presence_df, background_df, n_pseudo,
            min_distance_km, max_distance_km, random_seed
        )
    elif method == "target_group":
        # Target-group sampling needs occurrences of related taxa, which the
        # pipeline does not collect; the distance-constrained method is used.
        print("  -> Target-group method requires additional taxa data.")
        print("  -> Falling back to distance-constrained method.")
        return _distance_constrained(
            presence_df, background_df, n_pseudo,
            min_distance_km, max_distance_km, random_seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _random_background(background_df, n_pseudo, random_seed):
    """Simple random sampling from background grid."""
    if len(background_df) < n_pseudo:
        print(f"  WARN Not enough background points ({len(background_df)}), using all")
        return background_df.copy()

    sample = background_df.sample(n=n_pseudo, random_state=random_seed)
    print(f"  OK Sampled {len(sample)} random background points")
    return sample.reset_index(drop=True)


def _distance_constrained(
    presence_df, background_df, n_pseudo,
    min_distance_km, max_distance_km, random_seed
):
    """
    Sample pseudo-absences at a constrained distance from presences.
    Points must be between min_distance_km and max_distance_km from
    the nearest presence point.
    """
    # Get presence coordinates
    pres_coords = presence_df[["longitude", "latitude"]].values
    bg_coords = background_df[["longitude", "latitude"]].values

    # Calculate distances from each background point to nearest presence
    # Using approximate km conversion (1° lat ≈ 111 km)
    # Scale longitude by cos(mean_latitude) for better approximation
    mean_lat = np.mean(pres_coords[:, 1])
    lon_scale = np.cos(np.radians(mean_lat))

    pres_scaled = pres_coords.copy()
    pres_scaled[:, 0] *= lon_scale
    bg_scaled = bg_coords.copy()
    bg_scaled[:, 0] *= lon_scale

    # Compute min distance to any presence point (in degrees, then convert)
    print("  Computing distances to nearest presence point...")

    # Process in chunks to avoid memory issues
    chunk_size = 5000
    min_distances_km = np.zeros(len(bg_coords))

    for i in range(0, len(bg_coords), chunk_size):
        end = min(i + chunk_size, len(bg_coords))
        chunk_scaled = bg_scaled[i:end]
        dists = cdist(chunk_scaled, pres_scaled, metric="euclidean")
        min_dists_deg = dists.min(axis=1)
        min_distances_km[i:end] = min_dists_deg * 111  # Approximate km

    # Filter by distance constraints
    valid_mask = (min_distances_km >= min_distance_km) & (min_distances_km <= max_distance_km)
    valid_bg = background_df[valid_mask]

    print(f"  Points within {min_distance_km}-{max_distance_km}km: {len(valid_bg)}")

    if len(valid_bg) < n_pseudo:
        print(f"  WARN Not enough valid points ({len(valid_bg)}), relaxing constraints...")
        # Relax: use all points beyond min_distance
        valid_mask = min_distances_km >= min_distance_km
        valid_bg = background_df[valid_mask]
        print(f"  Points beyond {min_distance_km}km: {len(valid_bg)}")

    if len(valid_bg) < n_pseudo:
        print(f"  WARN Still not enough ({len(valid_bg)}), using all available")
        return valid_bg.reset_index(drop=True)

    sample = valid_bg.sample(n=n_pseudo, random_state=random_seed)
    print(f"  OK Sampled {len(sample)} distance-constrained pseudo-absences")
    return sample.reset_index(drop=True)


def combine_presence_absence(
    presence_df: pd.DataFrame,
    absence_df: pd.DataFrame,
    bioclim_cols: list,
) -> tuple:
    """
    Combine presence and pseudo-absence data into training-ready format.

    Returns
    -------
    X : np.ndarray
        Feature matrix (bioclimatic variables)
    y : np.ndarray
        Labels (1 = presence, 0 = pseudo-absence)
    coords : pd.DataFrame
        Coordinates for spatial CV
    feature_names : list
        Names of bioclimatic variables used
    """
    # Add labels
    pres = presence_df.copy()
    pres["presence"] = 1
    abse = absence_df.copy()
    abse["presence"] = 0

    # Combine
    combined = pd.concat([pres, abse], ignore_index=True)

    # Ensure all bioclim columns are present
    available_cols = [c for c in bioclim_cols if c in combined.columns]
    if len(available_cols) < len(bioclim_cols):
        missing = set(bioclim_cols) - set(available_cols)
        print(f"  WARN Missing bioclim columns: {missing}")

    # Drop rows with NaN in bioclim columns
    combined = combined.dropna(subset=available_cols).reset_index(drop=True)

    X = combined[available_cols].values
    y = combined["presence"].values
    coords = combined[["longitude", "latitude"]]
    feature_names = available_cols

    print(f"\n  Combined dataset: {len(combined)} samples")
    print(f"    Presences: {np.sum(y == 1)}")
    print(f"    Pseudo-absences: {np.sum(y == 0)}")
    print(f"    Features: {len(feature_names)}")

    return X, y, coords, feature_names
