"""
Spatial Block Cross-Validation
================================
Implements spatial block CV following Roberts et al. (2017).
Ensures that training and testing folds are spatially separated
to prevent spatial autocorrelation from inflating accuracy estimates.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def spatial_block_cv(
    coords: pd.DataFrame,
    n_folds: int = 5,
    block_size_km: float = 50,
    random_seed: int = 42,
) -> list:
    """
    Create spatial block cross-validation folds.

    Divides the study area into rectangular blocks, then assigns
    blocks to folds. Points within the same block go to the same fold.

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame with 'longitude' and 'latitude' columns
    n_folds : int
        Number of CV folds
    block_size_km : float
        Approximate block size in km
    random_seed : int
        Random seed for block-to-fold assignment

    Returns
    -------
    list of tuples
        Each tuple is (train_indices, test_indices) as numpy arrays
    """
    np.random.seed(random_seed)

    lons = coords["longitude"].values
    lats = coords["latitude"].values

    # Convert block size from km to degrees (approximate)
    mean_lat = np.mean(lats)
    block_size_lat = block_size_km / 111.0
    block_size_lon = block_size_km / (111.0 * np.cos(np.radians(mean_lat)))

    # Assign each point to a block
    block_ids_lon = np.floor(lons / block_size_lon).astype(int)
    block_ids_lat = np.floor(lats / block_size_lat).astype(int)

    # Create unique block ID per grid cell
    # Combine lon/lat block IDs into a single unique block ID
    block_ids = block_ids_lon * 10000 + block_ids_lat

    # Get unique blocks
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    print(f"\nSpatial Block CV:")
    print(f"  Block size: ~{block_size_km}km ({block_size_lon:.3f}° lon × {block_size_lat:.3f}° lat)")
    print(f"  Total blocks: {n_blocks}")
    print(f"  Folds: {n_folds}")

    # Randomly assign blocks to folds
    block_fold_assignment = {}
    shuffled_blocks = unique_blocks.copy()
    np.random.shuffle(shuffled_blocks)

    for i, block in enumerate(shuffled_blocks):
        block_fold_assignment[block] = i % n_folds

    # Create fold indices
    point_folds = np.array([block_fold_assignment[b] for b in block_ids])

    folds = []
    for fold_idx in range(n_folds):
        test_mask = point_folds == fold_idx
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        folds.append((train_indices, test_indices))

        print(f"  Fold {fold_idx + 1}: train={len(train_indices)}, test={len(test_indices)}")

    return folds


def random_cv(n_samples: int, n_folds: int = 5, random_seed: int = 42) -> list:
    """
    Standard random k-fold cross-validation (for comparison with spatial CV).

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_folds : int
        Number of folds
    random_seed : int
        Random seed

    Returns
    -------
    list of tuples
        Each tuple is (train_indices, test_indices)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(n_samples)

    folds = []
    for train_idx, test_idx in kf.split(indices):
        folds.append((train_idx, test_idx))

    print(f"\nRandom {n_folds}-fold CV: {n_samples} samples")
    return folds
