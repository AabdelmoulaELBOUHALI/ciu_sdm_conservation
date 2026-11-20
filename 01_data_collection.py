"""
Step 1: Data Collection
========================
Downloads species occurrence data from GBIF and bioclimatic data from WorldClim.
This script is real-data only and does not generate synthetic occurrences or bioclim values.

Output:
- data/background_grid.csv
- data/{species}_raw_occurrences.csv
- data/{species}_clean_occurrences.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SPECIES,
    STUDY_AREA,
    DATA_DIR,
    WORLDCLIM_RESOLUTION,
    PREPROCESSING,
    DATA_COLLECTION,
)


def collect_real_gbif(sp_key, sp_info, bbox, worldclim_dir):
    """
    Download and prepare real GBIF + WorldClim data for one species.
    """
    from utils.gbif_utils import download_gbif_occurrences, clean_occurrences
    from utils.raster_utils import extract_bioclim_values

    print(f"  Downloading GBIF data for {sp_info['gbif_name']}...")
    raw_df = download_gbif_occurrences(
        species_name=sp_info["gbif_name"],
        bbox=bbox,
        max_records=DATA_COLLECTION["gbif_max_records"],
    )

    if raw_df.empty:
        raise ValueError("No GBIF records returned.")

    # Save raw records
    raw_path = os.path.join(DATA_DIR, f"{sp_key}_raw_occurrences.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"  Raw records saved: {raw_path} ({len(raw_df)} records)")

    # Clean occurrences: coordinate, bounding-box, year and uncertainty filters,
    # then thinning on a ~2 km grid (approximation of spThin).
    clean_df = clean_occurrences(
        raw_df,
        bbox=bbox,
        min_year=1970,
        remove_duplicates_resolution_km=PREPROCESSING["spatial_thinning_km"],
    )

    if len(clean_df) < DATA_COLLECTION["min_clean_records"]:
        raise ValueError(f"Too few clean records ({len(clean_df)}) after cleaning.")

    # Extract WorldClim bioclimatic values
    print("  Extracting WorldClim bioclim values...")
    clean_df = extract_bioclim_values(clean_df, worldclim_dir=worldclim_dir)
    bioclim_cols = [c for c in clean_df.columns if c.startswith("bio")]
    clean_df = clean_df.dropna(subset=bioclim_cols).reset_index(drop=True)

    if len(clean_df) < DATA_COLLECTION["min_clean_records"]:
        raise ValueError(
            f"Too few records ({len(clean_df)}) after removing rows with missing bioclim values."
        )
    return clean_df


def main():
    from utils.raster_utils import download_worldclim, extract_bioclim_for_grid

    print("=" * 70)
    print("STEP 1: DATA COLLECTION")
    print("CIU for Species Distribution Models - Extended Pipeline")
    print(f"Species: {len(SPECIES)} | Mode: Real GBIF + Real WorldClim")
    print("=" * 70)

    # Background grid (same for all species)
    worldclim_dir = download_worldclim(DATA_DIR, resolution=WORLDCLIM_RESOLUTION)
    bg = extract_bioclim_for_grid(worldclim_dir, STUDY_AREA)
    bg_path = os.path.join(DATA_DIR, "background_grid.csv")
    bg.to_csv(bg_path, index=False)
    print(f"Background grid: {len(bg)} cells -> {bg_path}")

    failures = []

    for sp_key, sp_info in SPECIES.items():
        print(f"\n{'='*60}")
        print(f"Species: {sp_info['gbif_name']} ({sp_info['common_name']})")
        print(f"{'='*60}")

        try:
            clean_df = collect_real_gbif(sp_key, sp_info, STUDY_AREA, worldclim_dir=worldclim_dir)
        except Exception as e:
            failures.append((sp_key, str(e)))
            print(f"  FAIL Real data collection failed for {sp_info['gbif_name']}: {e}")
            continue

        clean_path = os.path.join(DATA_DIR, f"{sp_key}_clean_occurrences.csv")
        clean_df.to_csv(clean_path, index=False)
        print(f"  Clean occurrences saved: {clean_path} ({len(clean_df)} records)")

    if failures:
        print("\nThe following species failed real data collection:")
        for sp_key, error in failures:
            print(f"- {sp_key}: {error}")
        if DATA_COLLECTION["fail_on_species_error"]:
            raise RuntimeError(
                "Real-data-only pipeline failed for one or more species. "
                "No synthetic fallback was used."
            )
        print("\nContinuing with successfully collected species only.")

    print(f"\n{'='*70}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
