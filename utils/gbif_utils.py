"""
GBIF Data Download Utilities
=============================
Functions to download and clean species occurrence data from GBIF.
Uses the GBIF REST API (requests).
"""

import pandas as pd
import numpy as np
import time
import os
import requests


GBIF_API_BASE = "https://api.gbif.org/v1"


def get_gbif_taxon_key(species_name: str) -> int:
    """Get GBIF taxon key for a species name."""
    url = f"{GBIF_API_BASE}/species/match"
    result = requests.get(url, params={"name": species_name, "rank": "species"}, timeout=60).json()

    usage_key = result.get("usageKey")
    if usage_key is None:
        raise ValueError(f"Species not found in GBIF match endpoint: {species_name}")

    if result.get("matchType") == "EXACT":
        print(f"  OK Found: {result.get('scientificName', species_name)} (key: {usage_key})")
    else:
        print(f"  WARN Fuzzy match: {result.get('scientificName', species_name)} (key: {usage_key})")

    return usage_key


def download_gbif_occurrences(
    species_name: str,
    bbox: dict,
    max_records: int = 5000,
    has_coordinate: bool = True,
    has_geospatial_issue: bool = False,
) -> pd.DataFrame:
    """
    Download occurrence records from GBIF for a species within a bounding box.

    Parameters
    ----------
    species_name : str
        Scientific name (e.g., "Quercus ilex")
    bbox : dict
        Bounding box with keys: min_lon, max_lon, min_lat, max_lat
    max_records : int
        Maximum number of records to download
    has_coordinate : bool
        Only records with coordinates
    has_geospatial_issue : bool
        Exclude records with geospatial issues

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: species, longitude, latitude, year, basisOfRecord
    """
    print(f"\n{'='*60}")
    print(f"Downloading GBIF data for: {species_name}")
    print(f"{'='*60}")

    # Get taxon key
    taxon_key = get_gbif_taxon_key(species_name)
    if taxon_key is None:
        raise ValueError(f"Species not found in GBIF: {species_name}")

    # Build geometry WKT for bounding box
    geometry = (
        f"POLYGON(("
        f"{bbox['min_lon']} {bbox['min_lat']}, "
        f"{bbox['max_lon']} {bbox['min_lat']}, "
        f"{bbox['max_lon']} {bbox['max_lat']}, "
        f"{bbox['min_lon']} {bbox['max_lat']}, "
        f"{bbox['min_lon']} {bbox['min_lat']}"
        f"))"
    )

    # Download in batches (GBIF API returns max 300 per call)
    all_records = []
    offset = 0
    batch_size = 300

    while offset < max_records:
        try:
            url = f"{GBIF_API_BASE}/occurrence/search"
            params = {
                "taxonKey": taxon_key,
                "geometry": geometry,
                "hasCoordinate": str(has_coordinate).lower(),
                "hasGeospatialIssue": str(has_geospatial_issue).lower(),
                "limit": min(batch_size, max_records - offset),
                "offset": offset,
            }
            result = requests.get(url, params=params, timeout=120).json()

            records = result.get("results", [])
            if not records:
                break

            all_records.extend(records)
            offset += len(records)

            # Check if there are more records
            if result.get("endOfRecords", True):
                break

            print(f"  Downloaded {offset} records...")
            time.sleep(0.5)  # pause between requests (GBIF API rate limits)

        except Exception as e:
            print(f"  WARN Error at offset {offset}: {e}")
            time.sleep(2)
            continue

    if not all_records:
        print(f"  FAIL No records found for {species_name}")
        return pd.DataFrame()

    # Extract relevant fields
    data = []
    for rec in all_records:
        data.append({
            "species": rec.get("species", species_name),
            "longitude": rec.get("decimalLongitude"),
            "latitude": rec.get("decimalLatitude"),
            "year": rec.get("year"),
            "basisOfRecord": rec.get("basisOfRecord"),
            "coordinateUncertaintyInMeters": rec.get("coordinateUncertaintyInMeters"),
            "institutionCode": rec.get("institutionCode"),
        })

    df = pd.DataFrame(data)
    print(f"  OK Total raw records: {len(df)}")

    return df


def clean_occurrences(
    df: pd.DataFrame,
    bbox: dict = None,
    min_year: int = 1970,
    max_coord_uncertainty_m: float = 10000,
    remove_duplicates_resolution_km: float = 1.0,
) -> pd.DataFrame:
    """
    Clean GBIF occurrence data following best practices for SDMs.

    Cleaning steps:
    1. Remove records without coordinates
    2. Remove records outside bounding box
    3. Remove records before min_year
    4. Remove records with high coordinate uncertainty
    5. Remove spatial duplicates (one record per grid cell)

    Parameters
    ----------
    df : pd.DataFrame
        Raw GBIF occurrence data
    bbox : dict, optional
        Bounding box for spatial filtering
    min_year : int
        Minimum collection year
    max_coord_uncertainty_m : float
        Maximum coordinate uncertainty in meters
    remove_duplicates_resolution_km : float
        Spatial thinning resolution in km

    Returns
    -------
    pd.DataFrame
        Cleaned occurrence data
    """
    print(f"\nCleaning occurrence data...")
    n_start = len(df)

    # 1. Remove records without coordinates
    df = df.dropna(subset=["longitude", "latitude"])
    print(f"  After removing no-coords: {len(df)} ({n_start - len(df)} removed)")

    # 2. Filter by bounding box
    if bbox:
        df = df[
            (df["longitude"] >= bbox["min_lon"])
            & (df["longitude"] <= bbox["max_lon"])
            & (df["latitude"] >= bbox["min_lat"])
            & (df["latitude"] <= bbox["max_lat"])
        ]
        print(f"  After bbox filter: {len(df)}")

    # 3. Filter by year
    if min_year:
        df = df[(df["year"].isna()) | (df["year"] >= min_year)]
        print(f"  After year filter (≥{min_year}): {len(df)}")

    # 4. Filter by coordinate uncertainty
    if max_coord_uncertainty_m:
        df = df[
            (df["coordinateUncertaintyInMeters"].isna())
            | (df["coordinateUncertaintyInMeters"] <= max_coord_uncertainty_m)
        ]
        print(f"  After uncertainty filter (≤{max_coord_uncertainty_m}m): {len(df)}")

    # 5. Spatial thinning (remove duplicates within grid cells)
    if remove_duplicates_resolution_km > 0:
        # Approximate: 1 degree ≈ 111 km
        resolution_deg = remove_duplicates_resolution_km / 111.0
        df = df.copy()
        df["grid_lon"] = np.round(df["longitude"] / resolution_deg) * resolution_deg
        df["grid_lat"] = np.round(df["latitude"] / resolution_deg) * resolution_deg
        df = df.drop_duplicates(subset=["grid_lon", "grid_lat"])
        df = df.drop(columns=["grid_lon", "grid_lat"])
        print(f"  After spatial thinning ({remove_duplicates_resolution_km}km): {len(df)}")

    print(f"  OK Final clean records: {len(df)} (removed {n_start - len(df)} of {n_start})")

    return df.reset_index(drop=True)
