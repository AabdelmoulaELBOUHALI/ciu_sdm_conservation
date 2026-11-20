"""
WorldClim Raster Extraction Utilities
======================================
Download WorldClim 2.1 bioclimatic variables and extract values at point locations.
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARN rasterio not available; real WorldClim extraction will fail until rasterio is installed")


WORLDCLIM_BASE_URL = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base"


def download_worldclim(data_dir: str, resolution: str = "2.5m") -> str:
    """
    Download WorldClim 2.1 bioclimatic variables.

    Parameters
    ----------
    data_dir : str
        Directory to save downloaded files
    resolution : str
        Resolution: "10m", "5m", "2.5m", or "30s"

    Returns
    -------
    str
        Path to the directory containing the .tif files
    """
    wc_dir = os.path.join(data_dir, f"worldclim_{resolution}")
    os.makedirs(wc_dir, exist_ok=True)

    # Check if already downloaded
    tif_files = [f for f in os.listdir(wc_dir) if f.endswith(".tif")]
    if len(tif_files) >= 19:
        print(f"  OK WorldClim data already downloaded ({len(tif_files)} files)")
        return wc_dir

    # Download the bioclim zip file
    filename = f"wc2.1_{resolution}_bio.zip"
    url = f"{WORLDCLIM_BASE_URL}/{filename}"
    zip_path = os.path.join(wc_dir, filename)

    if not os.path.exists(zip_path):
        print(f"  Downloading WorldClim {resolution} bioclimatic variables...")
        print(f"  URL: {url}")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) < 8192:  # Every ~1MB
                            print(f"    {pct:.1f}% ({downloaded / 1e6:.1f} MB)")

            print(f"  OK Downloaded: {zip_path}")
        except Exception as e:
            print(f"  FAIL Download failed: {e}")
            print(f"  -> Please download manually from: {url}")
            print(f"  -> Place the .tif files in: {wc_dir}")
            return wc_dir

    # Extract
    if os.path.exists(zip_path):
        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(wc_dir)
        print(f"  OK Extracted to {wc_dir}")

    return wc_dir


def extract_bioclim_values(
    points_df: pd.DataFrame,
    worldclim_dir: str,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """
    Extract bioclimatic variable values at point locations from WorldClim rasters.

    Parameters
    ----------
    points_df : pd.DataFrame
        DataFrame with longitude and latitude columns
    worldclim_dir : str
        Path to directory containing WorldClim .tif files
    lon_col, lat_col : str
        Column names for coordinates

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added bio1-bio19 columns
    """
    print(f"\nExtracting bioclimatic values for {len(points_df)} points...")

    result_df = points_df.copy()

    # Find all bioclim tif files
    tif_files = sorted([
        f for f in os.listdir(worldclim_dir)
        if f.endswith(".tif") and "bio" in f.lower()
    ])

    if not tif_files:
        raise FileNotFoundError(
            f"No WorldClim .tif files found in {worldclim_dir}. "
            "Real-data mode requires WorldClim rasters."
        )

    if not HAS_RASTERIO:
        raise RuntimeError(
            "rasterio is not installed. Real WorldClim extraction requires rasterio."
        )

    lons = result_df[lon_col].values
    lats = result_df[lat_col].values

    for tif_file in tif_files:
        # Extract variable name (e.g., "bio1" from "wc2.1_2.5m_bio_1.tif")
        parts = tif_file.replace(".tif", "").split("_")
        bio_num = parts[-1]
        var_name = f"bio{bio_num}"

        tif_path = os.path.join(worldclim_dir, tif_file)

        values = _extract_with_rasterio(tif_path, lons, lats)

        result_df[var_name] = values
        n_valid = np.sum(~np.isnan(values))
        print(f"  {var_name}: {n_valid}/{len(values)} valid values extracted")

    return result_df


def _extract_with_rasterio(tif_path, lons, lats):
    """Extract raster values at point locations using rasterio."""
    values = np.full(len(lons), np.nan)

    with rasterio.open(tif_path) as src:
        # Batch sampling is much faster than reading raster cell-by-cell.
        coords = list(zip(lons, lats))
        for i, sample in enumerate(src.sample(coords)):
            try:
                val = sample[0]
                if src.nodata is None or val != src.nodata:
                    values[i] = val
            except Exception:
                continue

    return values


def extract_bioclim_for_grid(
    worldclim_dir: str,
    bbox: dict,
    resolution_deg: float = 0.0417,  # ~2.5 arc-min ≈ 0.0417°
) -> pd.DataFrame:
    """
    Create a grid of points covering the study area and extract bioclim values.
    Useful for generating pseudo-absence background points.

    Parameters
    ----------
    worldclim_dir : str
        Path to WorldClim .tif files
    bbox : dict
        Study area bounding box
    resolution_deg : float
        Grid resolution in degrees

    Returns
    -------
    pd.DataFrame
        Grid points with bioclim values (excluding ocean/NoData cells)
    """
    lons = np.arange(bbox["min_lon"], bbox["max_lon"], resolution_deg)
    lats = np.arange(bbox["min_lat"], bbox["max_lat"], resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    grid_df = pd.DataFrame({
        "longitude": lon_grid.ravel(),
        "latitude": lat_grid.ravel(),
    })

    print(f"\nCreating background grid: {len(grid_df)} cells")
    grid_df = extract_bioclim_values(grid_df, worldclim_dir)

    # Remove cells with NoData (ocean, etc.)
    bioclim_cols = [c for c in grid_df.columns if c.startswith("bio")]
    grid_df = grid_df.dropna(subset=bioclim_cols).reset_index(drop=True)
    print(f"  OK Valid land cells: {len(grid_df)}")

    return grid_df
