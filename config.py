"""
Configuration of the CIU-SDM pipeline: paths, species, data preparation,
models, explanation methods and evaluation settings.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# SPECIES CONFIGURATION
# ============================================================
# Ten Moroccan bird species: four wheatears (Oenanthe), three redstarts
# (Phoenicurus) and three species from other families.
SPECIES = {
    "Oenanthe_deserti": {
        "gbif_name": "Oenanthe deserti",
        "common_name": "Desert Wheatear",
        "description": "Arid-land wheatear of open desert and steppe habitats in Morocco",
        "region": "Morocco",
    },
    "Oenanthe_leucopyga": {
        "gbif_name": "Oenanthe leucopyga",
        "common_name": "White-crowned Wheatear",
        "description": "Rocky desert wheatear associated with warm, dry habitats",
        "region": "Morocco",
    },
    "Oenanthe_leucura": {
        "gbif_name": "Oenanthe leucura",
        "common_name": "Black Wheatear",
        "description": "Cliff- and rock-dwelling wheatear in dry Mediterranean landscapes",
        "region": "Morocco",
    },
    "Oenanthe_oenanthe": {
        "gbif_name": "Oenanthe oenanthe",
        "common_name": "Northern Wheatear",
        "description": "Migratory wheatear occupying open uplands and semi-arid grasslands",
        "region": "Morocco",
    },
    "Phoenicurus_moussieri": {
        "gbif_name": "Phoenicurus moussieri",
        "common_name": "Moussier's Redstart",
        "description": "Near-endemic North African redstart strongly associated with Morocco",
        "region": "Morocco",
    },
    "Phoenicurus_ochruros": {
        "gbif_name": "Phoenicurus ochruros",
        "common_name": "Black Redstart",
        "description": "Broadly distributed redstart found in montane and human-modified habitats",
        "region": "Morocco",
    },
    "Phoenicurus_phoenicurus": {
        "gbif_name": "Phoenicurus phoenicurus",
        "common_name": "Common Redstart",
        "description": "Migratory redstart using woodland-edge and open forest habitats",
        "region": "Morocco",
    },
    "Alectoris_barbara": {
        "gbif_name": "Alectoris barbara",
        "common_name": "Barbary Partridge",
        "description": "Resident ground bird of scrub, rocky slopes, and semi-arid Mediterranean habitats",
        "region": "Morocco",
    },
    "Bubulcus_ibis": {
        "gbif_name": "Bubulcus ibis",
        "common_name": "Cattle Egret",
        "description": "Widespread wetland and farmland species often associated with irrigated lowlands",
        "region": "Morocco",
    },
    "Upupa_epops": {
        "gbif_name": "Upupa epops",
        "common_name": "Eurasian Hoopoe",
        "description": "Open-country bird of orchards, steppe, and lightly wooded agricultural land",
        "region": "Morocco",
    },
}

# ============================================================
# STUDY AREA (Morocco bounding box)
# ============================================================
STUDY_AREA = {
    "min_lon": -13.5,
    "max_lon": -1.0,
    "min_lat": 27.5,
    "max_lat": 36.0,
}

# ============================================================
# BIOCLIMATIC VARIABLES (WorldClim 2.1)
# ============================================================
# 19 standard bioclimatic variables at 2.5 arc-minute resolution
WORLDCLIM_RESOLUTION = "2.5m"  # Options: "10m", "5m", "2.5m", "30s"
BIOCLIM_VARS = [f"bio{i}" for i in range(1, 20)]  # bio1 through bio19

MAX_VIF_THRESHOLD = 10  # stepwise removal of the predictor with the largest VIF while VIF > 10

# ============================================================
# DATA PREPROCESSING
# ============================================================
PSEUDO_ABSENCE = {
    # Pseudo-absences are sampled between a minimum and a maximum distance
    # from the presences of the focal species.
    "method": "distance_constrained",  # Options: "random", "distance_constrained", "target_group"
    "n_pseudo_absences_ratio": 1.0,    # Ratio of pseudo-absences to presences (1:1)
    "min_distance_km": 10,              # Minimum distance from presence points
    "max_distance_km": 200,             # Maximum distance (stay within plausible range)
    "random_seed": 42,
}

PREPROCESSING = {
    # Grid thinning (one record per ~2 km cell) and IQR outlier filter of the
    # presence records on the 19 bioclimatic variables.
    "spatial_thinning_km": 2.0,
    "apply_iqr_filter": True,
    "iqr_multiplier": 1.5,
}

DATA_COLLECTION = {
    # A species without usable GBIF records is reported and skipped.
    "fail_on_species_error": False,
    "min_clean_records": 30,
    "gbif_max_records": 5000,
}

# Spatial block cross-validation
SPATIAL_CV = {
    "n_folds": 5,
    "block_size_km": 50,  # Size of spatial blocks
    "random_seed": 42,
}

# ============================================================
# MODEL CONFIGURATION
# ============================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2  # held-out share used to measure surrogate-tree fidelity

MODELS = {
    # Base learners
    "RandomForest": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "XGBoost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    },
    "MLP": {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 1000,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": RANDOM_SEED,
    },
    "LightGBM": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,  # Suppress LightGBM training output
    },
    "GAM": {
        # pygam LogisticGAM parameters (wrapped via utils/gam_wrapper.py)
        # Higher lam = stronger regularisation = more stable with many features
        "n_splines": 10,
        "lam": 10.0,
    },
    "SVM": {
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale",
        "probability": True,   # Required for predict_proba support
        "random_state": RANDOM_SEED,
    },
    # Ensembles
    "VotingEnsemble": {
        "estimators": ["RandomForest", "XGBoost", "LightGBM", "SVM", "MLP"],
        "weights": [2, 2, 2, 1, 1],
        "voting": "soft",
        "n_jobs": -1,
    },
    "StackingEnsemble": {
        "estimators": ["RandomForest", "XGBoost", "LightGBM", "SVM"],
        "final_estimator": "MLP",
        "stack_method": "predict_proba",
        "passthrough": False,
        "cv": 5,
        "n_jobs": -1,
    },
    "BaggingEnsemble": {
        "n_estimators": 25,
        "max_samples": 0.8,
        "max_features": 1.0,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
}

# ============================================================
# XAI CONFIGURATION
# ============================================================
XAI = {
    "shap": {
        "n_background_samples": 100,  # Background dataset size for KernelSHAP
        "n_explain": 50,             # Number of instances to explain
    },
    "lime": {
        "n_features": 10,             # Max features in LIME explanation
        "n_samples": 1000,            # Perturbation samples
        "n_explain": 50,
    },
    "ciu": {
        "nsamples": 200,             # Monte Carlo samples for CIU
        "n_explain": 50,
    },
    "ale": {
        "n_bins": 20,                 # Number of bins for ALE plots
    },
}

# ============================================================
# EVALUATION METRICS
# ============================================================
METRICS = {
    "n_faithfulness_steps": 10,  # Steps for feature removal in faithfulness test
}

# ============================================================
# FIGURE SETTINGS
# ============================================================
FIGURE = {
    "dpi": 300,
    "format": "pdf",  # For publication
    "style": "seaborn-v0_8-whitegrid",
    "figsize_single": (6, 4),
    "figsize_double": (12, 5),
    "figsize_full": (14, 10),
    "color_palette": {
        "CIU": "#E63946",
        "SHAP": "#457B9D",
        "LIME": "#2A9D8F",
        "ALE": "#E9C46A",
    },
}

# ============================================================
# ROBUSTNESS ANALYSIS (06_robustness_analysis.py)
# Reads results/ and writes results/robustness/.
# - repeated spatial block CV with the scaler fitted inside each training fold
# - CIU and the faithfulness/monotonicity evaluation on every record
# - sensitivity of the 50-instance explanations to the sampling seed
# ============================================================
ROBUSTNESS = {
    "cv_seeds": [42, 1, 2, 3, 4, 5],          # block-to-fold assignment and model seeds
    "sampling_seeds": [1, 2, 3, 4, 5],        # 50-instance CIU sample / evaluation sample
    "full_data_mc_seeds": [42, 1],            # Monte Carlo draws of CIU on every record
    "n_random_rankings": 50,                  # random-ranking baseline for faithfulness
    "ci_cut": 0.3,                            # role thresholds (as in 05_evaluate_metrics)
    "cu_cut": 0.5,
    "ci_cut_grid": [0.25, 0.30, 0.35],
    "cu_cut_grid": [0.4, 0.5, 0.6],
}
