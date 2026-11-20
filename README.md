# Contextual Importance and Utility for species distribution models

Code, data and results for the article *Contextual Importance and Utility:
Separating Climatic Drivers from Limiting Factors in Species Distribution Models
for Bird Conservation* (Journal for Nature Conservation, submitted).

Ten Moroccan bird species are modelled with nine model families under spatial
block cross-validation. The fitted models are explained with CIU, SHAP, LIME and
ALE, the explanations are compared on faithfulness, monotonicity and run time,
and the two CIU outputs (contextual importance and contextual utility) are used
to label each predictor as an optimal driver, a limiting factor or a
low-influence variable.

## Environment

Python 3.13.2 with the exact package versions in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

All settings (species, filters, model hyperparameters, explanation settings,
seeds) are in `config.py`.

## Pipeline

Run the scripts in order from this folder.

| Step | Script | Main outputs |
|---|---|---|
| 1 | `01_data_collection.py` | GBIF occurrences (`data/*_raw_occurrences.csv`, `data/*_clean_occurrences.csv`), WorldClim 2.1 layers, background grid |
| 2 | `02_data_preprocessing.py` | IQR filter, pseudo-absences, VIF screening, standardisation, spatial folds (`data/*_dataset.npz`, `data/*_spatial_folds.npz`) |
| 3 | `03_train_sdms.py` | 5-fold spatial block CV and final models (`models/*.pkl`, `results/model_performance.csv`, Scott-Knott and Borda rankings) |
| 4 | `04_apply_xai.py` | CIU, SHAP, LIME and ALE explanations, surrogate and white-box trees (`results/*_xai_*.csv`, `results/*_ciu_details.csv`, `results/global_surrogate_trees.csv`, `results/whitebox_trees.csv`) |
| 5 | `05_evaluate_metrics.py` | Faithfulness, monotonicity, run time, Kendall's tau, CIU roles on the 50-instance explanations (`results/xai_*.csv`, `results/ciu_ecological_analysis.csv`) |
| 6 | `06_robustness_analysis.py` | Repeated spatial CV, explanation quality and CIU on every record (with random-ranking baseline), sampling sensitivity, reproducibility check (`results/robustness/`) |
| 7 | `07_figures.py` | Figures 3-8 of the article as vector PDF and EPS (`figures/Figure_3` ... `figures/Figure_8`); `--png` adds 1000-dpi PNG copies |
| 8 | `08_export_results.py` | Summary tables in one workbook (`results/exports/results_summary.xlsx`) |

Step 6 is run part by part:

```
python 06_robustness_analysis.py --part reproduce      --workers 8
python 06_robustness_analysis.py --part repeated-cv    --workers 8
python 06_robustness_analysis.py --part sampling       --workers 8
python 06_robustness_analysis.py --part full-data-eval --workers 8
python 06_robustness_analysis.py --part single-feature-eval --workers 8
python 06_robustness_analysis.py --part full-data-ciu  --workers 8
python 06_robustness_analysis.py --part summarise
```

Approximate run times on a 22-core workstation (32 GB RAM): steps 1-5 about
14 h in total (mostly data download and raster extraction), step 6 about 5 h
(`full-data-ciu` about 2.5 h).

## What the article reports

- Predictive performance: `results/robustness/repeated_cv_*.csv` (spatial block
  CV repeated with six seeds; the scaler is fitted inside each training fold).
- Explanation quality: `results/robustness/full_data_xai_quality.csv` and
  `full_data_xai_evaluation.csv`, computed on every record from each method's
  global ranking: faithfulness as the correlation between importance and the
  change in prediction when one predictor is set to its mean (Alvarez-Melis &
  Jaakkola, 2018), monotonicity on the cumulative deletion curve, and the
  deletion-curve correlation used as faithfulness by step 5, each with 50
  random rankings per combination as a baseline. Kendall's tau:
  `results/xai_ranking_agreement.csv`; run times:
  `results/xai_comparison_metrics.csv`.
- Ecological roles: `results/robustness/ciu_ecological_roles_all_records.csv`
  and `ciu_threshold_sensitivity_all_records.csv` (CIU computed on every record).
- Robustness: `results/robustness/summary.json` (reproduction with the original
  seed, agreement between seeds, random-ranking baseline).

The outputs of steps 3-5 are kept unchanged in `results/`; step 6 only reads them.

## Data

- Occurrences: GBIF occurrence search service, Morocco bounding box
  (27.5-36.0 N, 13.5-1.0 W), at most 5,000 records per species, retrieved on
  23 April 2026. Records were kept from 1970 onwards with a stated coordinate
  uncertainty of at most 10 km (or none stated), then thinned on a 2 km grid.
  Morocco south of 27.5 N is not included because few records of the ten
  species are available there.
- Climate: WorldClim 2.1 bioclimatic variables bio1-bio19 at 2.5 arc-minutes
  (temperatures in degrees Celsius, precipitation in mm).
- Two further candidate species did not meet the selection criterion of at
  least 200 cleaned presences and were not analysed (*Cyanopica cooki*: no
  usable records; *Psittacula krameri*: 42 cleaned records).

## Layout

```
config.py                settings
01_ ... 08_*.py          pipeline steps
utils/                   CIU implementation, GAM wrapper, GBIF and raster helpers,
                         pseudo-absences, spatial CV, metrics, Scott-Knott/Borda,
                         surrogate trees
data/                    occurrences, climate layers, modelling datasets, folds
models/                  90 final models (10 species x 9 families)
results/                 outputs of steps 3-5; robustness/ (step 6); exports/ (step 8)
figures/                 figures of the article (step 7)
```

Figures 1 and 2 of the article (workflow and role grid) are drawn in LaTeX and
are part of the manuscript source, not of this code.

The WorldClim 2.1 rasters (`data/worldclim_2.5m/`, about 670 MB) are not
included; step 1 downloads them from the WorldClim server.

## Licence

- Code: MIT License (`LICENSE`).
- Models, results and figures: CC BY 4.0 (`LICENSE-DATA.md`).
- GBIF occurrence records in `data/`: the licence of each original publisher
  (CC0, CC BY or CC BY-NC); see `LICENSE-DATA.md`.

Citation details are in `CITATION.cff`.
