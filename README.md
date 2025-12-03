# Understanding Food Safety Through NYC Restaurant Inspections

Short description
We analyze thousands of NYC restaurant inspection records to evaluate hygiene conditions citywide, identify safety violations, and assess how official grades reflect real sanitation risks across neighborhoods.

Overview
This repository provides a reproducible pipeline to:
- Download the DOHMH "Restaurant Inspection Results" dataset (NYC Open Data).
- Clean and standardize inspection records.
- Build establishment-level longitudinal histories and neighborhood aggregates.
- Compute hygiene indices and critical-violation metrics.
- Fit simple fixed-effects models comparing restaurants to themselves over time.
- Produce maps and time-series visualizations.

Contents
- data/download_data.py  — download CSV exports from NYC Open Data
- data/cleaning.py       — cleaning, standardization, spatial join to ZIPs
- src/features.py        — compute establishment histories and indices
- src/aggregate.py       — neighborhood-level aggregation
- src/modeling.py        — panel/fixed-effects modeling utilities
- src/visualize.py       — mapping and plotting helpers
- scripts/run_pipeline.py— orchestrates pipeline end-to-end
- requirements.txt       — Python package dependencies

Quickstart
1. Create a new Python environment and install dependencies:
   pip install -r requirements.txt

2. Download raw data (saves to data/raw/restaurant_inspections.csv):
   python data/download_data.py --out data/raw/restaurant_inspections.csv

   Note: The script uses the public CSV endpoint for the DOHMH dataset (resource id `43nn-pn8j`).
   The dataset is large; the script downloads in pages and saves a CSV.

3. Clean and enrich (requires optional ZIP shapefile for spatial joins):
   python data/cleaning.py --in data/raw/restaurant_inspections.csv --out data/processed/cleaned.parquet --zip-shapefile path/to/zip_shapefile.shp

   If you don't have a ZIP shapefile, the script will attempt to use the provided postal_code/zipcode fields.

4. Build features, aggregate, model, and visualize:
   python scripts/run_pipeline.py --input data/processed/cleaned.parquet --output results/

Outputs
- results/establishment_history.parquet
- results/neighborhood_aggregates.parquet
- results/models/*.pkl
- results/maps/*.html
- results/plots/*.png

Further work
- Incorporate 311 complaint data and local socioeconomic covariates.
- Improve models (dynamic treatment of inspection frequency, instrumental variables).
- Build a web dashboard for interactive exploration.

License
MIT
