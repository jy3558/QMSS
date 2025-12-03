#!/usr/bin/env python3
"""
Orchestrate the pipeline:
1) Read cleaned parquet with inspection-level data
2) Build establishment-level histories and hygiene index
3) Aggregate by ZIP and save results
4) Run a simple panel fixed-effects model
5) Produce a sample folium map and time-series plot (if ZIP shapefile is available)

Usage:
python scripts/run_pipeline.py --input data/processed/cleaned.parquet --output results/ --zip-shapefile data/geo/zip_shapefile.shp
"""
import argparse
import os
import pandas as pd
from src.features import build_establishment_history, compute_hygiene_index
from src.aggregate import aggregate_by_zip
from src.modeling import fit_panel_fe
from src.visualize import folium_map_by_zip, plot_time_series
import geopandas as gpd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cleaned parquet with inspection-level records")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument("--zip-shapefile", default=None, help="Optional ZIP shapefile for maps")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading cleaned data...")
    df = pd.read_parquet(args.input)
    print("Building establishment histories...")
    est_hist = build_establishment_history(df)
    est_hist = compute_hygiene_index(est_hist)
    est_out = os.path.join(args.output, "establishment_history.parquet")
    est_hist.to_parquet(est_out, index=False)
    print("Wrote", est_out)

    print("Aggregating by ZIP...")
    agg = aggregate_by_zip(est_hist)
    agg_out = os.path.join(args.output, "neighborhood_aggregates.parquet")
    agg.to_parquet(agg_out, index=False)
    print("Wrote", agg_out)

    print("Running a sample panel fixed-effects regression...")
    try:
        res = fit_panel_fe(est_hist, depvar="hygiene_index", exog=["inspection_number", "critical_violations"], entity_id="camis", time_id="inspection_date")
        model_out = os.path.join(args.output, "models")
        os.makedirs(model_out, exist_ok=True)
        # save model summary to text
        with open(os.path.join(model_out, "sample_model_summary.txt"), "w") as fh:
            try:
                fh.write(str(res.summary))
            except Exception:
                fh.write(res.summary.as_text())
        print("Saved model summary to", model_out)
    except Exception as e:
        print("Modeling failed:", e)

    # Visualizations
    if args.zip_shapefile and os.path.exists(args.zip_shapefile):
        print("Generating choropleth map using ZIP shapefile...")
        zips = gpd.read_file(args.zip_shapefile)
        try:
            map_out = os.path.join(args.output, "maps", "zip_hygiene_map.html")
            folium_map_by_zip(zips, agg, out_html=map_out)
        except Exception as e:
            print("Map generation failed:", e)
    else:
        print("ZIP shapefile not provided or not found; skipping map generation.")

    # sample time series for a sample zipcode
    sample_zip = agg["zipcode"].dropna().unique()
    if len(sample_zip) > 0:
        sample_zip = sample_zip[0]
        print("Creating time series plot for sample zipcode:", sample_zip)
        try:
            fig = plot_time_series(agg, sample_zip)
            # save as html
            ts_out = os.path.join(args.output, "plots", f"time_series_{sample_zip}.html")
            os.makedirs(os.path.dirname(ts_out), exist_ok=True)
            fig.write_html(ts_out)
            print("Wrote time-series interactive plot to", ts_out)
        except Exception as e:
            print("Time-series plotting failed:", e)

if __name__ == "__main__":
    main()
