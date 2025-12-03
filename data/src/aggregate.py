"""
Neighborhood-level aggregation utilities.

Produces time-series summaries by zipcode or neighborhood name:
- mean hygiene index
- share of inspections with closures or critical violations
- counts of inspections and unique establishments
"""
import pandas as pd
import numpy as np

def aggregate_by_zip(df, date_col="inspection_date", zip_col="zipcode", time_freq="M"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, zip_col])
    df["period"] = df[date_col].dt.to_period(time_freq).dt.to_timestamp()
    agg = df.groupby([zip_col, "period"]).agg(
        mean_hygiene_index=("hygiene_index", "mean"),
        median_hygiene_index=("hygiene_index", "median"),
        inspections=("inspection_number", "count"),
        unique_establishments=("camis", pd.Series.nunique),
        mean_score=("score", "mean"),
        mean_critical_violations=("critical_violations", "mean"),
        closure_share=("action", lambda s: (s.str.contains("closed", case=False, na=False)).mean())
    ).reset_index()
    return agg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.infile)
    out = aggregate_by_zip(df)
    out.to_parquet(args.outfile, index=False)
    print("Wrote", args.outfile)
