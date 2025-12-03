"""
Compute establishment-level histories and hygiene indices.

Functions:
- build_establishment_history: groups inspections by CAMIS, sorts by date, computes lagged features.
- compute_hygiene_index: a simple weighted index of critical violations, score, and closures.
"""
import pandas as pd
import numpy as np

def build_establishment_history(df, camis_col="camis", date_col="inspection_date"):
    df = df.copy()
    # Keep only necessary columns for history
    required_cols = [camis_col, date_col, "score", "grade", "violation_code", "violation_description", "is_critical_description", "action"]
    existing = [c for c in required_cols if c in df.columns]
    df_sub = df[existing + [c for c in df.columns if c not in existing]].copy()
    df_sub = df_sub.sort_values([camis_col, date_col])
    # compute inspection counts and days since prior inspection
    df_sub["inspection_number"] = df_sub.groupby(camis_col).cumcount() + 1
    df_sub["prev_inspection_date"] = df_sub.groupby(camis_col)[date_col].shift(1)
    df_sub["days_since_prev"] = (pd.to_datetime(df_sub[date_col]) - pd.to_datetime(df_sub["prev_inspection_date"])).dt.days
    # aggregate violations per inspection (some datasets have one row per violation)
    if "violation_code" in df_sub.columns:
        agg = df_sub.groupby([camis_col, date_col]).agg(
            violation_count=("violation_code", "count"),
            critical_violations=("is_critical_description", "sum"),
            score=("score", "first"),
            grade=("grade", "first"),
            action=("action", "first"),
        ).reset_index()
    else:
        agg = df_sub.groupby([camis_col, date_col]).agg(
            violation_count=("violation_description", lambda s: s.notna().sum()),
            critical_violations=("is_critical_description", "sum"),
            score=("score", "first"),
            grade=("grade", "first"),
            action=("action", "first"),
        ).reset_index()
    # merge back counts and timings
    agg = agg.sort_values([camis_col, date_col])
    agg["inspection_number"] = agg.groupby(camis_col).cumcount() + 1
    agg["prev_inspection_date"] = agg.groupby(camis_col)[date_col].shift(1)
    agg["days_since_prev"] = (pd.to_datetime(agg[date_col]) - pd.to_datetime(agg["prev_inspection_date"])).dt.days
    return agg

def compute_hygiene_index(df, weight_score=0.4, weight_critical=0.5, weight_violation_count=0.1):
    """
    Compute a simple hygiene index where higher values indicate worse performance.
    - score: numeric inspection score (higher = worse)
    - critical_violations: count of critical violations during inspection
    - violation_count: total violations
    Index scaled to 0-100.
    """
    df = df.copy()
    # normalize components
    # For score, since score could be NaN, fill with median
    df["score_filled"] = df["score"].fillna(df["score"].median())
    # Clip to reasonable ranges
    s = df["score_filled"]
    s_scaled = (s - s.min()) / (s.max() - s.min() + 1e-9)
    cv = df["critical_violations"].fillna(0)
    cv_scaled = (cv - cv.min()) / (cv.max() - cv.min() + 1e-9)
    vc = df["violation_count"].fillna(0)
    vc_scaled = (vc - vc.min()) / (vc.max() - vc.min() + 1e-9)
    df["hygiene_index_raw"] = weight_score * s_scaled + weight_critical * cv_scaled + weight_violation_count * vc_scaled
    df["hygiene_index"] = 100 * (df["hygiene_index_raw"] - df["hygiene_index_raw"].min()) / (df["hygiene_index_raw"].max() - df["hygiene_index_raw"].min() + 1e-9)
    return df

if __name__ == "__main__":
    # simple usage example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.infile)
    hist = build_establishment_history(df)
    hist = compute_hygiene_index(hist)
    hist.to_parquet(args.out, index=False)
    print("Wrote", args.out)
