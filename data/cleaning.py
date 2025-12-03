#!/usr/bin/env python3
"""
Load raw inspection CSV, standardize columns, parse dates, detect critical violations,
and optionally spatially join to a ZIP-code shapefile (GeoDataFrame).

Produces a parquet file with cleaned records.

Notes:
- The DOHMH dataset contains many textual variations in violation codes. We normalize common fields.
- The script is conservative and keeps original columns with prefix RAW_ for traceability.
"""
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

CRITICAL_KEYWORDS = [
    "rodent", "roaches", "sewage", "no hot water",
    "no hot", "improper holding", "critical", "major",
    "risk", "imminent", "closure"
]

def load_csv(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    return df

def standardize(df):
    # copy raw columns
    for c in df.columns:
        df.rename(columns={c: c.strip() if isinstance(c, str) else c}, inplace=True)
    # keep raw copy of important fields
    for col in ["violation_description", "grade", "score", "inspection_date"]:
        if col in df.columns:
            df["RAW_" + col] = df[col]
    # parse dates
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    # numeric conversions
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    # standardize grades (A/B/C/Not Yet Graded/Not Applicable)
    if "grade" in df.columns:
        df["grade"] = df["grade"].str.upper().str.strip().replace({"N/A":"NOT APPLICABLE"})
    # detect critical violations by keywords in violation_description
    df["violation_description"] = df.get("violation_description", pd.Series([""]*len(df), index=df.index)).astype(str)
    df["is_critical_description"] = df["violation_description"].str.lower().apply(
        lambda s: any(k in s for k in CRITICAL_KEYWORDS)
    )
    # canonicalize restaurant id
    if "camis" in df.columns:
        df["camis"] = df["camis"].astype(str)
    else:
        # fallback: combine building + phone (not ideal)
        df["camis"] = df.get("camis", pd.Series([None]*len(df), index=df.index)).fillna("").astype(str)
    return df

def spatially_join_zip(df, zip_shapefile=None):
    """
    If a ZIP code shapefile is provided, perform spatial join using latitude/longitude.
    Otherwise, prefer existing zipcode/postal_code fields.
    Expect shapefile to be a GeoDataFrame with a column named "ZIPCODE" or "ZIP".
    """
    # Try to use existing columns
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str).str.slice(0,5).str.zfill(5)
    elif "postal_code" in df.columns:
        df["zipcode"] = df["postal_code"].astype(str).str.slice(0,5).str.zfill(5)
    # If shapefile provided, do spatial join for better accuracy
    if zip_shapefile and os.path.exists(zip_shapefile):
        print("Loading ZIP shapefile:", zip_shapefile)
        zips = gpd.read_file(zip_shapefile)
        # ensure geometry is present
        if "geometry" not in zips:
            raise RuntimeError("ZIP shapefile has no geometry")
        # Detect lat/lon in df
        if "latitude" in df.columns and "longitude" in df.columns:
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs=zips.crs)
            joined = gpd.sjoin(gdf, zips, how="left", predicate="within")
            # try common names for ZIP field
            zip_field = None
            for candidate in ["ZIPCODE", "ZIP", "zip", "zipcode"]:
                if candidate in joined.columns:
                    zip_field = candidate
                    break
            if zip_field:
                joined["zipcode"] = joined[zip_field].astype(str).str.slice(0,5).str.zfill(5)
            joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
            return pd.DataFrame(joined)
        else:
            print("No lat/lon found; falling back to existing zipcode field")
    return df

def save_parquet(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print("Wrote cleaned data to", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="Raw CSV path")
    parser.add_argument("--out", dest="outfile", required=True, help="Cleaned parquet path")
    parser.add_argument("--zip-shapefile", dest="zip_shapefile", default=None, help="Optional ZIP shapefile for spatial join")
    args = parser.parse_args()

    df = load_csv(args.infile)
    print("Loaded", len(df), "rows")
    df = standardize(df)
    df = spatially_join_zip(df, zip_shapefile=args.zip_shapefile)
    save_parquet(df, args.outfile)

if __name__ == "__main__":
    main()
