#!/usr/bin/env python3
"""
Download DOHMH Restaurant Inspection Results (NYC Open Data resource id 43nn-pn8j)
Saves a CSV to the provided output path. Downloads in pages to avoid memory spikes.
"""
import argparse
import requests
import csv
from tqdm import tqdm

CSV_ENDPOINT = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv"

def download_csv(out_path, limit=50000, max_rows=None):
    """
    Downloads the dataset in paged chunks using $limit and $offset.
    Set max_rows to stop early for testing.
    """
    offset = 0
    first = True
    rows_downloaded = 0
    with open(out_path, "w", newline='', encoding='utf-8') as fout:
        writer = None
        pbar = tqdm(total=max_rows or 0, unit="rows", desc="Downloading") if max_rows else None
        while True:
            params = {"$limit": limit, "$offset": offset}
            r = requests.get(CSV_ENDPOINT, params=params, timeout=60)
            r.raise_for_status()
            chunk = r.text
            if not chunk.strip():
                break
            lines = chunk.splitlines()
            reader = csv.DictReader(lines)
            if first:
                writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                writer.writeheader()
                first = False
            count = 0
            for row in reader:
                writer.writerow(row)
                count += 1
                rows_downloaded += 1
                if max_rows and rows_downloaded >= max_rows:
                    break
            if pbar:
                pbar.update(count)
            if count < limit or (max_rows and rows_downloaded >= max_rows):
                break
            offset += limit
    if pbar:
        pbar.close()
    print(f"Saved {rows_downloaded} rows to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=50000, help="Rows per request")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows (for testing)")
    args = parser.parse_args()
    download_csv(args.out, limit=args.limit, max_rows=args.max_rows)

if __name__ == "__main__":
    main()
