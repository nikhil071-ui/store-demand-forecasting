"""ETL: load raw Kaggle CSV, basic sanity checks, save processed parquet."""
from pathlib import Path
import pandas as pd
from utils import load_config, read_data, ensure_dir

def main():
    cfg = load_config()
    raw_fp = cfg["paths"]["raw_data"]
    out_fp = cfg["paths"]["processed_data"]

    df = read_data(raw_fp)
    # basic checks
    assert set(["date","store","item","sales"]).issubset(df.columns), "Unexpected columns in raw data."

    # enforce dtypes
    df["store"] = df["store"].astype(int)
    df["item"]  = df["item"].astype(int)
    df["sales"] = df["sales"].astype(float)

    # sort and save
    df = df.sort_values(["store","item","date"]).reset_index(drop=True)

    ensure_dir(out_fp)
    df.to_parquet(out_fp, index=False)
    print(f"Wrote processed data -> {out_fp} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
