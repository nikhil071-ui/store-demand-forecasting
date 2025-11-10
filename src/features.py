"""Feature engineering: calendar + lags + rolling stats; write features parquet."""
import pandas as pd
from utils import load_config, top_n_series, add_calendar_features, make_lags, make_rolls
from pathlib import Path

def main():
    cfg = load_config()
    proc_fp = cfg["paths"]["processed_data"]
    feat_fp = cfg["paths"]["features_data"]
    n = int(cfg.get("top_n_series", 0))

    df = pd.read_parquet(proc_fp)
    if n and n > 0:
        df = top_n_series(df, n)

    df = add_calendar_features(df)
    df = make_lags(df)
    df = make_rolls(df)

    # drop rows with NaNs introduced by lags/rolls
    df = df.dropna().reset_index(drop=True)

    Path(feat_fp).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_fp, index=False)
    print(f"Wrote features -> {feat_fp} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
