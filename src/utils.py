import os
import yaml
import pandas as pd
from pathlib import Path

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def read_data(fp: str) -> pd.DataFrame:
    if not Path(fp).exists():
        raise FileNotFoundError(f"File not found: {fp}. Please place Kaggle CSV there.")
    df = pd.read_csv(fp, parse_dates=["date"])
    return df

def top_n_series(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n and n > 0:
        totals = df.groupby(["store", "item"], as_index=False)["sales"].sum()
        keep = set(tuple(x) for x in totals.sort_values("sales", ascending=False).head(n)[["store","item"]].to_records(index=False))
        return df[df.apply(lambda r: (r["store"], r["item"]) in keep, axis=1)]
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def make_lags(df: pd.DataFrame, group_cols=["store","item"], target="sales", lags=(1,7,14,28)) -> pd.DataFrame:
    df = df.sort_values(["store","item","date"]).copy()
    for L in lags:
        df[f"lag_{L}"] = df.groupby(group_cols)[target].shift(L)
    return df

def make_rolls(df: pd.DataFrame, group_cols=["store","item"], target="sales", windows=(7,14,28)) -> pd.DataFrame:
    df = df.sort_values(["store","item","date"]).copy()
    for W in windows:
        df[f"roll_mean_{W}"] = df.groupby(group_cols)[target].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"]  = df.groupby(group_cols)[target].shift(1).rolling(W).std()
    return df
