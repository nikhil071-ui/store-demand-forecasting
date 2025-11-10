"""Generate next-horizon forecasts and inventory KPIs per (store,item)."""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from utils import load_config, add_calendar_features, make_lags, make_rolls

def _future_calendar(df_hist: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    # create a future calendar for each (store,item)
    last_date = df_hist["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    keys = df_hist[["store","item"]].drop_duplicates().assign(key=1)
    fut = (
        pd.DataFrame({"date": future_dates})
        .assign(key=1)
        .merge(keys, on="key", how="left")
        .drop(columns="key")
    )
    fut = add_calendar_features(fut)
    return fut

def inventory_metrics(df: pd.DataFrame, z=1.65, lead_time_days=7):
    # z=1.65 ~ 95th percentile safety
    # compute per (store,item)
    out = []
    for (s,i), g in df.groupby(["store","item"]):
        avg_daily = g["forecast"].mean()
        std_daily = g["forecast"].std(ddof=0)
        safety_stock = z * (std_daily if not np.isnan(std_daily) else 0.0)
        reorder_level = avg_daily * lead_time_days + safety_stock
        out.append({"store": s, "item": i, "avg_daily": avg_daily, "std_daily": std_daily,
                    "safety_stock": safety_stock, "reorder_level": reorder_level})
    return pd.DataFrame(out)

def main():
    cfg = load_config()
    proc_fp = cfg["paths"]["processed_data"]
    feat_fp = cfg["paths"]["features_data"]
    model_fp = cfg["paths"]["model_file"]
    out_fp = cfg["paths"]["forecast_file"]
    horizon = int(cfg.get("forecast_horizon_days", 90))

    df_hist = pd.read_parquet(proc_fp).sort_values(["store","item","date"])
    model = joblib.load(model_fp)

    # to build features for future, we need to iteratively extend using lags/rolls
    hist_feat = df_hist.copy()
    hist_feat = add_calendar_features(hist_feat)
    hist_feat = make_lags(hist_feat)
    hist_feat = make_rolls(hist_feat)

    # start with last known window per (store,item)
    FEATURES = [
        "store","item","year","month","day","dayofweek","is_weekend","weekofyear",
        "lag_1","lag_7","lag_14","lag_28",
        "roll_mean_7","roll_std_7","roll_mean_14","roll_std_14","roll_mean_28","roll_std_28",
    ]

    futures = []
    # iterative forecasting to roll lags forward
    for (s,i), g in hist_feat.groupby(["store","item"], sort=False):
        g = g.sort_values("date").copy()
        hist = g[["date","sales"]].copy()

        # Seed last known values to compute lags/rolls forward
        last_known = g.iloc[-28:][["date","sales"]].copy()

        # Build future frame
        fut_dates = pd.date_range(hist["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        fut = pd.DataFrame({"date": fut_dates})
        fut["store"] = s; fut["item"] = i
        fut = add_calendar_features(fut)

        # We'll maintain a rolling list of last sales to compute lags/rolls
        window_sales = list(last_known["sales"].values)

        preds = []
        for d in fut["date"]:
            # construct feature row
            row = {
                "store": s, "item": i, "date": d,
                "year": d.year, "month": d.month, "day": d.day,
                "dayofweek": d.dayofweek, "is_weekend": int(d.dayofweek >= 5),
                "weekofyear": int(d.isocalendar().week),
            }
            # lags
            def lag(k):
                return window_sales[-k] if len(window_sales) >= k else np.nan
            row["lag_1"] = lag(1)
            row["lag_7"] = lag(7)
            row["lag_14"] = lag(14)
            row["lag_28"] = lag(28)
            # rolls
            def roll_mean(k):
                arr = window_sales[-k:] if len(window_sales) >= k else window_sales[:]
                return np.mean(arr) if len(arr) else np.nan
            def roll_std(k):
                arr = window_sales[-k:] if len(window_sales) >= k else window_sales[:]
                return np.std(arr, ddof=0) if len(arr) else np.nan
            row["roll_mean_7"] = roll_mean(7); row["roll_std_7"] = roll_std(7)
            row["roll_mean_14"] = roll_mean(14); row["roll_std_14"] = roll_std(14)
            row["roll_mean_28"] = roll_mean(28); row["roll_std_28"] = roll_std(28)

            X = pd.DataFrame([row])[FEATURES]
            yhat = float(model.predict(X)[0])
            preds.append(yhat)
            window_sales.append(yhat)

        fut["forecast"] = preds
        futures.append(fut[["date","store","item","forecast"]])

    fc = pd.concat(futures, ignore_index=True)
    # Inventory metrics over horizon
    inv = inventory_metrics(fc)
    out = fc.merge(inv, on=["store","item"], how="left")

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_fp, index=False)
    print(f"Wrote forecasts -> {out_fp} ({len(out):,} rows)")

if __name__ == "__main__":
    main()
