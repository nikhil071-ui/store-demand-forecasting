"""Train an XGBoost global model on lag/rolling features for all (store,item)."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from utils import load_config

FEATURES = [
    "store","item","year","month","day","dayofweek","is_weekend","weekofyear",
    "lag_1","lag_7","lag_14","lag_28",
    "roll_mean_7","roll_std_7","roll_mean_14","roll_std_14","roll_mean_28","roll_std_28",
]

def main():
    cfg = load_config()
    feat_fp = cfg["paths"]["features_data"]
    model_fp = cfg["paths"]["model_file"]
    Path(model_fp).parent.mkdir(parents=True, exist_ok=True)

    # Load feature data
    df = pd.read_parquet(feat_fp).sort_values("date")

    X = df[FEATURES]
    y = df["sales"]

    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    rmses = []

    model = XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=cfg.get("random_state", 42),
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=-1,
    )

    # Cross-validation loop
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])

        mae = mean_absolute_error(y.iloc[va], pred)
        mse = mean_squared_error(y.iloc[va], pred)
        rmse = mse ** 0.5  # manual RMSE fix (works on all sklearn versions)

        maes.append(mae)
        rmses.append(rmse)

        print(f"Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    print(f"\nCV MAE:  {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"CV RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}\n")

    # Final training on full dataset
    model.fit(X, y)
    joblib.dump(model, model_fp)
    print(f"✅ Saved model -> {model_fp}")

if __name__ == "__main__":
    main()
