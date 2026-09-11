from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_pipeline import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET, TRAINING_DATA, prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"


def train_model() -> dict:
    _, frame = prepare_data()
    numeric = [column for column in NUMERIC_FEATURES if column in frame.columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame.columns]
    features = numeric + categorical
    X = frame[features]
    y = frame[TARGET].astype(float)

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=700,
            max_depth=12,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    groups = frame["Deposit_ID"].where(frame["Deposit_ID"].notna(), frame.index.to_series().astype(str)).astype(str)
    group_count = groups.nunique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    grouped_scores = cross_val_score(
        model,
        X,
        y,
        cv=GroupKFold(n_splits=min(5, group_count)),
        groups=groups,
        scoring="r2",
    ) if group_count >= 2 else np.array([])
    model.fit(X, y)
    metrics = {
        "model": "RandomForestRegressor",
        "target": TARGET,
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "training_rows": int(len(frame)),
        "test_rows": int(len(y_test)),
        "test_size": 0.2,
        "validation_method": "GroupShuffleSplit by Deposit_ID; synthetic siblings stay with their source group.",
        "cleaned_rows": int((frame["Synthetic"] == False).sum()),
        "original_training_rows": int((frame["Synthetic"] == False).sum()),
        "synthetic_training_rows": int((frame["Synthetic"] == True).sum()),
        "years": sorted(frame["Year"].unique().tolist()),
        "features": features,
        "feature_count": len(features),
        "model_version": "rf-v2-grouped-700",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_parameters": model.named_steps["regressor"].get_params(),
        "training_r2": float(r2_score(y_train, Pipeline(model.steps).fit(X_train, y_train).predict(X_train))),
        "cross_validation_r2_mean": float(grouped_scores.mean()) if len(grouped_scores) else None,
        "cross_validation_r2_std": float(grouped_scores.std()) if len(grouped_scores) else None,
        "grouped_cross_validation_r2_mean": float(grouped_scores.mean()) if len(grouped_scores) else None,
        "validation_note": "Validation is grouped by Deposit_ID to prevent synthetic siblings from crossing the evaluation boundary. Training R2 is shown only as an overfit diagnostic.",
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "features": features, "metrics": metrics, "reference": X.iloc[0].to_dict()}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plots_dir = ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, predictions, color="steelblue", alpha=0.75)
    bounds = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(bounds, bounds, "r--", linewidth=2)
    plt.xlabel("Actual Annual Production (tonnes)")
    plt.ylabel("Predicted Annual Production (tonnes)")
    plt.title("Random Forest: Actual vs Predicted Production")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "ml_actual_vs_predicted.png", dpi=200)
    plt.close()
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.named_steps["regressor"].feature_importances_, index=transformed_names)
    importances.sort_values(ascending=False).head(10).sort_values().plot(kind="barh", color="darkgreen")
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "ml_feature_importance.png", dpi=200)
    plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_model()
