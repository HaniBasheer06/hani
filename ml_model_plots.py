from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "combined_dataset.csv"
PLOTS_DIR = ROOT / "plots"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def build_model() -> tuple[Pipeline, dict]:
    df = load_data()

    target = "Annual_Production_tonnes"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    feature_columns = [
        col for col in df.columns if col not in {target, "District", "Deposit_ID"}
    ]

    X = df[feature_columns]
    y = df[target].astype(float)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=8,
        min_samples_leaf=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
        "y_test": y_test,
        "y_pred": pred,
        "feature_names": feature_columns,
    }

    return pipeline, metrics


def plot_results(metrics: dict) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    y_test = metrics["y_test"]
    y_pred = metrics["y_pred"]

    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred, color="steelblue", alpha=0.8)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("Actual Annual Production (tonnes)")
    plt.ylabel("Predicted Annual Production (tonnes)")
    plt.title("ML Model: Actual vs Predicted Production")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ml_actual_vs_predicted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    model = build_model()[0]
    model_steps = model.named_steps["model"]
    feature_importances = getattr(model_steps, "feature_importances_", None)

    if feature_importances is not None:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = pd.Series(feature_importances, index=feature_names)
        importances = importances.sort_values(ascending=False).head(10)
        importances.plot(kind="barh", color="darkgreen")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ml_feature_importance.png", dpi=200)
        plt.close()
    else:
        print("Feature importance is unavailable for this model type.")


def main() -> None:
    pipeline, metrics = build_model()
    print("Model metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")

    plot_results(metrics)
    print(f"Plots saved to: {PLOTS_DIR}")
    print("Generated files:")
    for file in sorted(PLOTS_DIR.iterdir()):
        print(f"- {file.name}")


if __name__ == "__main__":
    main()
