from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from src.data_pipeline import CLEAN_DATA, RAW_DATA, TARGET, TRAINING_DATA
from src.hydration import FeatureHydrator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.pkl"
PLOTS_DIR = ROOT / "plots"
app = FastAPI(title="India Mining Minerals Analytics API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _load_data() -> pd.DataFrame:
    if not CLEAN_DATA.exists():
        raise RuntimeError("Run python -m src.data_pipeline before starting the API.")
    return pd.read_csv(CLEAN_DATA)


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError("Run python -m src.train before using model endpoints.")
    return joblib.load(MODEL_PATH)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    Latitude: float | None = None
    Longitude: float | None = None
    State: str | None = None
    District: str | None = None
    Avg_Temperature_C: float | None = None
    Annual_Precip_mm: float | None = None
    Soil_Type: str | None = None
    Topo_Slope_deg: float | None = None
    Road_Accessibility: str | None = None
    Distance_to_Port_km: float | None = None


@app.get("/")
def root() -> dict:
    return {
        "service": "India Mining Minerals Analytics API",
        "status": "ok",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    data = _load_data()
    return {"status": "ok", "dataset_rows": len(data), "inventory_rows": len(data), "dataset_file": "data/cleaned_districts.csv"}


@app.get("/api/summary")
def summary() -> dict:
    data = _load_data()
    raw = pd.read_csv(RAW_DATA)
    training = pd.read_csv(TRAINING_DATA) if TRAINING_DATA.exists() else pd.DataFrame()
    production = pd.to_numeric(data[TARGET], errors="coerce").sum()
    return {
        "total_records": int(len(data)),
        "inventory_records": int(len(data)),
        "raw_records": int(len(raw)),
        "training_rows": int(len(training)),
        "synthetic_training_rows": int(training.get("Synthetic", pd.Series(dtype=bool)).eq(True).sum()),
        "original_training_rows": int(training.get("Synthetic", pd.Series(dtype=bool)).eq(False).sum()),
        "real_observations": int(training.get("Synthetic", pd.Series(dtype=bool)).eq(False).sum()),
        "synthetic_observations": int(training.get("Synthetic", pd.Series(dtype=bool)).eq(True).sum()),
        "states": int(data["State"].nunique()),
        "districts": int(data["District"].nunique()),
        "total_annual_production_tonnes": float(production),
        "average_grade_pct": _json_value(pd.to_numeric(data.get("Grade_pct"), errors="coerce").mean()),
        "year_range": sorted(data["Year"].dropna().unique().tolist()),
        "year_basis": "financial_year",
        "soil_types": sorted(data["Soil_Type"].dropna().astype(str).unique().tolist()),
        "provenance_note": "ML training includes augmented observations derived from available historical inventory data.",
    }


@app.get("/api/deposits")
def deposits(state: str | None = None, district: str | None = None, search: str | None = None, page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=200)) -> dict:
    data = _load_data()
    if state:
        data = data[data["State"].str.casefold() == state.casefold()]
    if district:
        data = data[data["District"].str.casefold() == district.casefold()]
    if search:
        mask = data.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
        data = data[mask]
    total = len(data)
    page_data = data.iloc[(page - 1) * limit : page * limit]
    return {"items": page_data.replace({np.nan: None}).to_dict(orient="records"), "page": page, "limit": limit, "total": total}


@app.get("/api/deposits/{deposit_id}")
def deposit_detail(deposit_id: str) -> dict:
    data = _load_data()
    matches = data[data["Deposit_ID"].astype(str) == deposit_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Deposit not found")
    return matches.iloc[0].replace({np.nan: None}).to_dict()


@app.get("/api/map/deposits")
def map_deposits() -> dict:
    data = _load_data().dropna(subset=["Latitude", "Longitude"])
    points = [{
        "deposit_id": row.get("Deposit_ID"), "state": row.get("State"), "district": row.get("District"),
        "latitude": float(row["Latitude"]), "longitude": float(row["Longitude"]),
        "production_tonnes": _json_value(row.get(TARGET)), "grade_pct": _json_value(row.get("Grade_pct")),
    } for _, row in data.iterrows()]
    return {"points": points}


@app.get("/api/production/trend")
def production_trend(state: str | None = None, district: str | None = None, deposit_id: str | None = None) -> dict:
    data = _load_data()
    if state:
        data = data[data["State"].str.casefold() == state.casefold()]
    if district:
        data = data[data["District"].str.casefold() == district.casefold()]
    if deposit_id:
        data = data[data["Deposit_ID"].astype(str) == deposit_id]
    series = data.groupby("Year", as_index=False)[TARGET].sum().sort_values("Year")
    return {"series": [{"year": row["Year"], "production_tonnes": float(row[TARGET])} for _, row in series.iterrows()]}


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    bundle = _load_bundle()
    metrics = bundle["metrics"].copy()
    metrics["charts"] = {
        "actual_vs_predicted": "/plots/ml_actual_vs_predicted.png",
        "feature_importance": "/plots/ml_feature_importance.png",
    }
    return metrics


def _training_coverage(bundle: dict, row: dict) -> dict:
    training = pd.read_csv(TRAINING_DATA)
    numeric_features = [
        feature for feature in bundle["features"]
        if feature in training.columns and pd.api.types.is_numeric_dtype(training[feature])
    ]
    out_of_range = []
    for feature in numeric_features:
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        values = pd.to_numeric(training[feature], errors="coerce").dropna()
        if pd.isna(value) or values.empty:
            continue
        minimum, maximum = float(values.min()), float(values.max())
        if value < minimum or value > maximum:
            out_of_range.append({"feature": feature, "value": float(value), "training_range": [minimum, maximum]})
    unknown_categories = []
    for feature in ["State", "Road_Accessibility", "Soil_Type", "Host_Rock", "Formation"]:
        if feature in row and feature in training.columns and row.get(feature) not in set(training[feature].dropna().astype(str)):
            unknown_categories.append(feature)
    score = 100 - min(100, len(out_of_range) * 20 + len(unknown_categories) * 10)
    level = "Moderate" if unknown_categories and score >= 50 else "High" if score >= 80 else "Moderate" if score >= 50 else "Low"
    return {
        "level": level,
        "score": score,
        "out_of_range_features": out_of_range,
        "unknown_categories": unknown_categories,
        "warning": "Some input conditions are underrepresented or use unseen categories in training data." if out_of_range or unknown_categories else None,
    }


def _explanations(bundle: dict, row: dict, prediction: float) -> list[dict]:
    pipeline = bundle["pipeline"]
    features = bundle["features"]
    baseline = bundle["reference"]
    impacts = []
    for feature in features:
        changed = row.copy()
        changed[feature] = baseline.get(feature, row.get(feature))
        counterfactual = float(pipeline.predict(pd.DataFrame([changed], columns=features))[0])
        contribution = prediction - counterfactual
        if abs(contribution) > 1:
            impacts.append({"feature": feature, "value": str(row.get(feature)), "impact": "increased" if contribution > 0 else "decreased", "contribution_tonnes": round(contribution, 2), "reason": "Model contribution estimated by replacing this value with the training reference."})
    return sorted(impacts, key=lambda item: abs(item["contribution_tonnes"]), reverse=True)[:5]


@app.post("/api/model/predict")
def predict(request: PredictionRequest) -> dict:
    if request.Latitude is not None and not 4 <= request.Latitude <= 38:
        raise HTTPException(status_code=422, detail="Latitude must be between 4 and 38 degrees for the supported India map extent.")
    if request.Longitude is not None and not 60 <= request.Longitude <= 105:
        raise HTTPException(status_code=422, detail="Longitude must be between 60 and 105 degrees for the supported India map extent.")
    bundle = _load_bundle()
    body = request.model_dump(exclude_none=True)
    if body.get("Latitude") is not None and body.get("Longitude") is not None:
        missing_environment = any(body.get(key) is None for key in ["Annual_Precip_mm", "Soil_Type", "Topo_Slope_deg", "Road_Accessibility", "Distance_to_Port_km"])
        if missing_environment:
            try:
                body = {**FeatureHydrator().hydrate(float(body["Latitude"]), float(body["Longitude"])), **body}
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
    reference = bundle.get("reference", {})
    row = {feature: body.get(feature, reference.get(feature)) for feature in bundle["features"]}
    missing = [feature for feature, value in row.items() if value is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model features: {missing}")
    prediction = float(bundle["pipeline"].predict(pd.DataFrame([row]))[0])
    r2 = float(bundle["metrics"]["r2"])
    coverage = _training_coverage(bundle, row)
    rating = "High Potential" if prediction >= 500000 else "Moderate Potential" if prediction >= 100000 else "Low Potential"
    return {
        "predicted_annual_production_tonnes": round(prediction, 2),
        "feasibility_rating": rating,
        "accuracy_metrics": {"model_r2_score": round(r2, 4), "r2_percentage": f"{r2 * 100:.1f}%"},
        "training_data_coverage": coverage,
        "input_features_used": {feature: _json_value(row.get(feature)) for feature in bundle["features"]},
        "hydrated_from_district": body.get("hydrated_from_district"),
        "nearest_port": body.get("nearest_port"),
        "explanation": {"top_contributing_factors": _explanations(bundle, row, prediction)},
    }


@app.get("/plots/{filename}")
def chart(filename: str):
    from fastapi.responses import FileResponse
    path = PLOTS_DIR / filename
    if not path.is_file() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(path)
