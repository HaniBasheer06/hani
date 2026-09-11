# Frontend API Specification

The repository now includes a FastAPI backend in `main.py`. Start it with `uvicorn main:app --reload` from the project root.

Base URL: `/api`

## 1. Health

`GET /api/health`

Response:

```json
{
  "status": "ok",
  "dataset_rows": 31,
  "dataset_file": "data/cleaned_districts.csv"
}
```

## 2. Dataset summary

`GET /api/summary`

Returns headline values for dashboard cards.

```json
{
  "total_records": 31,
  "states": 6,
  "districts": 6,
  "total_annual_production_tonnes": 7227957.0,
  "average_grade_pct": null,
  "year_range": ["2022-23", "2023-24"]
}
```

## 3. Deposit list

`GET /api/deposits`

Query parameters:

- `state`: filter by state
- `district`: filter by district
- `search`: search state, district, or deposit ID
- `page`: 1-based page number
- `limit`: page size, recommended default 25

Response:

```json
{
  "items": [
    {
      "State": "Odisha",
      "District": "Keonjhar",
      "Deposit_ID": "OD-001",
      "Latitude": 21.63,
      "Longitude": 85.58,
      "Annual_Production_tonnes": 100000.0,
      "Grade_pct": null,
      "Host_Rock": null,
      "Formation": null,
      "Soil_Type": null
    }
  ],
  "page": 1,
  "limit": 25,
  "total": 75
}
```

## 4. Deposit details

`GET /api/deposits/{deposit_id}`

Returns one complete row using the canonical dataset fields:

`State`, `District`, `Deposit_ID`, `Latitude`, `Longitude`, `Reserves_tonnes`, `Annual_Production_tonnes`, `Grade_pct`, `Host_Rock`, `Formation`, `Avg_Temperature_C`, `Annual_Precip_mm`, `Rainy_Days`, `Road_Accessibility`, `Distance_to_Port_km`, `Reserve_to_Production_ratio`, `Production_Gap_tonnes`, `Topo_Slope_deg`, and `Soil_Type`.

## 5. Production trend

`GET /api/production/trend`

Query parameters:

- `state`: optional state filter
- `district`: optional district filter
- `deposit_id`: optional deposit filter

Response:

```json
{
  "series": [
    {"year": "2019-20", "production_tonnes": 100000.0},
    {"year": "2020-21", "production_tonnes": 110000.0}
  ]
}
```

## 6. Geographic data

`GET /api/map/deposits`

Returns map-ready records. Only rows with valid coordinates should be returned.

```json
{
  "points": [
    {
      "deposit_id": "OD-001",
      "state": "Odisha",
      "district": "Keonjhar",
      "latitude": 21.63,
      "longitude": 85.58,
      "production_tonnes": 100000.0,
      "grade_pct": null
    }
  ]
}
```

## 7. Model metrics

`GET /api/model/metrics`

Response:

```json
{
  "model": "RandomForestRegressor",
  "target": "Annual_Production_tonnes",
  "mae": 253465.18,
  "rmse": 341110.32,
  "r2": 0.1309,
  "charts": {
    "actual_vs_predicted": "/plots/ml_actual_vs_predicted.png",
    "feature_importance": "/plots/ml_feature_importance.png"
  }
}
```

## 8. Production prediction

`POST /api/model/predict`

Request body should contain the model input fields. `Annual_Production_tonnes` is not included because it is the target.

```json
{
  "State": "Odisha",
  "Latitude": 21.63,
  "Longitude": 85.58,
  "Reserves_tonnes": null,
  "Grade_pct": null,
  "Host_Rock": null,
  "Formation": null,
  "Avg_Temperature_C": 27.0,
  "Annual_Precip_mm": 1400.0,
  "Rainy_Days": 100,
  "Road_Accessibility": "Moderate",
  "Distance_to_Port_km": 300.0,
  "Reserve_to_Production_ratio": null,
  "Production_Gap_tonnes": null,
  "Topo_Slope_deg": 5.0,
  "Soil_Type": "Laterite"
}
```

Response:

```json
{
  "predicted_annual_production_tonnes": 296803.45,
  "feasibility_rating": "Moderate Potential",
  "accuracy_metrics": {
    "model_r2_score": 0.1309,
    "r2_percentage": "13.1%"
  },
  "hydrated_from_district": "Odisha",
  "nearest_port": "Paradip",
  "explanation": {
    "top_contributing_factors": []
  }
}
```

## Frontend priorities

Start with these endpoints:

1. `/api/summary`
2. `/api/deposits`
3. `/api/map/deposits`
4. `/api/production/trend`
5. `/api/model/metrics`
6. `/api/model/predict`

The backend trains from the cleaned district data in `backend/data/cleaned_districts.csv` and serves the consolidated raw source at `backend/combined_dataset.csv` for auditability. Coordinate-only prediction requests are hydrated with the nearest district context and nearest major port.