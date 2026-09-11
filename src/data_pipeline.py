from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "combined_dataset.csv"
DATA_DIR = ROOT / "data"
CLEAN_DATA = DATA_DIR / "cleaned_districts.csv"
TRAINING_DATA = DATA_DIR / "training_dataset.csv"
TARGET = "Annual_Production_tonnes"

NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Avg_Temperature_C",
    "Annual_Precip_mm",
    "Rainy_Days",
    "Distance_to_Port_km",
    "Topo_Slope_deg",
    "Year_Index",
]
CATEGORICAL_FEATURES = ["State", "Road_Accessibility", "Soil_Type", "Host_Rock", "Formation"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_state_summary(row: pd.Series) -> bool:
    state = _normalise(row.get("State", ""))
    deposit_id = str(row.get("Deposit_ID", "")).strip()
    if not deposit_id or state == "india":
        return True
    if _normalise(deposit_id) in {"india", "national", "allindia"}:
        return True
    last_token = deposit_id.rsplit("_", 1)[-1]
    return _normalise(last_token) == state


def _source_year_lookup() -> dict[tuple[str, float], str]:
    lookup: dict[tuple[str, float], str] = {}
    sources = [
        (ROOT / "ibm_yearbook" / "cleaned_table5a_2022_23.csv", "2022-23"),
        (ROOT / "ibm_yearbook" / "cleaned_table5b_2023_24.csv", "2023-24"),
    ]
    for path, year in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not {"Deposit_ID", "Annual_Production"}.issubset(frame.columns):
            continue
        for _, row in frame.iterrows():
            deposit_id = str(row["Deposit_ID"]).strip()
            production = pd.to_numeric(row["Annual_Production"], errors="coerce")
            if deposit_id and pd.notna(production):
                lookup[(deposit_id, float(production))] = year
    return lookup


def _assign_years(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lookup = _source_year_lookup()
    production = pd.to_numeric(result[TARGET], errors="coerce")
    result["Year"] = [
        lookup.get((str(deposit_id).strip(), float(value)), pd.NA)
        if pd.notna(value)
        else pd.NA
        for deposit_id, value in zip(result["Deposit_ID"], production)
    ]
    for deposit_id, indexes in result.groupby("Deposit_ID", sort=False).groups.items():
        missing = result.loc[indexes, "Year"].isna()
        fallback_years = ["2022-23", "2023-24"]
        for position, index in enumerate(result.loc[indexes].index):
            if pd.isna(result.at[index, "Year"]):
                result.at[index, "Year"] = fallback_years[min(position, 1)]
    result["Year"] = result["Year"].fillna("2023-24")
    result["Year_Index"] = result["Year"].map({"2019-20": 0, "2020-21": 1, "2021-22": 2, "2022-23": 3, "2023-24": 4}).fillna(4)
    return result


def load_clean_dataset() -> pd.DataFrame:
    frame = pd.read_csv(RAW_DATA)
    frame = frame.loc[~frame.apply(_is_state_summary, axis=1)].copy()
    frame = frame.loc[frame[TARGET].notna()].copy()
    frame = _assign_years(frame)
    frame = frame.drop_duplicates(subset=["Deposit_ID", "Year"], keep="last")

    for column in NUMERIC_FEATURES + [TARGET]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in NUMERIC_FEATURES:
        if column in frame:
            frame[column] = frame[column].fillna(frame[column].median())
    for column in CATEGORICAL_FEATURES:
        if column in frame:
            frame[column] = frame[column].fillna("Unknown").astype(str)
    return frame.reset_index(drop=True)


def generate_synthetic_data(cleaned: pd.DataFrame, samples: int = 360, seed: int = 42) -> pd.DataFrame:
    if cleaned.empty:
        raise ValueError("No district-level records remain after aggregate filtering.")
    rng = np.random.default_rng(seed)
    rows = []
    for index in rng.integers(0, len(cleaned), size=max(samples, len(cleaned))):
        row = cleaned.iloc[index].copy()
        row["Synthetic"] = True
        row["Annual_Precip_mm"] = max(250.0, float(row["Annual_Precip_mm"]) * rng.normal(1.0, 0.035))
        row["Topo_Slope_deg"] = max(0.2, float(row["Topo_Slope_deg"]) + rng.normal(0, 1.0))
        row["Avg_Temperature_C"] = float(row["Avg_Temperature_C"]) + rng.normal(0, 0.35)
        row["Rainy_Days"] = max(1.0, float(row["Rainy_Days"]) + rng.normal(0, 2.0))
        row["Latitude"] = float(row["Latitude"]) + rng.normal(0, 0.04)
        row["Longitude"] = float(row["Longitude"]) + rng.normal(0, 0.04)
        row[TARGET] = max(0.0, float(row[TARGET]) * rng.lognormal(0, 0.08))
        rows.append(row)
    augmented = pd.DataFrame(rows)
    original = cleaned.copy()
    original["Synthetic"] = False
    return pd.concat([original, augmented], ignore_index=True)


def prepare_data(samples: int = 360, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = load_clean_dataset()
    training = generate_synthetic_data(cleaned, samples=samples, seed=seed)
    DATA_DIR.mkdir(exist_ok=True)
    cleaned.to_csv(CLEAN_DATA, index=False)
    training.to_csv(TRAINING_DATA, index=False)
    return cleaned, training


if __name__ == "__main__":
    cleaned, training = prepare_data()
    print(f"Cleaned district rows: {len(cleaned)}")
    print(f"Training rows: {len(training)}")
    print(f"Years: {sorted(cleaned['Year'].unique())}")
