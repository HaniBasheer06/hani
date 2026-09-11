from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data_pipeline import RAW_DATA, TRAINING_DATA, prepare_data


def audit_frame(path: Path) -> None:
    frame = pd.read_csv(path)
    print(f"\nDATASET: {path.name}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Columns: {', '.join(frame.columns)}")
    print("Dtypes:")
    print(frame.dtypes.to_string())
    print("Missing values:")
    print(frame.isna().sum().to_string())
    print(f"Duplicate rows: {int(frame.duplicated().sum())}")
    for column in ("State", "District", "Soil_Type", "Year", "Synthetic"):
        if column in frame:
            values = sorted(frame[column].dropna().astype(str).unique().tolist())
            print(f"{column} ({len(values)}): {values}")


def main() -> None:
    cleaned, training = prepare_data()
    audit_frame(RAW_DATA)
    audit_frame(TRAINING_DATA)
    print("\nPIPELINE COUNTS")
    print(f"Inventory records: {len(cleaned)}")
    print(f"Real observations: {int((training['Synthetic'] == False).sum())}")
    print(f"Synthetic observations: {int((training['Synthetic'] == True).sum())}")
    print(f"ML training observations: {len(training)}")
    print(f"Years: {sorted(cleaned['Year'].dropna().unique().tolist())}")
    print("\nTARGET LEAKAGE CHECK")
    forbidden = {"Annual_Production_tonnes", "Production_Gap_tonnes", "Reserve_to_Production_ratio", "Reserves_tonnes"}
    print(f"Forbidden target-derived columns used as model features: {sorted(forbidden.intersection(training.columns))}")


if __name__ == "__main__":
    main()
