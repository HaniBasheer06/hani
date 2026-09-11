"""Build the SIH manganese ore master dataset from structured source files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MASTER_COLUMNS = [
    "State",
    "District",
    "Deposit_ID",
    "Latitude",
    "Longitude",
    "Reserves_tonnes",
    "Annual_Production_tonnes",
    "Grade_pct",
    "Host_Rock",
    "Formation",
    "Avg_Temperature_C",
    "Annual_Precip_mm",
    "Rainy_Days",
    "Road_Accessibility",
    "Distance_to_Port_km",
    "Reserve_to_Production_ratio",
    "Production_Gap_tonnes",
    "Topo_Slope_deg",
    "Soil_Type",
]
FEATURE_COLUMNS = MASTER_COLUMNS.copy()

STATE_METADATA = {
    "Andhra Pradesh": {
        "Latitude": 15.9129,
        "Longitude": 79.7400,
        "Host_Rock": "Serpentinized ultramafic rocks and khondalitic gneiss",
        "Formation": "Eastern Ghats / Cuddapah metasediments",
        "Avg_Temperature_C": 27.5,
        "Annual_Precip_mm": 1040,
        "Rainy_Days": 60,
        "Road_Accessibility": "Moderate",
        "Distance_to_Port_km": 460,
        "Topo_Slope_deg": 18,
        "Soil_Type": "Red loamy and lateritic soil",
    },
    "Karnataka": {
        "Latitude": 15.3173,
        "Longitude": 75.7139,
        "Host_Rock": "Serpentinized ultramafic rocks and Dharwar schists",
        "Formation": "Dharwar Supergroup / peninsular metamorphic belt",
        "Avg_Temperature_C": 26.8,
        "Annual_Precip_mm": 1160,
        "Rainy_Days": 62,
        "Road_Accessibility": "Good",
        "Distance_to_Port_km": 410,
        "Topo_Slope_deg": 16,
        "Soil_Type": "Red sandy loam and lateritic soil",
    },
    "Madhya Pradesh": {
        "Latitude": 22.9734,
        "Longitude": 78.6569,
        "Host_Rock": "Dolomitic limestone, quartzite and metasediments",
        "Formation": "Vindhyan / Archaean basement belt",
        "Avg_Temperature_C": 25.4,
        "Annual_Precip_mm": 1165,
        "Rainy_Days": 57,
        "Road_Accessibility": "Good",
        "Distance_to_Port_km": 790,
        "Topo_Slope_deg": 14,
        "Soil_Type": "Black cotton and red loamy soil",
    },
    "Maharashtra": {
        "Latitude": 19.7515,
        "Longitude": 75.7139,
        "Host_Rock": "Basaltic laterite, quartzite and metasediments",
        "Formation": "Deccan Traps / Precambrian gneissic suite",
        "Avg_Temperature_C": 27.1,
        "Annual_Precip_mm": 1225,
        "Rainy_Days": 68,
        "Road_Accessibility": "Good",
        "Distance_to_Port_km": 620,
        "Topo_Slope_deg": 12,
        "Soil_Type": "Deep black and red lateritic soil",
    },
    "Odisha": {
        "Latitude": 20.9517,
        "Longitude": 85.0985,
        "Host_Rock": "Khondalite, quartzite and iron formation host rocks",
        "Formation": "Eastern Ghats / Iron Ore Group",
        "Avg_Temperature_C": 26.6,
        "Annual_Precip_mm": 1450,
        "Rainy_Days": 70,
        "Road_Accessibility": "Moderate",
        "Distance_to_Port_km": 280,
        "Topo_Slope_deg": 20,
        "Soil_Type": "Red lateritic and alluvial soil",
    },
    "Rajasthan": {
        "Latitude": 27.0238,
        "Longitude": 74.2179,
        "Host_Rock": "Dolomite, limestone and metamorphic carbonate rocks",
        "Formation": "Aravalli fold belt / Delhi Supergroup",
        "Avg_Temperature_C": 28.8,
        "Annual_Precip_mm": 540,
        "Rainy_Days": 28,
        "Road_Accessibility": "Moderate",
        "Distance_to_Port_km": 980,
        "Topo_Slope_deg": 17,
        "Soil_Type": "Sandy desert and brown loam soil",
    },
    "Telangana": {
        "Latitude": 18.1124,
        "Longitude": 79.0193,
        "Host_Rock": "Quartzite, dolomitic sediment and granite-derived weathered mantle",
        "Formation": "Peninsular gneiss / Cuddapah sediments",
        "Avg_Temperature_C": 27.0,
        "Annual_Precip_mm": 930,
        "Rainy_Days": 52,
        "Road_Accessibility": "Good",
        "Distance_to_Port_km": 510,
        "Topo_Slope_deg": 15,
        "Soil_Type": "Red and black mixed soil",
    },
    "India": {
        "Latitude": 22.3511,
        "Longitude": 78.6677,
        "Host_Rock": "Mixed igneous and metamorphic country rocks",
        "Formation": "Peninsular shield and Indo-Gangetic basin transition",
        "Avg_Temperature_C": 25.8,
        "Annual_Precip_mm": 1170,
        "Rainy_Days": 60,
        "Road_Accessibility": "Good",
        "Distance_to_Port_km": 660,
        "Topo_Slope_deg": 13,
        "Soil_Type": "Alluvial, red and black soil mosaic",
    },
}


def clean_name(name: object) -> str:
    """Convert source headers to the canonical names used by this project."""
    text = str(name).strip().replace("% Mn", "%Mn")
    text = re.sub(r"[‘’']", "", text)
    text = re.sub(r"[(),:/-]+", " ", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    aliases = {
        "State": "State",
        "District": "District",
        "District_Name": "District",
        "Deposit_ID": "Deposit_ID",
        "Latitude": "Latitude",
        "Lat": "Latitude",
        "Longitude": "Longitude",
        "Long": "Longitude",
        "Reserves_tonnes": "Reserves_tonnes",
        "Reserve_tonnes": "Reserves_tonnes",
        "Reserves_000_tonnes": "Reserves_tonnes",
        "Reserve_000_tonnes": "Reserves_tonnes",
        "Annual_Production_tonnes": "Annual_Production_tonnes",
        "Annual_Production": "Annual_Production_tonnes",
        "Production": "Annual_Production_tonnes",
        "Grade_pct": "Grade_pct",
        "Grade_%Mn": "Grade_pct",
        "Grade_Mn": "Grade_pct",
        "Host_Rock": "Host_Rock",
        "Formation": "Formation",
        "Avg_Temperature_C": "Avg_Temperature_C",
        "Temperature_C": "Avg_Temperature_C",
        "Avg_Temp_C": "Avg_Temperature_C",
        "Annual_Precip_mm": "Annual_Precip_mm",
        "Annual_Precipitation_mm": "Annual_Precip_mm",
        "Rainfall_mm": "Annual_Precip_mm",
        "Rainy_Days": "Rainy_Days",
        "Road_Accessibility": "Road_Accessibility",
        "Road_Access": "Road_Accessibility",
        "Distance_to_Port_km": "Distance_to_Port_km",
        "Port_Distance_km": "Distance_to_Port_km",
        "Reserve_to_Production_ratio": "Reserve_to_Production_ratio",
        "Reserve_to_Production": "Reserve_to_Production_ratio",
        "Production_Gap_tonnes": "Production_Gap_tonnes",
        "Production_Gap": "Production_Gap_tonnes",
        "Topo_Slope_deg": "Topo_Slope_deg",
        "Slope_deg": "Topo_Slope_deg",
        "Soil_Type": "Soil_Type",
        "Soil": "Soil_Type",
    }
    return aliases.get(text, text)


def read_csv_folder(folder: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("*.csv")):
        frame = pd.read_csv(path)
        thousand_columns = {
            clean_name(column)
            for column in frame.columns
            if "000" in str(column) or "thousand" in str(column).lower()
        }
        frame.columns = [clean_name(column) for column in frame.columns]
        for column in thousand_columns & {"Reserve_tonnes", "Annual_Production", "Expected_Production"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * 1000
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_source_folder(folder: Path) -> pd.DataFrame:
    frames = [read_csv_folder(folder)]
    if folder.name == "arcgis_data":
        try:
            import geopandas as gpd
        except ImportError:
            gpd = None
        if gpd is not None:
            for path in sorted(folder.glob("*.geojson")) + sorted(folder.glob("*.shp")):
                frame = gpd.read_file(path).drop(columns=["geometry"], errors="ignore")
                frame.columns = [clean_name(column) for column in frame.columns]
                frames.append(pd.DataFrame(frame))
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def standardize_units(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("Reserves_tonnes", "Annual_Production_tonnes", "Grade_pct", "Latitude", "Longitude", "Distance_to_Port_km", "Avg_Temperature_C", "Annual_Precip_mm", "Rainy_Days", "Topo_Slope_deg"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "Grade_pct" in frame and "Grade_%Mn" in frame:
        frame["Grade_pct"] = frame["Grade_pct"].combine_first(frame["Grade_%Mn"])
    if "Annual_Production_tonnes" in frame and "Annual_Production" in frame:
        frame["Annual_Production_tonnes"] = frame["Annual_Production_tonnes"].combine_first(frame["Annual_Production"])
    if "Reserves_tonnes" in frame and "Reserve_tonnes" in frame:
        frame["Reserves_tonnes"] = frame["Reserves_tonnes"].combine_first(frame["Reserve_tonnes"])
    if "Production_Gap_tonnes" in frame and "Production_Gap" in frame:
        frame["Production_Gap_tonnes"] = frame["Production_Gap_tonnes"].combine_first(frame["Production_Gap"])
    if "Reserve_to_Production_ratio" in frame and "Reserve_to_Production" in frame:
        frame["Reserve_to_Production_ratio"] = frame["Reserve_to_Production_ratio"].combine_first(frame["Reserve_to_Production"])
    return frame


def extract_pdf_tables() -> None:
    """Extract every PDF table when pdfplumber is installed.

    PDFs are intentionally kept as source files; extracted tables are written
    beside them with a predictable name for later review.
    """
    try:
        import pdfplumber
    except ImportError:
        print("Skipping PDF extraction: install requirements.txt first.")
        return

    for pdf_path in sorted((ROOT / "ibm_yearbook").glob("*.pdf")):
        with pdfplumber.open(pdf_path) as pdf:
            table_number = 0
            for page_number, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    if not table or len(table) < 2:
                        continue
                    table_number += 1
                    output = pd.DataFrame(table[1:], columns=table[0])
                    output.columns = [clean_name(column) for column in output.columns]
                    output.to_csv(
                        ROOT / "ibm_yearbook" / f"{pdf_path.stem}_p{page_number}_t{table_number}.csv",
                        index=False,
                    )


def merge_sources() -> pd.DataFrame:
    folders = ["ibm_yearbook", "arcgis_data", "gsi_data", "external_data"]
    frames = [read_source_folder(ROOT / folder) for folder in folders]
    frames = [standardize_units(frame) for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    result = frames[0]
    for frame in frames[1:]:
        if "Deposit_ID" in result and "Deposit_ID" in frame:
            keys = ["Deposit_ID"]
        elif {"State", "District"}.issubset(result.columns) and {"State", "District"}.issubset(frame.columns):
            keys = ["State", "District"]
        elif "State" in result.columns and "State" in frame.columns:
            keys = ["State"]
        else:
            continue
        result = result.merge(frame, on=keys, how="outer", suffixes=("", "_source"))
        result = result.drop(columns=[column for column in result if column.endswith("_source")])
    return result


def fill_missing_state_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill blank geology, weather, and terrain fields using state-level defaults."""
    result = frame.copy()
    for column in MASTER_COLUMNS:
        if column not in result:
            result[column] = pd.NA

    for state, metadata in STATE_METADATA.items():
        mask = result["State"].astype(str).str.strip().eq(state)
        if not mask.any():
            continue
        for key, value in metadata.items():
            if key not in result.columns:
                continue
            result.loc[mask & result[key].isna(), key] = value

    if "District" in result:
        result["District"] = result["District"].replace({pd.NA: ""})
        result["District"] = result["District"].astype(str).str.strip()
        state_rows = result["State"].notna() & (result["District"] == "")
        result.loc[state_rows, "District"] = result.loc[state_rows, "State"]

    for column in ("Latitude", "Longitude", "Reserves_tonnes", "Annual_Production_tonnes", "Grade_pct", "Avg_Temperature_C", "Annual_Precip_mm", "Rainy_Days", "Distance_to_Port_km", "Topo_Slope_deg"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "Annual_Production_tonnes" in result:
        production = pd.to_numeric(result["Annual_Production_tonnes"], errors="coerce")
        reserve = pd.to_numeric(result.get("Reserves_tonnes", pd.Series(dtype="float64")), errors="coerce")
        ratio = reserve.div(production.replace(0, pd.NA))
        if "Reserve_to_Production_ratio" in result:
            result["Reserve_to_Production_ratio"] = result["Reserve_to_Production_ratio"].combine_first(ratio)
        else:
            result["Reserve_to_Production_ratio"] = ratio
    else:
        result["Reserve_to_Production_ratio"] = pd.NA

    if "Expected_Production" in result:
        expected = pd.to_numeric(result["Expected_Production"], errors="coerce")
        production = pd.to_numeric(result.get("Annual_Production_tonnes", pd.Series(dtype="float64")), errors="coerce")
        result["Production_Gap_tonnes"] = expected.sub(production)
    else:
        result["Production_Gap_tonnes"] = pd.NA

    return result


def build() -> None:
    extract_pdf_tables()
    # If clean_ibm_tables exists, ensure cleaned tables are ready
    try:
        from clean_ibm_tables import extract_table4_production, parse_raw_table5
        t4 = extract_table4_production()
        t4.to_csv(ROOT / "ibm_yearbook" / "cleaned_table4_state_production.csv", index=False)
        if (ROOT / "ibm_yearbook" / "234_production.csv").exists():
            t5a = parse_raw_table5(ROOT / "ibm_yearbook" / "234_production.csv", "2022-23")
            t5a.to_csv(ROOT / "ibm_yearbook" / "cleaned_table5a_2022_23.csv", index=False)
        if (ROOT / "ibm_yearbook" / "123_production.csv").exists():
            t5b = parse_raw_table5(ROOT / "ibm_yearbook" / "123_production.csv", "2023-24")
            t5b.to_csv(ROOT / "ibm_yearbook" / "cleaned_table5b_2023_24.csv", index=False)
    except Exception as e:
        print(f"Notice: clean_ibm_tables helper skipped: {e}")

    result = merge_sources()
    for column in MASTER_COLUMNS:
        if column not in result:
            result[column] = pd.NA

    result["State"] = result.get("State", pd.Series(dtype="object")).astype("string")
    result["District"] = result.get("District", pd.Series(dtype="object")).astype("string")
    result["Deposit_ID"] = result.get("Deposit_ID", pd.Series(dtype="object")).astype("string")

    result = fill_missing_state_metadata(result)

    result = result[result["State"].notna() | result["Deposit_ID"].notna()].copy()
    result = result[MASTER_COLUMNS].drop_duplicates().reset_index(drop=True)

    result.to_csv(ROOT / "combined_dataset.csv", index=False)
    print(f"Wrote {len(result)} valid records to combined_dataset.csv")


if __name__ == "__main__":
    build()
