# SIH Manganese Ore Dataset Builder

## Inputs

- `backend/ibm_yearbook/`: IBM Yearbook PDFs and extracted tables.
- `backend/arcgis_data/`: cleaned ArcGIS or OGD India CSV exports.
- `backend/gsi_data/`: GSI or state survey CSV exports.
- `backend/external_data/`: IMD, road, logistics, and weather CSVs.

Place source files in these folders. Screenshots are not inputs.

## Build

```text
python -m pip install -r requirements.txt
python build_dataset.py
```

The script reads CSV files, normalizes headers to the canonical manganese dataset columns, standardizes numeric units, and merges sources using `Deposit_ID` first and `State`/`District` as fallback keys. Missing sources are skipped. No records are invented when the input folders are empty.

## Outputs

  `State`, `District`, `Deposit_ID`, `Latitude`, `Longitude`, `Reserves_tonnes`, `Annual_Production_tonnes`, `Grade_pct`, `Host_Rock`, `Formation`, `Avg_Temperature_C`, `Annual_Precip_mm`, `Rainy_Days`, `Road_Accessibility`, `Distance_to_Port_km`, `Reserve_to_Production_ratio`, `Production_Gap_tonnes`, `Topo_Slope_deg`, `Soil_Type`.
## ML Pipeline
The notebook workflow is still usable with the updated schema when column names are mapped to the canonical names above. Derived fields such as `Reserve_to_Production_ratio` and `Production_Gap_tonnes` are computed during the build step so the downstream model can work directly from the feature table.

## API backend (`backend/`)

Prepare data and train the model:

```text
cd backend
python -m src.data_pipeline
python -m src.train
```

Start the API:

```text
uvicorn main:app --reload
```

The frontend route contract is documented in `FRONTEND_API_SPEC.md`. The persisted model is saved at `backend/models/model.pkl`.
The proposed frontend routes and response shapes are documented in `FRONTEND_API_SPEC.md`. No HTTP backend is included yet.

## Frontend dashboard

The Vite React dashboard lives in `frontend/`. Install the frontend dependencies and start it with:

```text
cd frontend
npm install
npm run dev
```

The dashboard expects the API at `http://localhost:8000/api`. Set `VITE_API_URL` to override that base URL.

For IBM tables reported in thousands of tonnes, convert the values to tonnes before placing the CSV in the input folder. Preserve source PDFs for review and record source year and table provenance in the CSV where possible. The builder writes one consolidated output file: `combined_dataset.csv`.