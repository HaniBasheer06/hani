from __future__ import annotations

from pathlib import Path
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "data" / "cleaned_districts.csv"
PORTS = pd.DataFrame([
    {"name": "Paradip", "latitude": 20.27, "longitude": 86.70},
    {"name": "Visakhapatnam", "latitude": 17.69, "longitude": 83.22},
    {"name": "Mumbai", "latitude": 18.95, "longitude": 72.95},
    {"name": "Chennai", "latitude": 13.08, "longitude": 80.29},
    {"name": "Mormugao", "latitude": 15.42, "longitude": 73.80},
])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0088
    lat_delta = radians(lat2 - lat1)
    lon_delta = radians(lon2 - lon1)
    value = sin(lat_delta / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(lon_delta / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(value))


class FeatureHydrator:
    def __init__(self, data_path: Path = CLEAN_DATA):
        self.data = pd.read_csv(data_path)
        valid = self.data.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        self.data = valid
        self.tree = KDTree(valid[["Latitude", "Longitude"]].to_numpy())
        self.port_tree = KDTree(PORTS[["latitude", "longitude"]].to_numpy())

    def hydrate(self, latitude: float, longitude: float) -> dict:
        if not (6.0 <= latitude <= 37.5 and 68.0 <= longitude <= 97.5):
            raise ValueError("Coordinates must fall within India's approximate bounding box.")
        _, index = self.tree.query([latitude, longitude])
        nearest = self.data.iloc[int(index)].copy()
        _, port_index = self.port_tree.query([latitude, longitude])
        port = PORTS.iloc[int(port_index)]
        nearest_port_distance = _haversine_km(latitude, longitude, float(port.latitude), float(port.longitude))
        payload = nearest.to_dict()
        payload.update({
            "Latitude": float(latitude),
            "Longitude": float(longitude),
            "Distance_to_Port_km": round(nearest_port_distance, 2),
            "hydrated_from_district": str(nearest.get("District", "Unknown")),
            "nearest_port": str(port["name"]),
        })
        return payload
