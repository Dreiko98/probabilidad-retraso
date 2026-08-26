from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ROOT_DIR

REQUIRED_COLUMNS = {
    "quedada_id",
    "fecha",
    "amigo",
    "tipo_plan",
    "dia_semana",
    "hora",
    "distancia_km",
    "transporte",
    "lluvia",
    "trafico",
    "retraso_min",
}


def find_dataset(root: Path = ROOT_DIR) -> Path:
    """Encuentra el CSV que contiene el histórico requerido."""
    matches: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            columns = set(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
        except Exception:
            continue
        if REQUIRED_COLUMNS.issubset(columns):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No se encontró bajo {root} un CSV con las columnas esperadas."
        )
    return matches[0]


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    path = path or find_dataset()
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(missing)}")
    if df.empty:
        raise ValueError("El dataset está vacío.")
    return df, path


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "observaciones": len(df),
        "quedadas": int(df["quedada_id"].nunique()),
        "amigos": sorted(df["amigo"].unique().tolist()),
        "observaciones_por_amigo": df["amigo"].value_counts().to_dict(),
        "nulos": int(df.isna().sum().sum()),
        "retraso": {
            "media": float(df["retraso_min"].mean()),
            "mediana": float(df["retraso_min"].median()),
            "min": float(df["retraso_min"].min()),
            "max": float(df["retraso_min"].max()),
        },
    }

