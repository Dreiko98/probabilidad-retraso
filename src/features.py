from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

FEATURE_COLUMNS = [
    "amigo",
    "tipo_plan",
    "dia_semana",
    "hora_minutos",
    "distancia_km",
    "transporte",
    "lluvia",
    "trafico",
]
CATEGORICAL_FEATURES = ["amigo", "tipo_plan", "dia_semana", "transporte", "trafico"]
NUMERIC_FEATURES = ["hora_minutos", "distancia_km", "lluvia"]

SPANISH_WEEKDAYS = [
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
]


def time_to_minutes(value: str) -> int:
    try:
        hour, minute = (int(part) for part in value.strip().split(":"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Hora no válida: {value!r}") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Hora fuera de rango: {value!r}")
    return hour * 60 + minute


def training_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["hora_minutos"] = result["hora"].map(time_to_minutes)
    return result[FEATURE_COLUMNS]


def historical_profiles(df: pd.DataFrame) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for friend, rows in df.groupby("amigo", sort=False):
        profiles[friend] = {
            "distancia_km": float(rows["distancia_km"].median()),
            "transporte": str(rows["transporte"].mode().iloc[0]),
        }
    return profiles


def build_observations(plan: dict, attendees: Iterable[str], profiles: dict) -> pd.DataFrame:
    hour_minutes = time_to_minutes(plan["hora"])
    rows = []
    for friend in attendees:
        if friend not in profiles:
            raise ValueError(f"No hay perfil histórico para {friend}.")
        profile = profiles[friend]
        rows.append(
            {
                "amigo": friend,
                "tipo_plan": plan["tipo_plan"],
                "dia_semana": plan["dia_semana"],
                "hora_minutos": hour_minutes,
                "distancia_km": profile["distancia_km"],
                "transporte": profile["transporte"],
                "lluvia": int(plan["lluvia"]),
                "trafico": plan["trafico"],
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

