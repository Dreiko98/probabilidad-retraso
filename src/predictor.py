from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import joblib

from .config import MODEL_PATH
from .features import build_observations


def load_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}. Ejecuta python train.py.")
    artifact = joblib.load(path)
    required = {"pipeline", "profiles", "friends", "plan_types"}
    if not required.issubset(artifact):
        raise ValueError("El artefacto de modelo no tiene el formato esperado.")
    return artifact


def predict_plan(plan: dict, artifact: dict) -> tuple[list[dict], object, list[float]]:
    observations = build_observations(plan, plan["asistentes"], artifact["profiles"])
    raw_predictions = artifact["pipeline"].predict(observations).tolist()
    meeting = datetime.strptime(plan["hora"], "%H:%M")
    ranking = []
    for friend, raw_delay in zip(plan["asistentes"], raw_predictions):
        rounded = int(round(float(raw_delay)))
        arrival = meeting + timedelta(minutes=rounded)
        ranking.append(
            {
                "amigo": friend,
                "retraso_min": rounded,
                "retraso_crudo": float(raw_delay),
                "llegada": arrival.strftime("%H:%M"),
            }
        )
    ranking.sort(key=lambda row: row["retraso_crudo"])
    return ranking, observations, raw_predictions

