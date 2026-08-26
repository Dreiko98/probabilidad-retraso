from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
TIMEZONE = "Europe/Madrid"
SELF_NAME = os.getenv("SELF_NAME", "Germán")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_HOURS = {
    "cena": "21:30",
    "fiesta": "23:00",
    "cervezas": "20:30",
    "fútbol": "17:00",
    "playa": "11:00",
    "comida": "14:00",
    "cine": "19:30",
    "viaje": "08:30",
    "cumpleaños": "21:00",
    "pádel": "18:30",
}

DEFAULT_RAIN = 0
DEFAULT_TRAFFIC = "medio"

