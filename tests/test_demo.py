from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.config import MODEL_PATH
from src.parser import parse_plan
from src.predictor import load_artifact, predict_plan
from src.training import train_and_save

DEMO = "vamos a cenar a donde siempre y somos carlos, german, gaston y delgado"


@pytest.fixture(scope="session")
def artifact():
    if not MODEL_PATH.exists():
        train_and_save(verbose=False)
    return load_artifact()


def test_happy_path_completo(artifact):
    now = datetime(2026, 8, 26, 12, tzinfo=ZoneInfo("Europe/Madrid"))
    plan = parse_plan(DEMO, artifact["friends"], artifact["plan_types"], now=now)

    assert plan["tipo_plan"] == "cena"
    assert plan["fecha"] == "2026-08-26"
    assert plan["dia_semana"] == "miércoles"
    assert plan["hora"] == "21:30"
    assert plan["lugar"] == "donde siempre"
    assert set(plan["asistentes"]) == {"Carlos", "Germán", "Gastón", "Delgado"}

    ranking, observations, predictions = predict_plan(plan, artifact)
    assert len(observations) == len(predictions) == len(ranking) == 4
    assert len({row["amigo"] for row in ranking}) == 4
    assert [row["retraso_crudo"] for row in ranking] == sorted(
        row["retraso_crudo"] for row in ranking
    )


@pytest.mark.parametrize(
    ("message", "plan_type", "attendee_count", "hour"),
    [
        ("esta noche vamos de fiesta enzo, gaston, oscar y yo", "fiesta", 4, "23:00"),
        ("mañana hemos quedado a cenar a las 22 carlos, gajas y colomino", "cena", 3, "22:00"),
        ("vamos al fútbol german, delgado, enzo y carlos", "fútbol", 4, "17:00"),
        ("el sábado vamos de comida todos", "comida", 8, "14:00"),
    ],
)
def test_otros_mensajes(artifact, message, plan_type, attendee_count, hour):
    now = datetime(2026, 8, 26, 12, tzinfo=ZoneInfo("Europe/Madrid"))
    plan = parse_plan(message, artifact["friends"], artifact["plan_types"], now=now)
    assert plan["tipo_plan"] == plan_type
    assert len(plan["asistentes"]) == attendee_count
    assert plan["hora"] == hour


def test_split_de_quedadas_sin_leakage(artifact):
    assert set(artifact["train_groups"]).isdisjoint(artifact["test_groups"])


def test_mensaje_vacio(artifact):
    with pytest.raises(ValueError, match="vacío"):
        parse_plan("  ", artifact["friends"], artifact["plan_types"])

