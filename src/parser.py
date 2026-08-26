from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .config import (
    DEFAULT_HOURS,
    DEFAULT_RAIN,
    DEFAULT_TRAFFIC,
    OPENAI_MODEL,
    SELF_NAME,
    TIMEZONE,
)
from .features import SPANISH_WEEKDAYS

PLAN_ALIASES = {
    "cena": ("cena", "cenar", "cenamos"),
    "fiesta": ("fiesta",),
    "cervezas": ("cervezas", "cerveza", "birras"),
    "fútbol": ("futbol", "fútbol"),
    "playa": ("playa",),
    "comida": ("comida", "comer"),
    "cine": ("cine", "pelicula", "película"),
    "viaje": ("viaje", "viajar"),
    "cumpleaños": ("cumpleanos", "cumpleaños", "cumple"),
    "pádel": ("padel", "pádel"),
}


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _canonical(value: str, choices: list[str]) -> str | None:
    wanted = _plain(value).strip()
    return next((choice for choice in choices if _plain(choice) == wanted), None)


def _next_weekday(today: date, weekday: int) -> date:
    delta = (weekday - today.weekday()) % 7
    return today + timedelta(days=delta)


def _parse_date(text: str, today: date) -> date:
    plain = _plain(text)
    if "pasado manana" in plain:
        return today + timedelta(days=2)
    if "manana" in plain:
        return today + timedelta(days=1)
    for index, weekday in enumerate(SPANISH_WEEKDAYS):
        if _plain(weekday) in plain:
            return _next_weekday(today, index)
    match = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", plain)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year
        if year < 100:
            year += 2000
        candidate = date(year, month, day)
        if not match.group(3) and candidate < today:
            candidate = date(year + 1, month, day)
        return candidate
    return today


def _parse_hour(text: str, plan_type: str) -> str:
    plain = _plain(text)
    patterns = (
        r"\ba\s+las\s+(\d{1,2})(?:[:.]([0-5]\d))?\b",
        r"\b(?:a\s+la\s+)?(\d{1,2}):([0-5]\d)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, plain)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            if 0 <= hour <= 23:
                return f"{hour:02d}:{minute:02d}"
    return DEFAULT_HOURS.get(plan_type, "20:30")


def _rule_parse(message: str, friends: list[str], plan_types: list[str], today: date) -> dict:
    plain = _plain(message)
    plan_type = None
    for canonical, aliases in PLAN_ALIASES.items():
        if canonical in plan_types and any(re.search(rf"\b{re.escape(_plain(a))}\b", plain) for a in aliases):
            plan_type = canonical
            break
    if plan_type is None:
        for candidate in plan_types:
            if re.search(rf"\b{re.escape(_plain(candidate))}\b", plain):
                plan_type = candidate
                break
    if plan_type is None:
        raise ValueError(f"No he podido identificar el tipo de plan. Opciones: {', '.join(plan_types)}")

    attendees: list[str] = []
    if re.search(r"\btodos\b", plain):
        attendees = list(friends)
    else:
        for friend in friends:
            if re.search(rf"\b{re.escape(_plain(friend))}\b", plain):
                attendees.append(friend)
        if re.search(r"\byo\b", plain):
            self_name = _canonical(SELF_NAME, friends)
            if self_name and self_name not in attendees:
                attendees.append(self_name)
    if not attendees:
        raise ValueError(f"No he encontrado asistentes válidos. Amigos: {', '.join(friends)}")

    plan_date = _parse_date(message, today)
    return {
        "tipo_plan": plan_type,
        "fecha": plan_date.isoformat(),
        "dia_semana": SPANISH_WEEKDAYS[plan_date.weekday()],
        "hora": _parse_hour(message, plan_type),
        "lugar": "donde siempre" if "donde siempre" in plain else "sin especificar",
        "asistentes": attendees,
        "lluvia": 1 if re.search(r"\b(llueve|lloviendo|lluvia)\b", plain) else DEFAULT_RAIN,
        "trafico": "alto" if "mucho trafico" in plain or "trafico alto" in plain else DEFAULT_TRAFFIC,
        "parser": "reglas",
    }


def _llm_parse(message: str, friends: list[str], plan_types: list[str], today: date) -> dict:
    from openai import OpenAI

    prompt = f"""Interpreta un plan entre amigos y devuelve exclusivamente JSON.
Fecha actual en Europe/Madrid: {today.isoformat()}.
Amigos válidos: {json.dumps(friends, ensure_ascii=False)}.
Tipos válidos: {json.dumps(plan_types, ensure_ascii=False)}.
Campos: tipo_plan, fecha (YYYY-MM-DD), hora (HH:MM o null), lugar (o null),
asistentes (solo nombres válidos), lluvia (0/1 o null), trafico (bajo/medio/alto o null).
"todos" significa todos los amigos y "yo" significa {SELF_NAME}.
No predigas retrasos ni orden de llegada. Mensaje: {message!r}"""
    response = OpenAI(timeout=8.0, max_retries=0).chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def _validate_and_complete(raw: dict, message: str, friends: list[str], plan_types: list[str], today: date) -> dict:
    plan_type = _canonical(str(raw.get("tipo_plan", "")), plan_types)
    if not plan_type:
        raise ValueError("El LLM devolvió un tipo de plan no válido.")
    attendees = []
    for value in raw.get("asistentes") or []:
        friend = _canonical(str(value), friends)
        if friend and friend not in attendees:
            attendees.append(friend)
    if not attendees:
        raise ValueError("El LLM no devolvió asistentes válidos.")
    try:
        plan_date = date.fromisoformat(str(raw.get("fecha")))
    except ValueError:
        plan_date = _parse_date(message, today)
    hour = raw.get("hora") or _parse_hour(message, plan_type)
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(hour)):
        hour = _parse_hour(message, plan_type)
    traffic = raw.get("trafico") if raw.get("trafico") in {"bajo", "medio", "alto"} else DEFAULT_TRAFFIC
    return {
        "tipo_plan": plan_type,
        "fecha": plan_date.isoformat(),
        "dia_semana": SPANISH_WEEKDAYS[plan_date.weekday()],
        "hora": hour,
        "lugar": raw.get("lugar") or "sin especificar",
        "asistentes": attendees,
        "lluvia": int(raw["lluvia"]) if raw.get("lluvia") in (0, 1, False, True) else DEFAULT_RAIN,
        "trafico": traffic,
        "parser": "llm",
    }


def parse_plan(
    message: str,
    friends: list[str],
    plan_types: list[str],
    *,
    now: datetime | None = None,
) -> dict:
    if not message or not message.strip():
        raise ValueError("El mensaje no puede estar vacío.")
    current = now or datetime.now(ZoneInfo(TIMEZONE))
    today = current.date()
    if os.getenv("OPENAI_API_KEY"):
        try:
            raw = _llm_parse(message, friends, plan_types, today)
            return _validate_and_complete(raw, message, friends, plan_types, today)
        except Exception:
            # La demo debe seguir disponible ante errores de red, API o JSON.
            pass
    return _rule_parse(message, friends, plan_types, today)
