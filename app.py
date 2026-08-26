from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import MODEL_PATH
from src.parser import parse_plan
from src.predictor import load_artifact, predict_plan
from src.training import train_and_save

console = Console()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def run(message: str, *, debug: bool = False) -> list[dict]:
    if not MODEL_PATH.exists():
        console.print("[yellow]No hay modelo entrenado; lo crearé ahora.[/yellow]")
        train_and_save(verbose=True)
    artifact = load_artifact()
    plan = parse_plan(message, artifact["friends"], artifact["plan_types"])
    ranking, observations, raw_predictions = predict_plan(plan, artifact)
    display_date = date.fromisoformat(plan["fecha"]).strftime("%d/%m/%Y")

    console.print(
        Panel.fit(
            f"[bold]PLAN INTERPRETADO[/bold]\n\n"
            f"[cyan]Tipo:[/cyan] {plan['tipo_plan']}\n"
            f"[cyan]Fecha:[/cyan] {display_date} ({plan['dia_semana']})\n"
            f"[cyan]Hora:[/cyan] {plan['hora']}\n"
            f"[cyan]Lugar:[/cyan] {plan['lugar']}\n"
            f"[cyan]Asistentes:[/cyan] {', '.join(plan['asistentes'])}",
            border_style="cyan",
        )
    )

    table = Table(title="🔮 PREDICCIÓN DE LLEGADA", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("#", justify="right")
    table.add_column("Amigo", style="bold")
    table.add_column("Llegada", justify="center")
    table.add_column("Retraso", justify="right")
    for position, row in enumerate(ranking, 1):
        delay = row["retraso_min"]
        delay_text = f"{delay:+d} min"
        table.add_row(str(position), row["amigo"], row["llegada"], delay_text)
    console.print(table)
    console.print(f"🏆 [green]{ranking[0]['amigo']}[/green] debería ser el primero.")
    if len(ranking) > 1:
        console.print(f"🐢 [yellow]{ranking[-1]['amigo']}[/yellow] debería ser el último.")

    if debug:
        console.rule("DEBUG")
        console.print("[bold]Mensaje[/bold]")
        console.print(repr(message))
        console.print("\n[bold]JSON interpretado[/bold]")
        console.print(_json(plan))
        console.print("\n[bold]Observaciones generadas[/bold]")
        console.print(observations.to_string(index=False))
        console.print("\n[bold]Predicciones crudas[/bold]")
        console.print(_json(raw_predictions))
        console.print(f"\nModelo: {artifact['model_name']}")
    return ranking


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predice el orden de llegada de tus amigos.")
    parser.add_argument("--debug", action="store_true", help="muestra datos internos")
    parser.add_argument("--message", help="mensaje directo, sin prompt interactivo")
    args = parser.parse_args()
    console.print(Panel.fit("🔮 [bold magenta]PREDICTOR DE PUNTUALIDAD[/bold magenta]", border_style="magenta"))
    message = args.message
    if message is None:
        console.print("\n[bold]¿Qué plan tienes?[/bold]")
        message = console.input("[cyan]> [/cyan]")
    try:
        run(message, debug=args.debug)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
