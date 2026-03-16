"""Click CLI for Aushadhi drug interaction analysis.

Commands
--------
aushadhi check --drugs "drug1,drug2,drug3"
    Quick interaction check with summary output.

aushadhi report --drugs "drug1,drug2,drug3" [--patient-id ID]
    Full polypharmacy report with interaction matrix and risk assessments.

aushadhi list [--class CLASS]
    List available drugs in the database.

aushadhi info DRUG_NAME
    Show detailed information about a single drug.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich import box

from aushadhi.drugs.database import DrugDatabase
from aushadhi.drugs.interactions import InteractionChecker
from aushadhi.models import Severity
from aushadhi.predictor import InteractionPredictor
from aushadhi.report import ReportGenerator
from aushadhi.risk_scorer import RiskScorer

console = Console()


def _parse_drug_list(drugs_str: str) -> list[str]:
    """Parse a comma-separated drug list string."""
    return [d.strip().lower() for d in drugs_str.split(",") if d.strip()]


@click.group()
@click.version_option(version="0.1.0", prog_name="aushadhi")
def cli() -> None:
    """Aushadhi -- AI Drug Interaction Predictor for Polypharmacy Patients."""
    pass


@cli.command()
@click.option(
    "--drugs", "-d",
    required=True,
    help='Comma-separated list of drug names, e.g. "warfarin,aspirin,omeprazole"',
)
@click.option("--no-llm", is_flag=True, default=False, help="Disable LLM augmentation")
def check(drugs: str, no_llm: bool) -> None:
    """Check for drug interactions between specified medications."""
    drug_names = _parse_drug_list(drugs)
    if len(drug_names) < 2:
        console.print("[red]Please provide at least 2 drugs to check interactions.[/red]")
        raise SystemExit(1)

    db = DrugDatabase()
    resolved = db.lookup_many(drug_names)
    not_found = set(drug_names) - {d.name for d in resolved}

    if not_found:
        console.print(f"[yellow]Not found in database: {', '.join(sorted(not_found))}[/yellow]")

    if len(resolved) < 2:
        console.print("[red]Need at least 2 recognized drugs for interaction check.[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold]Checking interactions for:[/bold] {', '.join(d.name for d in resolved)}")
    console.print()

    predictor = InteractionPredictor(db=db)
    med_list = predictor.predict(
        [d.name for d in resolved],
        use_llm=not no_llm,
    )

    if not med_list.interactions:
        console.print("[green]No interactions detected.[/green]")
        return

    scorer = RiskScorer()
    assessments = scorer.score_all(med_list.interactions)

    for assessment in assessments:
        ix = assessment.interaction
        sev = ix.severity.value.upper()
        color_map = {
            "MILD": "green",
            "MODERATE": "yellow",
            "SEVERE": "red",
            "CONTRAINDICATED": "bold red",
        }
        color = color_map.get(sev, "white")

        console.print(
            f"  [{color}]{sev:16s}[/{color}] "
            f"[bold]{' + '.join(ix.drugs)}[/bold] -- "
            f"{ix.mechanism.value.replace('_', ' ')} "
            f"(risk: {assessment.risk_score}/10)"
        )
        if ix.description:
            console.print(f"                   {ix.description[:120]}")
        console.print()

    overall = scorer.aggregate_risk(assessments)
    console.print(f"[bold]Overall risk:[/bold] {overall}/10")
    severe_count = sum(
        1 for a in assessments if a.severity in (Severity.SEVERE, Severity.CONTRAINDICATED)
    )
    if severe_count:
        console.print(
            f"[bold red]WARNING: {severe_count} severe/contraindicated interaction(s).[/bold red]"
        )


@cli.command()
@click.option(
    "--drugs", "-d",
    required=True,
    help='Comma-separated list of drug names',
)
@click.option("--patient-id", "-p", default=None, help="Patient identifier for the report")
@click.option("--no-llm", is_flag=True, default=False, help="Disable LLM augmentation")
def report(drugs: str, patient_id: str | None, no_llm: bool) -> None:
    """Generate a full polypharmacy interaction report."""
    drug_names = _parse_drug_list(drugs)
    if len(drug_names) < 2:
        console.print("[red]Please provide at least 2 drugs for a report.[/red]")
        raise SystemExit(1)

    db = DrugDatabase()
    resolved = db.lookup_many(drug_names)
    not_found = set(drug_names) - {d.name for d in resolved}

    if not_found:
        console.print(f"[yellow]Not found in database: {', '.join(sorted(not_found))}[/yellow]")

    if len(resolved) < 2:
        console.print("[red]Need at least 2 recognized drugs.[/red]")
        raise SystemExit(1)

    predictor = InteractionPredictor(db=db)
    med_list = predictor.predict(
        [d.name for d in resolved],
        use_llm=not no_llm,
        patient_id=patient_id,
    )

    reporter = ReportGenerator(console=console)
    reporter.generate(med_list)


@cli.command("list")
@click.option("--drug-class", "-c", default=None, help="Filter by drug class")
def list_drugs(drug_class: str | None) -> None:
    """List available drugs in the database."""
    db = DrugDatabase()

    if drug_class:
        from aushadhi.models import DrugClass

        try:
            cls = DrugClass(drug_class.lower())
        except ValueError:
            console.print(f"[red]Unknown drug class: {drug_class}[/red]")
            console.print("Available classes: " + ", ".join(c.value for c in DrugClass))
            raise SystemExit(1)
        drugs = db.by_class(cls)
        title = f"Drugs -- class: {cls.value}"
    else:
        drugs = db.all_drugs
        title = f"All Drugs ({db.count} total)"

    table = Table(title=title, box=box.ROUNDED, show_lines=False)
    table.add_column("Drug", style="cyan")
    table.add_column("Class")
    table.add_column("Brand Names", style="dim")

    for drug in sorted(drugs, key=lambda d: d.name):
        table.add_row(drug.name, drug.drug_class.value, ", ".join(drug.brand_names))

    console.print(table)


@cli.command()
@click.argument("drug_name")
def info(drug_name: str) -> None:
    """Show detailed information about a drug."""
    db = DrugDatabase()
    drug = db.lookup(drug_name)

    if drug is None:
        console.print(f"[red]Drug not found: {drug_name}[/red]")
        # Suggest alternatives
        matches = db.search(drug_name)
        if matches:
            console.print("Did you mean: " + ", ".join(m.name for m in matches[:5]))
        raise SystemExit(1)

    console.print(f"\n[bold cyan]{drug.name}[/bold cyan]")
    if drug.brand_names:
        console.print(f"  Brand names: {', '.join(drug.brand_names)}")
    console.print(f"  Class: {drug.drug_class.value}")
    console.print(f"  Mechanism: {drug.mechanism}")
    if drug.cyp450:
        cyp_str = "; ".join(f"{e.enzyme} ({e.role.value})" for e in drug.cyp450)
        console.print(f"  CYP450: {cyp_str}")
    if drug.half_life_hours is not None:
        console.print(f"  Half-life: {drug.half_life_hours} hours")
    if drug.protein_binding_pct is not None:
        console.print(f"  Protein binding: {drug.protein_binding_pct}%")
    if drug.narrow_therapeutic_index:
        console.print("  [red]Narrow therapeutic index[/red]")
    console.print()


if __name__ == "__main__":
    cli()
