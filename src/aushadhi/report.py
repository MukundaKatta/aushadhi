"""Rich terminal report generation for polypharmacy analysis.

Generates formatted reports with:
  - Medication list table
  - Interaction matrix
  - Risk assessment details
  - Overall polypharmacy risk summary
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from aushadhi.models import (
    Interaction,
    MedicationList,
    RiskAssessment,
    Severity,
)


# ---------------------------------------------------------------------------
# Severity → color mapping for rich output
# ---------------------------------------------------------------------------

_SEVERITY_COLORS: dict[Severity, str] = {
    Severity.MILD: "green",
    Severity.MODERATE: "yellow",
    Severity.SEVERE: "red",
    Severity.CONTRAINDICATED: "bold red on white",
}

_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.MILD: "[green]LOW[/green]",
    Severity.MODERATE: "[yellow]MED[/yellow]",
    Severity.SEVERE: "[red]HIGH[/red]",
    Severity.CONTRAINDICATED: "[bold red]CRIT[/bold red]",
}


def _risk_bar(score: float) -> str:
    """Create a visual risk bar for a 0-10 score."""
    filled = int(score)
    empty = 10 - filled
    if score >= 8:
        color = "red"
    elif score >= 5:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {score}/10"


class ReportGenerator:
    """Generates rich terminal reports from MedicationList analysis results.

    Parameters
    ----------
    console : Console, optional
        Rich Console instance.  Creates a new one if not provided.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def generate(self, med_list: MedicationList) -> None:
        """Generate and print a full polypharmacy analysis report."""
        self._print_header(med_list)
        self._print_medication_table(med_list)
        self._print_interaction_matrix(med_list)
        self._print_risk_assessments(med_list)
        self._print_summary(med_list)

    def _print_header(self, med_list: MedicationList) -> None:
        """Print report header."""
        title = "AUSHADHI -- Drug Interaction Report"
        subtitle_parts: list[str] = []
        if med_list.patient_id:
            subtitle_parts.append(f"Patient: {med_list.patient_id}")
        subtitle_parts.append(f"Medications: {med_list.drug_count}")
        if med_list.is_polypharmacy:
            subtitle_parts.append("[yellow]POLYPHARMACY[/yellow]")
        subtitle = " | ".join(subtitle_parts)

        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]{title}[/bold cyan]\n{subtitle}",
                box=box.DOUBLE,
                expand=False,
                padding=(1, 4),
            )
        )
        self.console.print()

    def _print_medication_table(self, med_list: MedicationList) -> None:
        """Print medication list as a table."""
        table = Table(
            title="Medication List",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Drug", style="bold cyan")
        table.add_column("Class")
        table.add_column("CYP450 Profile", max_width=40)
        table.add_column("NTI", justify="center", width=4)

        for i, drug in enumerate(med_list.drugs, 1):
            cyp_parts: list[str] = []
            for entry in drug.cyp450:
                role_color = {"substrate": "white", "inhibitor": "red", "inducer": "yellow"}
                color = role_color.get(entry.role.value, "white")
                cyp_parts.append(f"[{color}]{entry.enzyme}({entry.role.value[0].upper()})[/{color}]")
            cyp_str = ", ".join(cyp_parts) if cyp_parts else "[dim]none[/dim]"
            nti = "[red]YES[/red]" if drug.narrow_therapeutic_index else "[dim]no[/dim]"

            table.add_row(str(i), drug.name, drug.drug_class.value, cyp_str, nti)

        self.console.print(table)
        self.console.print()

    def _print_interaction_matrix(self, med_list: MedicationList) -> None:
        """Print a pairwise interaction matrix."""
        if not med_list.interactions:
            self.console.print("[green]No interactions detected.[/green]")
            return

        drugs = med_list.drugs
        if len(drugs) > 15:
            # Matrix gets unwieldy for large drug lists; skip it
            self.console.print(
                f"[dim]Interaction matrix omitted ({len(drugs)} drugs -- see detailed list below)[/dim]"
            )
            self.console.print()
            return

        # Build severity lookup for pairwise interactions
        pair_severity: dict[tuple[str, str], Severity] = {}
        for ix in med_list.interactions:
            if len(ix.drugs) == 2:
                key = tuple(sorted(ix.drugs))
                existing = pair_severity.get(key)  # type: ignore[arg-type]
                if existing is None or _severity_rank(ix.severity) > _severity_rank(existing):
                    pair_severity[key] = ix.severity  # type: ignore[index]

        table = Table(
            title="Interaction Matrix",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            title_style="bold",
        )
        table.add_column("", style="bold cyan", width=16)
        for drug in drugs:
            table.add_column(drug.name[:8], justify="center", width=9)

        for i, row_drug in enumerate(drugs):
            cells: list[str] = []
            for j, col_drug in enumerate(drugs):
                if i == j:
                    cells.append("[dim]---[/dim]")
                else:
                    key = tuple(sorted([row_drug.name, col_drug.name]))
                    sev = pair_severity.get(key)  # type: ignore[arg-type]
                    if sev:
                        cells.append(_SEVERITY_EMOJI[sev])
                    else:
                        cells.append("[dim].[/dim]")
            table.add_row(row_drug.name[:16], *cells)

        self.console.print(table)
        self.console.print()

        # Legend
        self.console.print(
            "[dim]Matrix legend: "
            "[green]LOW[/green]=mild  "
            "[yellow]MED[/yellow]=moderate  "
            "[red]HIGH[/red]=severe  "
            "[bold red]CRIT[/bold red]=contraindicated  "
            ".=none[/dim]"
        )
        self.console.print()

    def _print_risk_assessments(self, med_list: MedicationList) -> None:
        """Print detailed risk assessments."""
        if not med_list.risk_assessments:
            return

        self.console.print("[bold]Detailed Interaction Analysis[/bold]")
        self.console.print()

        for i, assessment in enumerate(med_list.risk_assessments, 1):
            ix = assessment.interaction
            sev_color = _SEVERITY_COLORS[assessment.severity]

            header = (
                f"[{sev_color}]{assessment.severity.value.upper()}[/{sev_color}] "
                f"[bold]{' + '.join(ix.drugs)}[/bold]"
            )

            content_lines: list[str] = [
                f"[bold]Mechanism:[/bold] {ix.mechanism.value.replace('_', ' ').title()}",
                f"[bold]Evidence:[/bold] {ix.evidence.value}",
                f"[bold]Risk Score:[/bold] {_risk_bar(assessment.risk_score)}",
            ]

            if ix.description:
                content_lines.append(f"\n[bold]Description:[/bold]\n{ix.description}")
            if ix.clinical_significance:
                content_lines.append(
                    f"\n[bold]Clinical Significance:[/bold]\n{ix.clinical_significance}"
                )

            if assessment.recommendations:
                content_lines.append("\n[bold]Recommendations:[/bold]")
                for rec in assessment.recommendations:
                    content_lines.append(f"  • {rec}")

            if assessment.monitoring_parameters:
                content_lines.append("\n[bold]Monitoring:[/bold]")
                for param in assessment.monitoring_parameters:
                    content_lines.append(f"  • {param}")

            self.console.print(
                Panel(
                    "\n".join(content_lines),
                    title=f"#{i} {header}",
                    border_style=sev_color.split()[-1] if " " in sev_color else sev_color,
                    box=box.ROUNDED,
                    expand=True,
                )
            )
            self.console.print()

    def _print_summary(self, med_list: MedicationList) -> None:
        """Print overall risk summary."""
        total_interactions = len(med_list.interactions)
        severe_count = len(med_list.severe_interactions)
        overall = med_list.overall_risk_score or 0.0

        # Severity breakdown
        breakdown: dict[str, int] = {}
        for ix in med_list.interactions:
            sev = ix.severity.value
            breakdown[sev] = breakdown.get(sev, 0) + 1

        summary_lines: list[str] = [
            f"[bold]Total Interactions:[/bold] {total_interactions}",
        ]

        if breakdown:
            parts = []
            for sev in ["contraindicated", "severe", "moderate", "mild"]:
                count = breakdown.get(sev, 0)
                if count > 0:
                    color = _SEVERITY_COLORS[Severity(sev)]
                    parts.append(f"[{color}]{sev}: {count}[/{color}]")
            summary_lines.append("[bold]Breakdown:[/bold] " + " | ".join(parts))

        summary_lines.append(f"\n[bold]Overall Polypharmacy Risk:[/bold] {_risk_bar(overall)}")

        if severe_count > 0:
            summary_lines.append(
                f"\n[bold red]WARNING: {severe_count} severe/contraindicated interaction(s) detected. "
                f"Clinical review strongly recommended.[/bold red]"
            )
        elif total_interactions > 0:
            summary_lines.append(
                "\n[yellow]Some interactions detected. Review recommendations above.[/yellow]"
            )
        else:
            summary_lines.append(
                "\n[green]No significant interactions detected in this medication list.[/green]"
            )

        self.console.print(
            Panel(
                "\n".join(summary_lines),
                title="[bold]Summary[/bold]",
                box=box.DOUBLE,
                border_style="cyan",
                expand=True,
            )
        )
        self.console.print()


def _severity_rank(severity: Severity) -> int:
    """Numeric rank for severity comparison."""
    return {
        Severity.MILD: 1,
        Severity.MODERATE: 2,
        Severity.SEVERE: 3,
        Severity.CONTRAINDICATED: 4,
    }[severity]
