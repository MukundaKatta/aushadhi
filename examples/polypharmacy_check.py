#!/usr/bin/env python3
"""Example: Polypharmacy interaction analysis for a complex patient.

Demonstrates how to use Aushadhi programmatically to analyze a patient
taking multiple medications -- a common scenario in geriatric medicine.
"""

from aushadhi.drugs.database import DrugDatabase
from aushadhi.drugs.interactions import InteractionChecker
from aushadhi.predictor import InteractionPredictor
from aushadhi.report import ReportGenerator
from aushadhi.risk_scorer import RiskScorer
from rich.console import Console


def main() -> None:
    console = Console()

    # Typical elderly patient medication list
    patient_drugs = [
        "warfarin",         # Anticoagulant for atrial fibrillation
        "metoprolol",       # Beta-blocker for rate control
        "lisinopril",       # ACE inhibitor for hypertension
        "atorvastatin",     # Statin for hyperlipidemia
        "metformin",        # Biguanide for type 2 diabetes
        "omeprazole",       # PPI for GERD
        "amlodipine",       # CCB for hypertension
        "sertraline",       # SSRI for depression
        "gabapentin",       # For neuropathic pain
        "levothyroxine",    # Thyroid replacement
    ]

    console.print("[bold cyan]Aushadhi Polypharmacy Analysis Example[/bold cyan]")
    console.print(f"Analyzing {len(patient_drugs)} medications...\n")

    # --- Approach 1: Quick programmatic check ---
    console.print("[bold]== Quick Interaction Check ==[/bold]\n")

    db = DrugDatabase()
    checker = InteractionChecker(db)
    scorer = RiskScorer()

    drugs = db.lookup_many(patient_drugs)
    interactions = checker.check_all(drugs)
    assessments = scorer.score_all(interactions)

    for assessment in assessments:
        ix = assessment.interaction
        console.print(
            f"  [{assessment.severity.value.upper():16s}] "
            f"{' + '.join(ix.drugs)} "
            f"-- {ix.mechanism.value.replace('_', ' ')} "
            f"(risk: {assessment.risk_score}/10)"
        )

    overall = scorer.aggregate_risk(assessments)
    console.print(f"\n  Overall risk: {overall}/10")
    console.print(f"  Total interactions found: {len(interactions)}")
    console.print(f"  Severe/contraindicated: {sum(1 for a in assessments if a.severity.value in ('severe', 'contraindicated'))}")

    # --- Approach 2: Full report via predictor + report generator ---
    console.print("\n\n[bold]== Full Polypharmacy Report ==[/bold]\n")

    predictor = InteractionPredictor(db=db)
    med_list = predictor.predict(
        patient_drugs,
        use_llm=False,  # Set to True if OPENAI_API_KEY is configured
        patient_id="EXAMPLE-001",
    )

    reporter = ReportGenerator(console=console)
    reporter.generate(med_list)

    # --- Approach 3: Investigating a specific pair ---
    console.print("[bold]== Specific Pair Investigation ==[/bold]\n")

    warfarin = db.lookup("warfarin")
    aspirin = db.lookup("aspirin")
    if warfarin and aspirin:
        pair_interactions = checker.check_pair(warfarin, aspirin)
        for ix in pair_interactions:
            assessment = scorer.score(ix)
            console.print(f"  {assessment.summary}")
            console.print(f"  Description: {ix.description}")
            console.print(f"  Management: {ix.management}")
            console.print()


if __name__ == "__main__":
    main()
