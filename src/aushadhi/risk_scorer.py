"""Risk scoring engine for drug interactions.

Assigns numeric risk scores (0-10) and generates clinical recommendations
based on interaction severity, mechanism, evidence level, and drug properties.
"""

from __future__ import annotations

from aushadhi.models import (
    EvidenceLevel,
    Interaction,
    InteractionMechanism,
    RiskAssessment,
    Severity,
)


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

_SEVERITY_SCORE: dict[Severity, float] = {
    Severity.MILD: 2.0,
    Severity.MODERATE: 5.0,
    Severity.SEVERE: 8.0,
    Severity.CONTRAINDICATED: 10.0,
}

_EVIDENCE_MULTIPLIER: dict[EvidenceLevel, float] = {
    EvidenceLevel.ESTABLISHED: 1.0,
    EvidenceLevel.PROBABLE: 0.9,
    EvidenceLevel.SUSPECTED: 0.75,
    EvidenceLevel.THEORETICAL: 0.5,
}

_MECHANISM_BONUS: dict[InteractionMechanism, float] = {
    InteractionMechanism.QT_PROLONGATION: 1.0,
    InteractionMechanism.SEROTONIN_SYNDROME: 0.8,
    InteractionMechanism.CNS_DEPRESSION: 0.5,
    InteractionMechanism.BLEEDING_RISK: 0.7,
    InteractionMechanism.NEPHROTOXICITY: 0.6,
    InteractionMechanism.HEPATOTOXICITY: 0.6,
    InteractionMechanism.HYPERKALEMIA: 0.5,
}

# ---------------------------------------------------------------------------
# Monitoring parameter templates by mechanism
# ---------------------------------------------------------------------------

_MONITORING: dict[InteractionMechanism, list[str]] = {
    InteractionMechanism.CYP450_INHIBITION: [
        "Therapeutic drug levels (if available)",
        "Signs/symptoms of toxicity for the affected drug",
    ],
    InteractionMechanism.CYP450_INDUCTION: [
        "Therapeutic response / drug levels",
        "May need dose increase during co-therapy and decrease after discontinuation",
    ],
    InteractionMechanism.BLEEDING_RISK: [
        "INR (if on warfarin)",
        "CBC / hemoglobin",
        "Signs of bleeding (bruising, melena, hematuria)",
    ],
    InteractionMechanism.QT_PROLONGATION: [
        "Baseline and serial ECG (QTc interval)",
        "Serum potassium and magnesium",
        "Symptoms: palpitations, syncope, dizziness",
    ],
    InteractionMechanism.SEROTONIN_SYNDROME: [
        "Mental status changes (agitation, confusion)",
        "Neuromuscular signs (clonus, hyperreflexia, tremor)",
        "Autonomic instability (tachycardia, diaphoresis, hyperthermia)",
    ],
    InteractionMechanism.CNS_DEPRESSION: [
        "Level of consciousness / sedation scale",
        "Respiratory rate and oxygen saturation",
        "Fall risk assessment",
    ],
    InteractionMechanism.NEPHROTOXICITY: [
        "Serum creatinine and BUN",
        "Urine output",
        "Electrolytes (potassium, sodium)",
    ],
    InteractionMechanism.HYPERKALEMIA: [
        "Serum potassium (within 1 week, then periodically)",
        "ECG if potassium >5.5 mEq/L",
        "Renal function (creatinine, GFR)",
    ],
    InteractionMechanism.HEPATOTOXICITY: [
        "Liver function tests (AST, ALT, bilirubin)",
        "Symptoms: jaundice, dark urine, abdominal pain",
    ],
    InteractionMechanism.GI_ABSORPTION: [
        "Therapeutic response / drug levels",
        "Administer drugs at separate times as recommended",
    ],
    InteractionMechanism.RENAL_COMPETITION: [
        "Drug levels for narrow therapeutic index agents",
        "Renal function (creatinine, GFR)",
    ],
    InteractionMechanism.PHARMACODYNAMIC_SYNERGY: [
        "Clinical effect monitoring specific to drug pair",
        "Vital signs as appropriate",
    ],
    InteractionMechanism.PHARMACODYNAMIC_ANTAGONISM: [
        "Therapeutic response of both agents",
    ],
    InteractionMechanism.PROTEIN_BINDING_DISPLACEMENT: [
        "Free drug levels if available",
        "Signs of toxicity for highly bound drug",
    ],
    InteractionMechanism.OTHER: [
        "Clinical monitoring as appropriate for the specific interaction",
    ],
}


class RiskScorer:
    """Scores drug interactions and generates clinical risk assessments."""

    def score(self, interaction: Interaction) -> RiskAssessment:
        """Compute a risk assessment for a single interaction."""
        base = _SEVERITY_SCORE[interaction.severity]
        multiplier = _EVIDENCE_MULTIPLIER[interaction.evidence]
        bonus = _MECHANISM_BONUS.get(interaction.mechanism, 0.0)

        # Higher-order interactions (3+ drugs) get a complexity bonus
        if len(interaction.drugs) > 2:
            bonus += 0.5 * (len(interaction.drugs) - 2)

        raw_score = base * multiplier + bonus
        risk_score = min(10.0, round(raw_score, 1))

        summary = self._build_summary(interaction, risk_score)
        recommendations = self._build_recommendations(interaction)
        monitoring = _MONITORING.get(interaction.mechanism, [])

        return RiskAssessment(
            interaction=interaction,
            severity=interaction.severity,
            risk_score=risk_score,
            summary=summary,
            recommendations=recommendations,
            monitoring_parameters=list(monitoring),
        )

    def score_all(self, interactions: list[Interaction]) -> list[RiskAssessment]:
        """Score a list of interactions and return sorted by risk (highest first)."""
        assessments = [self.score(ix) for ix in interactions]
        assessments.sort(key=lambda a: a.risk_score, reverse=True)
        return assessments

    def aggregate_risk(self, assessments: list[RiskAssessment]) -> float:
        """Compute an aggregate polypharmacy risk score (0-10).

        Uses a weighted combination: the highest individual score dominates,
        with diminishing contributions from additional interactions.
        """
        if not assessments:
            return 0.0

        scores = sorted([a.risk_score for a in assessments], reverse=True)
        total = scores[0]
        for i, s in enumerate(scores[1:], start=1):
            # Each subsequent interaction adds a fraction of its score
            total += s * (0.5 ** i)

        return min(10.0, round(total, 1))

    def _build_summary(self, interaction: Interaction, risk_score: float) -> str:
        """Build a human-readable summary line."""
        drug_str = " + ".join(interaction.drugs)
        sev = interaction.severity.value.upper()
        return (
            f"[{sev}] {drug_str} -- {interaction.mechanism.value.replace('_', ' ')} "
            f"(risk {risk_score}/10, evidence: {interaction.evidence.value})"
        )

    def _build_recommendations(self, interaction: Interaction) -> list[str]:
        """Generate prioritized recommendations."""
        recs: list[str] = []

        if interaction.severity == Severity.CONTRAINDICATED:
            recs.append("AVOID this combination. Seek therapeutic alternatives.")
        elif interaction.severity == Severity.SEVERE:
            recs.append("Use only if benefits clearly outweigh risks; close monitoring required.")
        elif interaction.severity == Severity.MODERATE:
            recs.append("Use with caution; adjust doses and monitor as recommended.")
        else:
            recs.append("Generally manageable; be aware of the interaction.")

        if interaction.management:
            recs.append(interaction.management)

        return recs
