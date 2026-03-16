"""LLM-augmented interaction predictor with RAG context.

InteractionPredictor wraps the rule-based InteractionChecker with an optional
LLM reasoning layer.  When an OpenAI-compatible API key is available, it sends
the RAG-retrieved pharmacological context along with the drug list to an LLM
for deeper analysis of novel or under-documented combinations.  When no API key
is configured, it falls back to the deterministic rule-based engine.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from aushadhi.drugs.database import DrugDatabase
from aushadhi.drugs.interactions import InteractionChecker
from aushadhi.models import (
    Drug,
    EvidenceLevel,
    Interaction,
    InteractionMechanism,
    MedicationList,
    Severity,
)
from aushadhi.rag import RAGPipeline
from aushadhi.risk_scorer import RiskScorer


# ---------------------------------------------------------------------------
# System prompt for LLM-augmented prediction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Aushadhi, a clinical pharmacology AI assistant specializing in drug
interaction analysis for polypharmacy patients.  You will be given a list of
medications and supporting pharmacological context retrieved from a knowledge base.

Your task is to identify potential drug interactions that may NOT be captured by
standard pairwise interaction databases.  Focus on:

1. Higher-order interactions where 3+ drugs combine to create emergent risks.
2. Pharmacokinetic cascades (e.g., Drug A inhibits CYP3A4, raising Drug B levels,
   which then competes with Drug C for renal excretion).
3. Pharmacodynamic summation effects across the full medication list.
4. Patient-population-specific risks (elderly, renal impairment, hepatic impairment).

Return your analysis as a JSON array of interaction objects with this schema:
{
  "drugs": ["drug_a", "drug_b", ...],
  "severity": "mild|moderate|severe|contraindicated",
  "mechanism": "cyp450_inhibition|cyp450_induction|pharmacodynamic_synergy|pharmacodynamic_antagonism|protein_binding_displacement|renal_competition|gi_absorption|qt_prolongation|serotonin_syndrome|bleeding_risk|cns_depression|hyperkalemia|nephrotoxicity|hepatotoxicity|other",
  "description": "...",
  "clinical_significance": "...",
  "management": "..."
}

Only report interactions not already identified in the rule-based analysis.
Be conservative -- only flag interactions with reasonable clinical plausibility.
Return an empty array [] if no additional interactions are identified.
"""


class InteractionPredictor:
    """LLM-augmented drug interaction predictor.

    Combines rule-based detection with optional LLM reasoning for comprehensive
    polypharmacy analysis.

    Parameters
    ----------
    db : DrugDatabase, optional
        Drug database instance.  Creates a new one if not provided.
    api_key : str, optional
        OpenAI API key.  Falls back to OPENAI_API_KEY env var.
    model : str
        LLM model to use for augmented prediction.
    """

    def __init__(
        self,
        db: Optional[DrugDatabase] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> None:
        self.db = db or DrugDatabase()
        self.checker = InteractionChecker(self.db)
        self.rag = RAGPipeline(self.db)
        self.scorer = RiskScorer()
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def llm_available(self) -> bool:
        """Whether an LLM API key is configured."""
        return bool(self._api_key)

    def predict(
        self,
        drug_names: list[str],
        use_llm: bool = True,
        patient_id: Optional[str] = None,
    ) -> MedicationList:
        """Run full interaction prediction pipeline.

        Parameters
        ----------
        drug_names : list[str]
            Generic drug names or brand names.
        use_llm : bool
            Whether to augment with LLM reasoning (requires API key).
        patient_id : str, optional
            Patient identifier for the report.

        Returns
        -------
        MedicationList
            Complete medication list with interactions and risk assessments.
        """
        # Resolve drugs
        drugs = self.db.lookup_many(drug_names)
        if len(drugs) < 2:
            return MedicationList(
                patient_id=patient_id,
                drugs=drugs,
                interactions=[],
                risk_assessments=[],
                overall_risk_score=0.0,
            )

        # Rule-based interaction detection
        interactions = self.checker.check_all(drugs)

        # LLM augmentation
        if use_llm and self.llm_available:
            llm_interactions = self._llm_predict(drugs, interactions)
            interactions.extend(llm_interactions)

        # Score all interactions
        assessments = self.scorer.score_all(interactions)
        overall = self.scorer.aggregate_risk(assessments)

        return MedicationList(
            patient_id=patient_id,
            drugs=drugs,
            interactions=interactions,
            risk_assessments=assessments,
            overall_risk_score=overall,
        )

    def _llm_predict(
        self, drugs: list[Drug], existing: list[Interaction]
    ) -> list[Interaction]:
        """Call LLM with RAG context for augmented prediction."""
        try:
            from openai import OpenAI
        except ImportError:
            return []

        if not self._api_key:
            return []

        # Retrieve context
        drug_names = [d.name for d in drugs]
        rag_result = self.rag.retrieve(drug_names)
        context = rag_result.to_prompt_context()

        # Format existing interactions for the LLM
        existing_summary = ""
        if existing:
            lines = []
            for ix in existing:
                lines.append(f"- {' + '.join(ix.drugs)}: {ix.severity.value} ({ix.mechanism.value})")
            existing_summary = (
                "\n## Already-Identified Interactions (do not repeat these)\n"
                + "\n".join(lines)
            )

        user_message = (
            f"## Medication List\n{', '.join(drug_names)}\n\n"
            f"{context}\n\n"
            f"{existing_summary}\n\n"
            f"Analyze the above medication list for additional interactions not yet identified."
        )

        try:
            client = OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "[]"
            return self._parse_llm_response(content)
        except Exception:
            # Gracefully fall back to rule-based only
            return []

    def _parse_llm_response(self, raw: str) -> list[Interaction]:
        """Parse LLM JSON response into Interaction models."""
        try:
            data = json.loads(raw)
            # Handle both {"interactions": [...]} and bare [...]
            if isinstance(data, dict):
                items = data.get("interactions", data.get("results", []))
            elif isinstance(data, list):
                items = data
            else:
                return []

            interactions: list[Interaction] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    severity = Severity(item.get("severity", "moderate"))
                    mechanism_str = item.get("mechanism", "other")
                    try:
                        mechanism = InteractionMechanism(mechanism_str)
                    except ValueError:
                        mechanism = InteractionMechanism.OTHER

                    interactions.append(
                        Interaction(
                            drugs=item.get("drugs", []),
                            severity=severity,
                            mechanism=mechanism,
                            evidence=EvidenceLevel.THEORETICAL,
                            description=item.get("description", ""),
                            clinical_significance=item.get("clinical_significance", ""),
                            management=item.get("management", ""),
                        )
                    )
                except (ValueError, KeyError):
                    continue

            return interactions
        except (json.JSONDecodeError, TypeError):
            return []
