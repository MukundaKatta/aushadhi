"""RAG (Retrieval-Augmented Generation) pipeline for drug interaction knowledge.

Provides contextual retrieval from the built-in drug knowledge base to augment
LLM predictions with relevant pharmacological information.  In production this
would connect to a vector store; here we use a lightweight in-memory approach
based on keyword matching and structured drug data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aushadhi.drugs.classes import (
    CNS_DEPRESSANTS,
    DRUG_CLASS_INFO,
    POTENT_CYP_INHIBITORS,
    POTENT_CYP_INDUCERS,
    QT_PROLONGING_DRUGS,
    SEROTONERGIC_DRUGS,
)
from aushadhi.drugs.database import DrugDatabase
from aushadhi.models import CYP450Role, Drug


@dataclass
class RetrievedContext:
    """A chunk of retrieved context for RAG augmentation."""

    source: str
    content: str
    relevance_score: float = 0.0


@dataclass
class RAGResult:
    """Aggregated RAG retrieval result for a set of drugs."""

    drug_profiles: list[str] = field(default_factory=list)
    cyp_context: list[str] = field(default_factory=list)
    class_context: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    interaction_hints: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format all retrieved context into a single string for LLM prompting."""
        sections: list[str] = []

        if self.drug_profiles:
            sections.append("## Drug Profiles\n" + "\n".join(self.drug_profiles))
        if self.cyp_context:
            sections.append("## CYP450 Metabolism Context\n" + "\n".join(self.cyp_context))
        if self.class_context:
            sections.append("## Drug Class Information\n" + "\n".join(self.class_context))
        if self.risk_flags:
            sections.append("## Risk Flags\n" + "\n".join(self.risk_flags))
        if self.interaction_hints:
            sections.append("## Known Interaction Patterns\n" + "\n".join(self.interaction_hints))

        return "\n\n".join(sections)


class RAGPipeline:
    """Retrieval pipeline that gathers pharmacological context for LLM augmentation.

    This pipeline retrieves:
      - Drug profiles (mechanism, class, CYP450 interactions)
      - CYP450 context (inhibitors/inducers for relevant enzymes)
      - Drug class information
      - Special risk flags (serotonergic, QT-prolonging, CNS depressant)
      - Interaction pattern hints

    Parameters
    ----------
    db : DrugDatabase
        Drug database for looking up drug information.
    """

    def __init__(self, db: DrugDatabase) -> None:
        self.db = db

    def retrieve(self, drug_names: list[str]) -> RAGResult:
        """Retrieve all relevant context for the given drug list."""
        drugs = self.db.lookup_many(drug_names)
        result = RAGResult()

        for drug in drugs:
            result.drug_profiles.append(self._format_drug_profile(drug))

        result.cyp_context = self._gather_cyp_context(drugs)
        result.class_context = self._gather_class_context(drugs)
        result.risk_flags = self._gather_risk_flags(drugs)
        result.interaction_hints = self._gather_interaction_hints(drugs)

        return result

    def _format_drug_profile(self, drug: Drug) -> str:
        """Format a single drug's profile as a context string."""
        lines = [
            f"- **{drug.name}** ({', '.join(drug.brand_names) or 'no brand names'})",
            f"  Class: {drug.drug_class.value}",
            f"  Mechanism: {drug.mechanism}",
        ]
        if drug.cyp450:
            cyp_str = "; ".join(f"{e.enzyme} ({e.role.value})" for e in drug.cyp450)
            lines.append(f"  CYP450: {cyp_str}")
        if drug.half_life_hours is not None:
            lines.append(f"  Half-life: {drug.half_life_hours} hours")
        if drug.protein_binding_pct is not None:
            lines.append(f"  Protein binding: {drug.protein_binding_pct}%")
        if drug.narrow_therapeutic_index:
            lines.append("  **Narrow therapeutic index**")
        return "\n".join(lines)

    def _gather_cyp_context(self, drugs: list[Drug]) -> list[str]:
        """Gather CYP450 context for enzymes relevant to the drug set."""
        relevant_enzymes: set[str] = set()
        for drug in drugs:
            for entry in drug.cyp450:
                relevant_enzymes.add(entry.enzyme)

        context: list[str] = []
        for enzyme in sorted(relevant_enzymes):
            substrates = [
                d.name for d in drugs if any(e.enzyme == enzyme and e.role == CYP450Role.SUBSTRATE for e in d.cyp450)
            ]
            inhibitors_in_set = [
                d.name for d in drugs if any(e.enzyme == enzyme and e.role == CYP450Role.INHIBITOR for e in d.cyp450)
            ]
            inducers_in_set = [
                d.name for d in drugs if any(e.enzyme == enzyme and e.role == CYP450Role.INDUCER for e in d.cyp450)
            ]

            if (substrates and inhibitors_in_set) or (substrates and inducers_in_set):
                parts = [f"- {enzyme}:"]
                if substrates:
                    parts.append(f"  Substrates in list: {', '.join(substrates)}")
                if inhibitors_in_set:
                    parts.append(f"  Inhibitors in list: {', '.join(inhibitors_in_set)}")
                if inducers_in_set:
                    parts.append(f"  Inducers in list: {', '.join(inducers_in_set)}")

                known_inhibitors = POTENT_CYP_INHIBITORS.get(enzyme, [])
                known_inducers = POTENT_CYP_INDUCERS.get(enzyme, [])
                if known_inhibitors:
                    parts.append(f"  Known potent inhibitors: {', '.join(known_inhibitors)}")
                if known_inducers:
                    parts.append(f"  Known potent inducers: {', '.join(known_inducers)}")

                context.append("\n".join(parts))

        return context

    def _gather_class_context(self, drugs: list[Drug]) -> list[str]:
        """Gather drug class information for classes represented in the drug set."""
        seen_classes: set[str] = set()
        context: list[str] = []

        for drug in drugs:
            cls = drug.drug_class.value
            if cls not in seen_classes:
                seen_classes.add(cls)
                info = DRUG_CLASS_INFO.get(cls, {})
                if info:
                    context.append(
                        f"- {cls}: {info.get('description', '')} -- "
                        f"Typical mechanism: {info.get('typical_mechanism', 'various')}"
                    )

        return context

    def _gather_risk_flags(self, drugs: list[Drug]) -> list[str]:
        """Identify special risk categories for the drug set."""
        flags: list[str] = []
        names = {d.name for d in drugs}

        sero = names & SEROTONERGIC_DRUGS
        if len(sero) >= 2:
            flags.append(
                f"SEROTONIN SYNDROME RISK: {len(sero)} serotonergic agents present "
                f"({', '.join(sorted(sero))})"
            )

        qt = names & QT_PROLONGING_DRUGS
        if len(qt) >= 2:
            flags.append(
                f"QT PROLONGATION RISK: {len(qt)} QT-prolonging agents present "
                f"({', '.join(sorted(qt))})"
            )

        cns = names & CNS_DEPRESSANTS
        if len(cns) >= 2:
            flags.append(
                f"CNS DEPRESSION RISK: {len(cns)} CNS depressant agents present "
                f"({', '.join(sorted(cns))})"
            )

        nti = [d.name for d in drugs if d.narrow_therapeutic_index]
        if nti:
            flags.append(
                f"NARROW THERAPEUTIC INDEX: {', '.join(nti)} -- "
                f"small changes in drug levels may cause toxicity or treatment failure"
            )

        high_binding = [d.name for d in drugs if (d.protein_binding_pct or 0) > 95]
        if len(high_binding) >= 2:
            flags.append(
                f"PROTEIN BINDING DISPLACEMENT RISK: Multiple highly protein-bound drugs "
                f"({', '.join(high_binding)}) -- displacement may increase free drug levels"
            )

        return flags

    def _gather_interaction_hints(self, drugs: list[Drug]) -> list[str]:
        """Provide hints about likely interaction patterns."""
        hints: list[str] = []
        classes = {d.drug_class for d in drugs}

        from aushadhi.models import DrugClass

        # Triple whammy hint
        raas = classes & {DrugClass.ACE_INHIBITOR, DrugClass.ARB}
        if raas and DrugClass.DIURETIC in classes and DrugClass.NSAID in classes:
            hints.append(
                "TRIPLE WHAMMY: ACE inhibitor/ARB + diuretic + NSAID combination "
                "identified -- high risk of acute kidney injury"
            )

        # Dual antiplatelet
        antiplatelets = [d for d in drugs if d.drug_class == DrugClass.ANTIPLATELET]
        if len(antiplatelets) >= 2:
            hints.append(
                "DUAL ANTIPLATELET THERAPY: Multiple antiplatelet agents -- "
                "increased bleeding risk; ensure appropriate indication"
            )

        # Anticoagulant + antiplatelet
        if DrugClass.ANTICOAGULANT in classes and DrugClass.ANTIPLATELET in classes:
            hints.append(
                "ANTICOAGULANT + ANTIPLATELET: Combination increases bleeding risk; "
                "ensure indication is clearly established (e.g., post-ACS with AF)"
            )

        return hints
