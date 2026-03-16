"""Pydantic data models for Aushadhi drug interaction prediction."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CYP450Role(str, Enum):
    """Role a drug plays with respect to a CYP450 enzyme."""

    SUBSTRATE = "substrate"
    INHIBITOR = "inhibitor"
    INDUCER = "inducer"


class CYP450Entry(BaseModel):
    """A single CYP450 enzyme interaction record."""

    enzyme: str = Field(..., description="CYP450 enzyme name, e.g. CYP3A4")
    role: CYP450Role


class DrugClass(str, Enum):
    """Broad pharmacological drug classes."""

    ANALGESIC = "analgesic"
    ANTICOAGULANT = "anticoagulant"
    ANTIPLATELET = "antiplatelet"
    ANTIHYPERTENSIVE = "antihypertensive"
    ANTIARRHYTHMIC = "antiarrhythmic"
    ANTIDIABETIC = "antidiabetic"
    STATIN = "statin"
    ANTIBIOTIC = "antibiotic"
    ANTIFUNGAL = "antifungal"
    ANTIVIRAL = "antiviral"
    ANTIDEPRESSANT = "antidepressant"
    ANTIPSYCHOTIC = "antipsychotic"
    ANXIOLYTIC = "anxiolytic"
    SEDATIVE = "sedative"
    ANTICONVULSANT = "anticonvulsant"
    OPIOID = "opioid"
    NSAID = "nsaid"
    PPI = "proton_pump_inhibitor"
    H2_BLOCKER = "h2_blocker"
    CORTICOSTEROID = "corticosteroid"
    BRONCHODILATOR = "bronchodilator"
    DIURETIC = "diuretic"
    ACE_INHIBITOR = "ace_inhibitor"
    ARB = "arb"
    BETA_BLOCKER = "beta_blocker"
    CCB = "calcium_channel_blocker"
    ANTIHISTAMINE = "antihistamine"
    IMMUNOSUPPRESSANT = "immunosuppressant"
    THYROID = "thyroid"
    MUSCLE_RELAXANT = "muscle_relaxant"
    ANTIEMETIC = "antiemetic"
    LAXATIVE = "laxative"
    ANTITUSSIVE = "antitussive"
    OTHER = "other"


class Drug(BaseModel):
    """Represents a single medication with its pharmacological profile."""

    name: str = Field(..., description="Generic drug name (lowercase)")
    brand_names: list[str] = Field(default_factory=list)
    drug_class: DrugClass
    mechanism: str = Field(..., description="Mechanism of action")
    cyp450: list[CYP450Entry] = Field(
        default_factory=list,
        description="CYP450 enzyme interactions",
    )
    half_life_hours: Optional[float] = None
    protein_binding_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Plasma protein binding percentage"
    )
    narrow_therapeutic_index: bool = Field(
        False, description="Whether the drug has a narrow therapeutic index"
    )
    pregnancy_category: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Drug):
            return self.name == other.name
        return NotImplemented


class Severity(str, Enum):
    """Interaction severity levels."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CONTRAINDICATED = "contraindicated"


class EvidenceLevel(str, Enum):
    """Evidence level for an interaction."""

    ESTABLISHED = "established"
    PROBABLE = "probable"
    SUSPECTED = "suspected"
    THEORETICAL = "theoretical"


class InteractionMechanism(str, Enum):
    """Broad mechanism category for a drug interaction."""

    CYP450_INHIBITION = "cyp450_inhibition"
    CYP450_INDUCTION = "cyp450_induction"
    PHARMACODYNAMIC_SYNERGY = "pharmacodynamic_synergy"
    PHARMACODYNAMIC_ANTAGONISM = "pharmacodynamic_antagonism"
    PROTEIN_BINDING_DISPLACEMENT = "protein_binding_displacement"
    RENAL_COMPETITION = "renal_competition"
    GI_ABSORPTION = "gi_absorption"
    QT_PROLONGATION = "qt_prolongation"
    SEROTONIN_SYNDROME = "serotonin_syndrome"
    BLEEDING_RISK = "bleeding_risk"
    CNS_DEPRESSION = "cns_depression"
    HYPERKALEMIA = "hyperkalemia"
    NEPHROTOXICITY = "nephrotoxicity"
    HEPATOTOXICITY = "hepatotoxicity"
    OTHER = "other"


class Interaction(BaseModel):
    """A detected interaction between two or more drugs."""

    drugs: list[str] = Field(..., min_length=2, description="Drug names involved")
    severity: Severity
    mechanism: InteractionMechanism
    evidence: EvidenceLevel = EvidenceLevel.SUSPECTED
    description: str = ""
    clinical_significance: str = ""
    management: str = ""

    @property
    def is_pairwise(self) -> bool:
        return len(self.drugs) == 2


class RiskAssessment(BaseModel):
    """Full risk assessment for an interaction."""

    interaction: Interaction
    severity: Severity
    risk_score: float = Field(..., ge=0.0, le=10.0, description="Numeric risk 0-10")
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    monitoring_parameters: list[str] = Field(default_factory=list)


class MedicationList(BaseModel):
    """A patient's current medication list for polypharmacy analysis."""

    patient_id: Optional[str] = None
    drugs: list[Drug] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)
    risk_assessments: list[RiskAssessment] = Field(default_factory=list)
    overall_risk_score: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Aggregate polypharmacy risk"
    )

    @property
    def drug_count(self) -> int:
        return len(self.drugs)

    @property
    def is_polypharmacy(self) -> bool:
        """Polypharmacy is commonly defined as 5+ concurrent medications."""
        return self.drug_count >= 5

    @property
    def severe_interactions(self) -> list[Interaction]:
        return [
            i
            for i in self.interactions
            if i.severity in (Severity.SEVERE, Severity.CONTRAINDICATED)
        ]
