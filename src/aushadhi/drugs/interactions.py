"""Interaction detection engine for pairwise and higher-order drug interactions.

The InteractionChecker combines:
  1. A curated table of known pairwise interactions (KNOWN_INTERACTIONS).
  2. Mechanism-based inference using CYP450 profiles, pharmacodynamic classes,
     and special-risk drug sets (serotonergic, QT-prolonging, CNS depressants).
  3. Higher-order interaction detection for 3+ drug combinations that amplify
     risk beyond what pairwise analysis would predict.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from aushadhi.drugs.classes import (
    CNS_DEPRESSANTS,
    QT_PROLONGING_DRUGS,
    SEROTONERGIC_DRUGS,
)
from aushadhi.drugs.database import DrugDatabase
from aushadhi.models import (
    CYP450Role,
    Drug,
    EvidenceLevel,
    Interaction,
    InteractionMechanism,
    Severity,
)


# ---------------------------------------------------------------------------
# Known pairwise interactions -- (drug_a, drug_b) -> Interaction template.
# Stored alphabetically by the first drug name to avoid duplicates.
# ---------------------------------------------------------------------------

KnownInteractionRecord = tuple[str, str, Severity, InteractionMechanism, EvidenceLevel, str, str, str]

KNOWN_INTERACTIONS: list[KnownInteractionRecord] = [
    # Anticoagulant + NSAID / Antiplatelet -- bleeding risk
    ("warfarin", "aspirin", Severity.SEVERE, InteractionMechanism.BLEEDING_RISK,
     EvidenceLevel.ESTABLISHED,
     "Aspirin inhibits platelet function and may displace warfarin from protein binding, greatly increasing bleeding risk.",
     "Major bleeding events including GI and intracranial hemorrhage.",
     "Avoid combination unless clearly indicated; monitor INR closely; consider PPI co-therapy."),
    ("warfarin", "ibuprofen", Severity.SEVERE, InteractionMechanism.BLEEDING_RISK,
     EvidenceLevel.ESTABLISHED,
     "NSAIDs inhibit COX-1 platelet function and may increase warfarin free fraction via protein binding displacement.",
     "Increased risk of GI bleeding and elevated INR.",
     "Avoid NSAIDs in anticoagulated patients; use acetaminophen for pain if possible."),
    ("warfarin", "naproxen", Severity.SEVERE, InteractionMechanism.BLEEDING_RISK,
     EvidenceLevel.ESTABLISHED,
     "Naproxen inhibits platelet COX-1 and competes for CYP2C9, potentially raising warfarin levels.",
     "Elevated INR and increased bleeding risk.",
     "Avoid combination; if necessary, monitor INR every 3-5 days."),
    # Warfarin + CYP inhibitors
    ("warfarin", "fluconazole", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Fluconazole potently inhibits CYP2C9, the major metabolic pathway of S-warfarin.",
     "Markedly elevated INR and bleeding risk.",
     "Reduce warfarin dose by 25-50%; monitor INR within 3-5 days of starting fluconazole."),
    ("warfarin", "metronidazole", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Metronidazole inhibits CYP2C9 metabolism of warfarin.",
     "Elevated INR, potential hemorrhage.",
     "Reduce warfarin dose; monitor INR closely during co-therapy."),
    ("warfarin", "amiodarone", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Amiodarone inhibits CYP2C9 and CYP3A4, markedly reducing warfarin clearance.",
     "Significant INR elevation; effect persists weeks after amiodarone discontinuation.",
     "Reduce warfarin dose by 30-50%; monitor INR weekly for first month."),
    # Warfarin + CYP inducers
    ("warfarin", "carbamazepine", Severity.SEVERE, InteractionMechanism.CYP450_INDUCTION,
     EvidenceLevel.ESTABLISHED,
     "Carbamazepine induces CYP2C9 and CYP3A4, accelerating warfarin metabolism.",
     "Subtherapeutic INR and increased thrombotic risk.",
     "Increase warfarin dose guided by frequent INR monitoring."),
    ("warfarin", "phenytoin", Severity.SEVERE, InteractionMechanism.CYP450_INDUCTION,
     EvidenceLevel.ESTABLISHED,
     "Phenytoin induces CYP2C9 lowering warfarin levels; warfarin may also inhibit phenytoin metabolism.",
     "Bidirectional: subtherapeutic anticoagulation and/or phenytoin toxicity.",
     "Monitor INR and phenytoin levels closely; adjust doses as needed."),
    # Clopidogrel + PPI
    ("clopidogrel", "omeprazole", Severity.MODERATE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel prodrug to active metabolite.",
     "Diminished antiplatelet effect; increased cardiovascular event risk.",
     "Consider pantoprazole (less CYP2C19 inhibition) or alternative acid suppression."),
    ("clopidogrel", "esomeprazole", Severity.MODERATE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.PROBABLE,
     "Esomeprazole may reduce CYP2C19-mediated activation of clopidogrel.",
     "Potentially reduced antiplatelet efficacy.",
     "Prefer pantoprazole if PPI needed with clopidogrel."),
    # Statin + CYP3A4 inhibitors
    ("simvastatin", "clarithromycin", Severity.CONTRAINDICATED, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Clarithromycin potently inhibits CYP3A4, dramatically increasing simvastatin exposure.",
     "Risk of rhabdomyolysis.",
     "Contraindicated. Suspend simvastatin during clarithromycin course."),
    ("simvastatin", "itraconazole", Severity.CONTRAINDICATED, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Itraconazole is a potent CYP3A4 inhibitor; simvastatin AUC increases >10-fold.",
     "High risk of rhabdomyolysis and acute kidney injury.",
     "Contraindicated. Use pravastatin or rosuvastatin if statin needed."),
    ("simvastatin", "erythromycin", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Erythromycin inhibits CYP3A4, increasing simvastatin levels.",
     "Increased risk of myopathy and rhabdomyolysis.",
     "Avoid combination or use alternative statin with less CYP3A4 dependence."),
    ("atorvastatin", "clarithromycin", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Clarithromycin inhibits CYP3A4, increasing atorvastatin exposure.",
     "Risk of myopathy.",
     "Limit atorvastatin to 20 mg/day or use azithromycin instead."),
    ("lovastatin", "ketoconazole", Severity.CONTRAINDICATED, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Ketoconazole potently inhibits CYP3A4; lovastatin AUC increases dramatically.",
     "High risk of rhabdomyolysis.",
     "Contraindicated."),
    # Digoxin interactions
    ("digoxin", "amiodarone", Severity.SEVERE, InteractionMechanism.OTHER,
     EvidenceLevel.ESTABLISHED,
     "Amiodarone increases digoxin levels by ~70% via P-glycoprotein inhibition and reduced renal clearance.",
     "Digoxin toxicity: nausea, visual disturbance, arrhythmia.",
     "Reduce digoxin dose by 50% when starting amiodarone; monitor serum digoxin levels."),
    ("digoxin", "verapamil", Severity.SEVERE, InteractionMechanism.OTHER,
     EvidenceLevel.ESTABLISHED,
     "Verapamil inhibits P-glycoprotein-mediated digoxin efflux, raising serum digoxin 50-75%.",
     "Digoxin toxicity and additive AV nodal depression.",
     "Reduce digoxin dose by 33-50%; monitor levels and heart rate."),
    # ACE inhibitor + Potassium-sparing
    ("lisinopril", "spironolactone", Severity.MODERATE, InteractionMechanism.HYPERKALEMIA,
     EvidenceLevel.ESTABLISHED,
     "Both drugs increase serum potassium via different mechanisms (RAAS inhibition + aldosterone antagonism).",
     "Hyperkalemia, potentially fatal cardiac arrhythmia.",
     "Monitor potassium within 1 week; use low spironolactone dose; avoid in renal impairment."),
    ("enalapril", "spironolactone", Severity.MODERATE, InteractionMechanism.HYPERKALEMIA,
     EvidenceLevel.ESTABLISHED,
     "ACE inhibitor + aldosterone antagonist: additive potassium retention.",
     "Hyperkalemia risk, especially in renal impairment.",
     "Monitor potassium and renal function closely."),
    # ACE/ARB + NSAID
    ("lisinopril", "ibuprofen", Severity.MODERATE, InteractionMechanism.NEPHROTOXICITY,
     EvidenceLevel.ESTABLISHED,
     "NSAIDs reduce prostaglandin-mediated renal blood flow, attenuating ACE inhibitor efficacy and increasing nephrotoxicity.",
     "Reduced antihypertensive effect, acute kidney injury risk.",
     "Avoid chronic NSAID use; monitor BP and renal function."),
    ("lisinopril", "naproxen", Severity.MODERATE, InteractionMechanism.NEPHROTOXICITY,
     EvidenceLevel.ESTABLISHED,
     "NSAIDs blunt ACE inhibitor renal hemodynamic effects.",
     "Reduced BP control, nephrotoxicity.",
     "Use acetaminophen for pain; monitor creatinine."),
    # Metformin + contrast / alcohol
    ("metformin", "ibuprofen", Severity.MILD, InteractionMechanism.NEPHROTOXICITY,
     EvidenceLevel.SUSPECTED,
     "NSAIDs may impair renal function in susceptible patients, reducing metformin clearance.",
     "Increased metformin exposure and lactic acidosis risk in renally impaired patients.",
     "Monitor renal function; use short courses of NSAIDs."),
    # Lithium + NSAIDs / diuretics
    ("lithium", "ibuprofen", Severity.SEVERE, InteractionMechanism.RENAL_COMPETITION,
     EvidenceLevel.ESTABLISHED,
     "NSAIDs reduce renal lithium clearance by decreasing prostaglandin-mediated renal blood flow.",
     "Lithium toxicity: tremor, ataxia, renal failure, seizures.",
     "Avoid combination; if necessary, reduce lithium dose and monitor levels within 5 days."),
    ("lithium", "hydrochlorothiazide", Severity.SEVERE, InteractionMechanism.RENAL_COMPETITION,
     EvidenceLevel.ESTABLISHED,
     "Thiazides decrease sodium and water excretion, promoting lithium reabsorption in proximal tubule.",
     "Elevated lithium levels and toxicity.",
     "Reduce lithium dose by 50%; monitor lithium levels."),
    ("lithium", "furosemide", Severity.MODERATE, InteractionMechanism.RENAL_COMPETITION,
     EvidenceLevel.PROBABLE,
     "Loop diuretics cause sodium depletion, potentially increasing lithium reabsorption.",
     "Elevated lithium levels.",
     "Monitor lithium levels; maintain adequate sodium intake."),
    # SSRI + SSRI / TCA / Tramadol -- serotonin syndrome
    ("fluoxetine", "tramadol", Severity.SEVERE, InteractionMechanism.SEROTONIN_SYNDROME,
     EvidenceLevel.ESTABLISHED,
     "Both agents increase serotonergic activity; fluoxetine also inhibits CYP2D6 altering tramadol metabolism.",
     "Serotonin syndrome: agitation, hyperthermia, clonus, autonomic instability.",
     "Avoid combination; if necessary, use lowest effective doses and monitor for serotonin syndrome."),
    ("sertraline", "tramadol", Severity.SEVERE, InteractionMechanism.SEROTONIN_SYNDROME,
     EvidenceLevel.ESTABLISHED,
     "Additive serotonergic activity with risk of serotonin syndrome.",
     "Serotonin syndrome risk.",
     "Avoid combination when possible; use non-serotonergic analgesic."),
    ("fluoxetine", "amitriptyline", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Fluoxetine potently inhibits CYP2D6, markedly increasing TCA levels. Additive serotonergic/anticholinergic effects.",
     "TCA toxicity: cardiac conduction abnormalities, seizures, anticholinergic crisis.",
     "Avoid combination or use very low TCA doses with therapeutic drug monitoring."),
    # SSRI + NSAID -- bleeding
    ("fluoxetine", "aspirin", Severity.MODERATE, InteractionMechanism.BLEEDING_RISK,
     EvidenceLevel.ESTABLISHED,
     "SSRIs impair platelet serotonin uptake; combined with antiplatelet/NSAID increases GI bleeding risk.",
     "2-3x increased risk of upper GI bleeding.",
     "Add PPI prophylaxis if combination necessary."),
    ("sertraline", "aspirin", Severity.MODERATE, InteractionMechanism.BLEEDING_RISK,
     EvidenceLevel.ESTABLISHED,
     "SSRIs reduce platelet serotonin; aspirin inhibits COX-1. Additive bleeding risk.",
     "Increased GI bleeding risk.",
     "Consider PPI co-therapy."),
    # Opioid + benzodiazepine
    ("oxycodone", "alprazolam", Severity.SEVERE, InteractionMechanism.CNS_DEPRESSION,
     EvidenceLevel.ESTABLISHED,
     "Additive CNS and respiratory depression from combined opioid and benzodiazepine use.",
     "Fatal respiratory depression.",
     "Avoid concurrent use; FDA black box warning. If necessary, use lowest effective doses."),
    ("morphine", "lorazepam", Severity.SEVERE, InteractionMechanism.CNS_DEPRESSION,
     EvidenceLevel.ESTABLISHED,
     "Additive CNS depression; both cause respiratory depression.",
     "Fatal respiratory depression risk.",
     "Avoid concurrent prescribing when possible; monitor respiratory status."),
    ("fentanyl", "alprazolam", Severity.CONTRAINDICATED, InteractionMechanism.CNS_DEPRESSION,
     EvidenceLevel.ESTABLISHED,
     "Potent opioid + benzodiazepine: high risk of fatal respiratory depression.",
     "Death from respiratory failure.",
     "Contraindicated except in monitored settings with no alternative."),
    ("hydrocodone", "diazepam", Severity.SEVERE, InteractionMechanism.CNS_DEPRESSION,
     EvidenceLevel.ESTABLISHED,
     "Additive CNS and respiratory depression.",
     "Respiratory depression, profound sedation.",
     "Avoid or minimize concurrent use; monitor closely."),
    # Methotrexate + NSAID / TMP-SMX
    ("methotrexate", "ibuprofen", Severity.SEVERE, InteractionMechanism.RENAL_COMPETITION,
     EvidenceLevel.ESTABLISHED,
     "NSAIDs reduce renal clearance of methotrexate and may displace it from protein binding.",
     "Methotrexate toxicity: pancytopenia, mucositis, renal failure.",
     "Avoid NSAIDs with high-dose methotrexate; monitor CBC and renal function."),
    ("methotrexate", "trimethoprim-sulfamethoxazole", Severity.SEVERE, InteractionMechanism.PHARMACODYNAMIC_SYNERGY,
     EvidenceLevel.ESTABLISHED,
     "Both are antifolate agents; additive bone marrow suppression.",
     "Severe pancytopenia, megaloblastic anemia.",
     "Avoid combination; use alternative antibiotic."),
    # Beta-blocker + CCB (non-DHP)
    ("metoprolol", "verapamil", Severity.SEVERE, InteractionMechanism.PHARMACODYNAMIC_SYNERGY,
     EvidenceLevel.ESTABLISHED,
     "Both drugs depress AV nodal conduction and myocardial contractility.",
     "Severe bradycardia, heart block, heart failure.",
     "Avoid IV combination; oral combination only with extreme caution and monitoring."),
    ("metoprolol", "diltiazem", Severity.MODERATE, InteractionMechanism.PHARMACODYNAMIC_SYNERGY,
     EvidenceLevel.ESTABLISHED,
     "Additive negative chronotropic and dromotropic effects.",
     "Bradycardia, heart block.",
     "Monitor heart rate and ECG; avoid in patients with conduction disease."),
    # QT prolongation combinations
    ("amiodarone", "ciprofloxacin", Severity.SEVERE, InteractionMechanism.QT_PROLONGATION,
     EvidenceLevel.ESTABLISHED,
     "Both drugs independently prolong QT interval; combination has additive effect.",
     "Torsades de pointes, sudden cardiac death.",
     "Avoid combination; use alternative antibiotic."),
    ("haloperidol", "methadone", Severity.SEVERE, InteractionMechanism.QT_PROLONGATION,
     EvidenceLevel.ESTABLISHED,
     "Both drugs prolong QT interval through different mechanisms.",
     "Torsades de pointes.",
     "ECG monitoring required; avoid if baseline QTc >500 ms."),
    # Immunosuppressant + CYP3A4 inhibitors
    ("cyclosporine", "ketoconazole", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Ketoconazole inhibits CYP3A4, increasing cyclosporine levels 2-3 fold.",
     "Nephrotoxicity, hepatotoxicity, neurotoxicity.",
     "Reduce cyclosporine dose; monitor drug levels closely."),
    ("tacrolimus", "clarithromycin", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Clarithromycin inhibits CYP3A4 and P-glycoprotein, increasing tacrolimus levels.",
     "Nephrotoxicity, neurotoxicity.",
     "Use azithromycin instead; if clarithromycin necessary, reduce tacrolimus dose and monitor levels."),
    # Theophylline interactions
    ("theophylline", "ciprofloxacin", Severity.SEVERE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Ciprofloxacin inhibits CYP1A2, the primary metabolic pathway of theophylline.",
     "Theophylline toxicity: seizures, arrhythmias.",
     "Reduce theophylline dose by 30-50%; monitor serum levels."),
    ("theophylline", "erythromycin", Severity.MODERATE, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Erythromycin inhibits CYP3A4, modestly increasing theophylline levels.",
     "Theophylline toxicity at higher levels.",
     "Monitor theophylline levels; adjust dose as needed."),
    # Colchicine interactions
    ("colchicine", "clarithromycin", Severity.CONTRAINDICATED, InteractionMechanism.CYP450_INHIBITION,
     EvidenceLevel.ESTABLISHED,
     "Clarithromycin inhibits CYP3A4 and P-glycoprotein, dramatically increasing colchicine exposure.",
     "Fatal colchicine toxicity reported.",
     "Contraindicated, especially in renal/hepatic impairment."),
    # Levothyroxine absorption
    ("levothyroxine", "omeprazole", Severity.MILD, InteractionMechanism.GI_ABSORPTION,
     EvidenceLevel.PROBABLE,
     "PPIs increase gastric pH, potentially reducing levothyroxine absorption.",
     "Subtherapeutic thyroid hormone levels.",
     "Separate administration by 4 hours; monitor TSH."),
    ("levothyroxine", "calcium carbonate", Severity.MODERATE, InteractionMechanism.GI_ABSORPTION,
     EvidenceLevel.ESTABLISHED,
     "Calcium chelates levothyroxine in the GI tract, reducing absorption.",
     "Reduced levothyroxine efficacy; elevated TSH.",
     "Separate by at least 4 hours."),
    # Sildenafil + nitrates
    ("sildenafil", "nitroglycerin", Severity.CONTRAINDICATED, InteractionMechanism.PHARMACODYNAMIC_SYNERGY,
     EvidenceLevel.ESTABLISHED,
     "Both increase cGMP-mediated vasodilation; combination causes severe hypotension.",
     "Life-threatening hypotension and cardiovascular collapse.",
     "Absolutely contraindicated. Allow 24-48 hour washout."),
    # Allopurinol + azathioprine (mercaptopurine)
    ("allopurinol", "methotrexate", Severity.MODERATE, InteractionMechanism.RENAL_COMPETITION,
     EvidenceLevel.PROBABLE,
     "Allopurinol may reduce renal excretion of methotrexate.",
     "Increased methotrexate toxicity.",
     "Monitor methotrexate levels and CBC."),
]


class InteractionChecker:
    """Detects drug interactions using known pairs and mechanism-based inference.

    Parameters
    ----------
    db : DrugDatabase
        The drug database to use for resolving drug profiles.
    """

    def __init__(self, db: DrugDatabase) -> None:
        self.db = db
        self._known: dict[tuple[str, str], KnownInteractionRecord] = {}
        self._index_known()

    def _index_known(self) -> None:
        """Build a lookup index of known interaction pairs."""
        for record in KNOWN_INTERACTIONS:
            key = self._pair_key(record[0], record[1])
            self._known[key] = record

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        """Canonical ordered pair key."""
        return tuple(sorted([a.lower(), b.lower()]))  # type: ignore[return-value]

    def check_pair(self, drug_a: Drug, drug_b: Drug) -> list[Interaction]:
        """Check for interactions between two drugs.

        Returns a list because a single pair may have multiple interaction
        mechanisms (e.g. bleeding risk *and* CYP inhibition).
        """
        interactions: list[Interaction] = []

        # 1. Known interaction lookup
        key = self._pair_key(drug_a.name, drug_b.name)
        if key in self._known:
            rec = self._known[key]
            interactions.append(
                Interaction(
                    drugs=[rec[0], rec[1]],
                    severity=rec[2],
                    mechanism=rec[3],
                    evidence=rec[4],
                    description=rec[5],
                    clinical_significance=rec[6],
                    management=rec[7],
                )
            )

        # 2. Mechanism-based inference (only if no known interaction found)
        if not interactions:
            interactions.extend(self._infer_cyp_interactions(drug_a, drug_b))
            interactions.extend(self._infer_pharmacodynamic(drug_a, drug_b))

        return interactions

    def check_all(self, drugs: list[Drug]) -> list[Interaction]:
        """Check all pairwise interactions plus higher-order interactions."""
        interactions: list[Interaction] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Pairwise
        for a, b in combinations(drugs, 2):
            key = self._pair_key(a.name, b.name)
            if key not in seen_pairs:
                seen_pairs.add(key)
                interactions.extend(self.check_pair(a, b))

        # Higher-order
        interactions.extend(self._check_higher_order(drugs))

        return interactions

    def _infer_cyp_interactions(self, a: Drug, b: Drug) -> list[Interaction]:
        """Infer CYP450-mediated interactions from enzyme profiles."""
        results: list[Interaction] = []

        for entry_a in a.cyp450:
            for entry_b in b.cyp450:
                if entry_a.enzyme != entry_b.enzyme:
                    continue

                # Inhibitor + substrate on same enzyme
                if entry_a.role == CYP450Role.INHIBITOR and entry_b.role == CYP450Role.SUBSTRATE:
                    severity = Severity.SEVERE if b.narrow_therapeutic_index else Severity.MODERATE
                    results.append(
                        Interaction(
                            drugs=[a.name, b.name],
                            severity=severity,
                            mechanism=InteractionMechanism.CYP450_INHIBITION,
                            evidence=EvidenceLevel.THEORETICAL,
                            description=(
                                f"{a.name} inhibits {entry_a.enzyme}, which metabolizes {b.name}. "
                                f"This may increase {b.name} exposure."
                            ),
                            clinical_significance=f"Increased {b.name} levels and toxicity risk.",
                            management=f"Monitor for {b.name} toxicity; consider dose reduction.",
                        )
                    )
                    break

                if entry_b.role == CYP450Role.INHIBITOR and entry_a.role == CYP450Role.SUBSTRATE:
                    severity = Severity.SEVERE if a.narrow_therapeutic_index else Severity.MODERATE
                    results.append(
                        Interaction(
                            drugs=[a.name, b.name],
                            severity=severity,
                            mechanism=InteractionMechanism.CYP450_INHIBITION,
                            evidence=EvidenceLevel.THEORETICAL,
                            description=(
                                f"{b.name} inhibits {entry_b.enzyme}, which metabolizes {a.name}. "
                                f"This may increase {a.name} exposure."
                            ),
                            clinical_significance=f"Increased {a.name} levels and toxicity risk.",
                            management=f"Monitor for {a.name} toxicity; consider dose reduction.",
                        )
                    )
                    break

                # Inducer + substrate on same enzyme
                if entry_a.role == CYP450Role.INDUCER and entry_b.role == CYP450Role.SUBSTRATE:
                    severity = Severity.SEVERE if b.narrow_therapeutic_index else Severity.MODERATE
                    results.append(
                        Interaction(
                            drugs=[a.name, b.name],
                            severity=severity,
                            mechanism=InteractionMechanism.CYP450_INDUCTION,
                            evidence=EvidenceLevel.THEORETICAL,
                            description=(
                                f"{a.name} induces {entry_a.enzyme}, which metabolizes {b.name}. "
                                f"This may decrease {b.name} exposure."
                            ),
                            clinical_significance=f"Reduced {b.name} efficacy.",
                            management=f"Monitor therapeutic response; may need to increase {b.name} dose.",
                        )
                    )
                    break

                if entry_b.role == CYP450Role.INDUCER and entry_a.role == CYP450Role.SUBSTRATE:
                    severity = Severity.SEVERE if a.narrow_therapeutic_index else Severity.MODERATE
                    results.append(
                        Interaction(
                            drugs=[a.name, b.name],
                            severity=severity,
                            mechanism=InteractionMechanism.CYP450_INDUCTION,
                            evidence=EvidenceLevel.THEORETICAL,
                            description=(
                                f"{b.name} induces {entry_b.enzyme}, which metabolizes {a.name}. "
                                f"This may decrease {a.name} exposure."
                            ),
                            clinical_significance=f"Reduced {a.name} efficacy.",
                            management=f"Monitor therapeutic response; may need to increase {a.name} dose.",
                        )
                    )
                    break

        return results

    def _infer_pharmacodynamic(self, a: Drug, b: Drug) -> list[Interaction]:
        """Infer pharmacodynamic interactions from drug class membership."""
        results: list[Interaction] = []
        names = {a.name, b.name}

        # Serotonin syndrome risk
        sero_overlap = names & SEROTONERGIC_DRUGS
        if len(sero_overlap) == 2:
            results.append(
                Interaction(
                    drugs=[a.name, b.name],
                    severity=Severity.SEVERE,
                    mechanism=InteractionMechanism.SEROTONIN_SYNDROME,
                    evidence=EvidenceLevel.SUSPECTED,
                    description="Both drugs have serotonergic activity; concurrent use increases serotonin syndrome risk.",
                    clinical_significance="Serotonin syndrome: agitation, hyperthermia, clonus, autonomic instability.",
                    management="Monitor for serotonin syndrome symptoms; use lowest effective doses.",
                )
            )

        # QT prolongation risk
        qt_overlap = names & QT_PROLONGING_DRUGS
        if len(qt_overlap) == 2:
            results.append(
                Interaction(
                    drugs=[a.name, b.name],
                    severity=Severity.SEVERE,
                    mechanism=InteractionMechanism.QT_PROLONGATION,
                    evidence=EvidenceLevel.SUSPECTED,
                    description="Both drugs prolong the QT interval; additive risk of torsades de pointes.",
                    clinical_significance="Torsades de pointes, potentially fatal arrhythmia.",
                    management="ECG monitoring; avoid if baseline QTc >450 ms.",
                )
            )

        # CNS depression
        cns_overlap = names & CNS_DEPRESSANTS
        if len(cns_overlap) == 2:
            results.append(
                Interaction(
                    drugs=[a.name, b.name],
                    severity=Severity.MODERATE,
                    mechanism=InteractionMechanism.CNS_DEPRESSION,
                    evidence=EvidenceLevel.SUSPECTED,
                    description="Both drugs cause CNS depression; additive sedation and respiratory depression risk.",
                    clinical_significance="Excessive sedation, respiratory depression, falls.",
                    management="Use lowest effective doses; counsel patient about sedation.",
                )
            )

        # Dual antiplatelet/anticoagulant bleeding risk
        bleed_classes = {DrugClass.ANTICOAGULANT, DrugClass.ANTIPLATELET, DrugClass.NSAID}
        if a.drug_class in bleed_classes and b.drug_class in bleed_classes and a.drug_class != b.drug_class:
            results.append(
                Interaction(
                    drugs=[a.name, b.name],
                    severity=Severity.MODERATE,
                    mechanism=InteractionMechanism.BLEEDING_RISK,
                    evidence=EvidenceLevel.SUSPECTED,
                    description=f"Combining {a.drug_class.value} ({a.name}) with {b.drug_class.value} ({b.name}) increases bleeding risk.",
                    clinical_significance="Increased risk of hemorrhagic events.",
                    management="Assess bleeding risk-benefit; consider GI prophylaxis.",
                )
            )

        return results

    def _check_higher_order(self, drugs: list[Drug]) -> list[Interaction]:
        """Detect higher-order interactions involving 3+ drugs.

        These capture synergistic risks that exceed what pairwise analysis
        alone can identify (e.g. triple whammy nephrotoxicity, multiple
        CNS depressants, serotonergic load).
        """
        results: list[Interaction] = []
        names = [d.name for d in drugs]
        name_set = set(names)
        classes = {d.drug_class for d in drugs}

        # Triple whammy: ACE/ARB + diuretic + NSAID -> acute kidney injury
        has_raas = classes & {DrugClass.ACE_INHIBITOR, DrugClass.ARB}
        has_diuretic = DrugClass.DIURETIC in classes
        has_nsaid = DrugClass.NSAID in classes
        if has_raas and has_diuretic and has_nsaid:
            involved = [
                d.name
                for d in drugs
                if d.drug_class in {DrugClass.ACE_INHIBITOR, DrugClass.ARB, DrugClass.DIURETIC, DrugClass.NSAID}
            ]
            results.append(
                Interaction(
                    drugs=involved,
                    severity=Severity.SEVERE,
                    mechanism=InteractionMechanism.NEPHROTOXICITY,
                    evidence=EvidenceLevel.ESTABLISHED,
                    description="'Triple whammy': ACE inhibitor/ARB + diuretic + NSAID synergistically impair renal hemodynamics.",
                    clinical_significance="Acute kidney injury, especially in elderly or volume-depleted patients.",
                    management="Avoid triple combination; if unavoidable, monitor renal function and hydration status closely.",
                )
            )

        # Multiple serotonergic agents (3+)
        sero_drugs = name_set & SEROTONERGIC_DRUGS
        if len(sero_drugs) >= 3:
            results.append(
                Interaction(
                    drugs=sorted(sero_drugs),
                    severity=Severity.SEVERE,
                    mechanism=InteractionMechanism.SEROTONIN_SYNDROME,
                    evidence=EvidenceLevel.PROBABLE,
                    description=f"Three or more serotonergic agents ({', '.join(sorted(sero_drugs))}) greatly increase serotonin syndrome risk.",
                    clinical_significance="High risk of serotonin syndrome.",
                    management="Review medication list; discontinue unnecessary serotonergic agents.",
                )
            )

        # Multiple CNS depressants (3+)
        cns_drugs = name_set & CNS_DEPRESSANTS
        if len(cns_drugs) >= 3:
            results.append(
                Interaction(
                    drugs=sorted(cns_drugs),
                    severity=Severity.SEVERE,
                    mechanism=InteractionMechanism.CNS_DEPRESSION,
                    evidence=EvidenceLevel.PROBABLE,
                    description=f"Three or more CNS depressants ({', '.join(sorted(cns_drugs))}) create high risk of profound sedation.",
                    clinical_significance="Profound sedation, respiratory arrest, death.",
                    management="Review and minimize CNS depressant burden; consider tapering.",
                )
            )

        # Multiple QT-prolonging drugs (3+)
        qt_drugs = name_set & QT_PROLONGING_DRUGS
        if len(qt_drugs) >= 3:
            results.append(
                Interaction(
                    drugs=sorted(qt_drugs),
                    severity=Severity.SEVERE,
                    mechanism=InteractionMechanism.QT_PROLONGATION,
                    evidence=EvidenceLevel.PROBABLE,
                    description=f"Three or more QT-prolonging drugs ({', '.join(sorted(qt_drugs))}) create high risk of torsades de pointes.",
                    clinical_significance="Torsades de pointes, sudden cardiac death.",
                    management="Baseline and serial ECG monitoring; discontinue non-essential QT-prolonging agents.",
                )
            )

        return results
