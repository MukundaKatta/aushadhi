"""Drug classes, mechanisms of action, and CYP450 enzyme definitions.

This module provides the canonical reference tables used by the drug database
and interaction checker.  All CYP450 enzyme names follow standard nomenclature
(e.g. CYP3A4, CYP2D6).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CYP450 enzymes relevant to clinical drug interactions
# ---------------------------------------------------------------------------

CYP450_ENZYMES: list[str] = [
    "CYP1A2",
    "CYP2B6",
    "CYP2C8",
    "CYP2C9",
    "CYP2C19",
    "CYP2D6",
    "CYP2E1",
    "CYP3A4",
    "CYP3A5",
]

# ---------------------------------------------------------------------------
# Drug class descriptions -- maps DrugClass enum values to human-readable
# descriptions with typical mechanism of action.
# ---------------------------------------------------------------------------

DRUG_CLASS_INFO: dict[str, dict[str, str]] = {
    "analgesic": {
        "description": "Pain-relieving medications",
        "typical_mechanism": "Inhibition of pain signal transmission or perception",
    },
    "anticoagulant": {
        "description": "Blood thinners that prevent clot formation",
        "typical_mechanism": "Inhibition of clotting factor synthesis or activity",
    },
    "antiplatelet": {
        "description": "Agents that prevent platelet aggregation",
        "typical_mechanism": "Inhibition of platelet activation pathways",
    },
    "antihypertensive": {
        "description": "Blood pressure-lowering medications",
        "typical_mechanism": "Various: vasodilation, reduced cardiac output, fluid loss",
    },
    "antiarrhythmic": {
        "description": "Heart rhythm regulators",
        "typical_mechanism": "Modulation of cardiac ion channels",
    },
    "antidiabetic": {
        "description": "Blood glucose-lowering agents",
        "typical_mechanism": "Insulin sensitization, secretagogue, or glucose excretion",
    },
    "statin": {
        "description": "Cholesterol-lowering HMG-CoA reductase inhibitors",
        "typical_mechanism": "Inhibition of HMG-CoA reductase in hepatic cholesterol synthesis",
    },
    "antibiotic": {
        "description": "Antibacterial agents",
        "typical_mechanism": "Disruption of bacterial cell wall, protein, or nucleic acid synthesis",
    },
    "antifungal": {
        "description": "Antifungal agents",
        "typical_mechanism": "Disruption of fungal cell membrane ergosterol synthesis",
    },
    "antiviral": {
        "description": "Antiviral agents",
        "typical_mechanism": "Inhibition of viral replication enzymes",
    },
    "antidepressant": {
        "description": "Mood-elevating medications",
        "typical_mechanism": "Modulation of serotonin, norepinephrine, or dopamine signaling",
    },
    "antipsychotic": {
        "description": "Agents for psychotic disorders",
        "typical_mechanism": "Dopamine D2 receptor antagonism",
    },
    "anxiolytic": {
        "description": "Anti-anxiety medications",
        "typical_mechanism": "GABA-A receptor positive allosteric modulation",
    },
    "sedative": {
        "description": "Sleep-promoting agents",
        "typical_mechanism": "CNS depression via GABAergic enhancement",
    },
    "anticonvulsant": {
        "description": "Seizure-preventing medications",
        "typical_mechanism": "Sodium/calcium channel blockade or GABA enhancement",
    },
    "opioid": {
        "description": "Opioid receptor agonists for pain",
        "typical_mechanism": "Mu-opioid receptor agonism in CNS",
    },
    "nsaid": {
        "description": "Non-steroidal anti-inflammatory drugs",
        "typical_mechanism": "Cyclooxygenase (COX-1/COX-2) inhibition",
    },
    "proton_pump_inhibitor": {
        "description": "Gastric acid suppressants",
        "typical_mechanism": "Irreversible inhibition of H+/K+ ATPase proton pump",
    },
    "h2_blocker": {
        "description": "Histamine H2 receptor antagonists",
        "typical_mechanism": "Competitive inhibition of histamine at H2 receptors on parietal cells",
    },
    "corticosteroid": {
        "description": "Anti-inflammatory / immunosuppressive steroids",
        "typical_mechanism": "Glucocorticoid receptor activation suppressing inflammatory genes",
    },
    "bronchodilator": {
        "description": "Airway-opening agents",
        "typical_mechanism": "Beta-2 adrenergic agonism or muscarinic antagonism",
    },
    "diuretic": {
        "description": "Agents promoting renal fluid excretion",
        "typical_mechanism": "Inhibition of renal sodium reabsorption",
    },
    "ace_inhibitor": {
        "description": "Angiotensin-converting enzyme inhibitors",
        "typical_mechanism": "Inhibition of ACE preventing angiotensin II formation",
    },
    "arb": {
        "description": "Angiotensin II receptor blockers",
        "typical_mechanism": "Competitive antagonism at AT1 receptors",
    },
    "beta_blocker": {
        "description": "Beta-adrenergic receptor antagonists",
        "typical_mechanism": "Blockade of beta-1/beta-2 adrenergic receptors",
    },
    "calcium_channel_blocker": {
        "description": "Calcium channel antagonists",
        "typical_mechanism": "Blockade of L-type calcium channels in vascular smooth muscle / heart",
    },
    "antihistamine": {
        "description": "Histamine H1 receptor antagonists",
        "typical_mechanism": "Competitive antagonism at H1 receptors",
    },
    "immunosuppressant": {
        "description": "Immune system suppressants",
        "typical_mechanism": "Calcineurin inhibition or antimetabolite activity",
    },
    "thyroid": {
        "description": "Thyroid hormone replacement or anti-thyroid agents",
        "typical_mechanism": "Supplementation or inhibition of thyroid hormone synthesis",
    },
    "muscle_relaxant": {
        "description": "Skeletal muscle relaxants",
        "typical_mechanism": "CNS depression or direct muscle fiber inhibition",
    },
    "antiemetic": {
        "description": "Anti-nausea and anti-vomiting agents",
        "typical_mechanism": "5-HT3 or dopamine receptor antagonism in CTZ",
    },
    "laxative": {
        "description": "Agents promoting bowel movements",
        "typical_mechanism": "Osmotic or stimulant action on intestinal motility",
    },
    "antitussive": {
        "description": "Cough suppressants",
        "typical_mechanism": "Suppression of cough reflex in medulla",
    },
    "other": {
        "description": "Miscellaneous pharmacological agents",
        "typical_mechanism": "Various",
    },
}

# ---------------------------------------------------------------------------
# Major CYP450 inhibitors / inducers -- quick-reference used by the
# interaction checker for mechanism-based inference.
# ---------------------------------------------------------------------------

POTENT_CYP_INHIBITORS: dict[str, list[str]] = {
    "CYP1A2": ["fluvoxamine", "ciprofloxacin", "enoxacin"],
    "CYP2C9": ["fluconazole", "amiodarone", "miconazole", "fluvoxamine"],
    "CYP2C19": ["fluvoxamine", "fluconazole", "omeprazole", "esomeprazole"],
    "CYP2D6": ["fluoxetine", "paroxetine", "bupropion", "quinidine", "duloxetine"],
    "CYP3A4": [
        "ketoconazole",
        "itraconazole",
        "clarithromycin",
        "erythromycin",
        "ritonavir",
        "verapamil",
        "diltiazem",
        "grapefruit",
    ],
}

POTENT_CYP_INDUCERS: dict[str, list[str]] = {
    "CYP1A2": ["smoking", "rifampin", "carbamazepine", "phenytoin"],
    "CYP2C9": ["rifampin", "carbamazepine", "phenytoin", "phenobarbital"],
    "CYP2C19": ["rifampin", "carbamazepine", "phenytoin"],
    "CYP2D6": [],  # CYP2D6 is generally not inducible
    "CYP3A4": [
        "rifampin",
        "carbamazepine",
        "phenytoin",
        "phenobarbital",
        "St. John's wort",
        "efavirenz",
    ],
}

# ---------------------------------------------------------------------------
# Serotonergic drugs -- used to flag serotonin syndrome risk
# ---------------------------------------------------------------------------

SEROTONERGIC_DRUGS: set[str] = {
    "fluoxetine",
    "sertraline",
    "paroxetine",
    "citalopram",
    "escitalopram",
    "fluvoxamine",
    "venlafaxine",
    "desvenlafaxine",
    "duloxetine",
    "milnacipran",
    "amitriptyline",
    "nortriptyline",
    "imipramine",
    "clomipramine",
    "doxepin",
    "trazodone",
    "mirtazapine",
    "tramadol",
    "fentanyl",
    "meperidine",
    "methadone",
    "linezolid",
    "methylene blue",
    "lithium",
    "buspirone",
    "ondansetron",
    "granisetron",
    "sumatriptan",
}

# ---------------------------------------------------------------------------
# QT-prolonging drugs -- used to flag additive QT risk
# ---------------------------------------------------------------------------

QT_PROLONGING_DRUGS: set[str] = {
    "amiodarone",
    "sotalol",
    "dofetilide",
    "dronedarone",
    "quinidine",
    "procainamide",
    "haloperidol",
    "ziprasidone",
    "thioridazine",
    "chlorpromazine",
    "methadone",
    "erythromycin",
    "clarithromycin",
    "moxifloxacin",
    "levofloxacin",
    "ciprofloxacin",
    "fluconazole",
    "ondansetron",
    "domperidone",
    "citalopram",
    "escitalopram",
}

# ---------------------------------------------------------------------------
# CNS depressants -- used to flag additive sedation
# ---------------------------------------------------------------------------

CNS_DEPRESSANTS: set[str] = {
    "alprazolam",
    "lorazepam",
    "diazepam",
    "clonazepam",
    "midazolam",
    "triazolam",
    "zolpidem",
    "zopiclone",
    "eszopiclone",
    "morphine",
    "oxycodone",
    "hydrocodone",
    "codeine",
    "fentanyl",
    "tramadol",
    "methadone",
    "gabapentin",
    "pregabalin",
    "amitriptyline",
    "doxepin",
    "trazodone",
    "mirtazapine",
    "quetiapine",
    "olanzapine",
    "chlorpromazine",
    "hydroxyzine",
    "diphenhydramine",
    "promethazine",
    "cyclobenzaprine",
    "baclofen",
    "carisoprodol",
    "phenobarbital",
}
