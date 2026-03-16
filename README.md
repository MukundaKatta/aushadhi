# Aushadhi -- AI Drug Interaction Predictor for Polypharmacy Patients

Aushadhi is an AI-powered drug interaction prediction system designed for patients
taking multiple medications simultaneously (polypharmacy). It combines a curated
database of 100+ common drugs with LLM-augmented reasoning to detect pairwise and
higher-order drug interactions, assess risk severity, and generate clinical reports.

## Features

- **Drug Database**: Built-in database of 100+ common medications with pharmacological
  properties, drug classes, mechanisms of action, and CYP450 enzyme profiles.
- **Interaction Detection**: Pairwise and higher-order interaction checking using known
  interaction pairs and mechanism-based inference (CYP450, pharmacodynamic synergy).
- **LLM-Augmented Prediction**: RAG pipeline retrieves relevant pharmacological context
  and feeds it to an LLM for reasoning about novel or under-documented combinations.
- **Risk Scoring**: Four-tier severity classification (mild / moderate / severe /
  contraindicated) with mechanism explanations and evidence levels.
- **Clinical Reports**: Rich terminal reports with medication lists, interaction matrices,
  and risk assessments formatted for quick clinical review.
- **CLI Interface**: Simple command-line interface via Click for checking interactions
  and generating reports.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Check interactions between medications
aushadhi check --drugs "warfarin,aspirin,omeprazole"

# Generate a full polypharmacy report
aushadhi report --drugs "metformin,lisinopril,atorvastatin,amlodipine,metoprolol"
```

## Usage from Python

```python
from aushadhi.drugs.database import DrugDatabase
from aushadhi.drugs.interactions import InteractionChecker
from aushadhi.risk_scorer import RiskScorer

db = DrugDatabase()
checker = InteractionChecker(db)
scorer = RiskScorer()

drugs = db.lookup_many(["warfarin", "aspirin", "omeprazole"])
interactions = checker.check_all(drugs)

for interaction in interactions:
    assessment = scorer.score(interaction)
    print(f"{assessment.severity}: {assessment.summary}")
```

## Project Structure

```
aushadhi/
  src/aushadhi/
    __init__.py
    cli.py            Click CLI entry points
    models.py          Pydantic data models
    predictor.py       LLM-augmented interaction predictor
    risk_scorer.py     Severity scoring and risk assessment
    report.py          Rich terminal report generation
    rag.py             RAG pipeline for knowledge retrieval
    drugs/
      __init__.py
      database.py      Drug database (100+ medications)
      interactions.py  Interaction detection engine
      classes.py       Drug classes, mechanisms, CYP450 enzymes
  tests/
  examples/
```

## Author

Mukunda Katta

## License

MIT
