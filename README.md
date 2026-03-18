# aushadhi

**Aushadhi — AI Drug Interaction Predictor. LLM-augmented multi-drug interaction prediction for polypharmacy safety.**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Aushadhi
 instance = Aushadhi()
r = instance.track(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `track()` | Track |
| `predict()` | Predict |
| `forecast()` | Forecast |
| `alert()` | Alert |
| `get_history()` | Get history |
| `visualize()` | Visualize |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.
