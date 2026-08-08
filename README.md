# bioimpedance-dx

AI-powered bone infection severity classifier based on bioimpedance spectroscopy.

Classifies tibia bone samples into 5 diagnostic categories:
- Normal Tibia Bone Sample
- Chronic Osteomyelitis Type 1 (early)
- Chronic Osteomyelitis Type 2
- Chronic Osteomyelitis Type 3
- Chronic Osteomyelitis Type 4 (severe)

**Status:** Research prototype — not approved for clinical use.

---

## Requirements

- Python 3.12.3 (via pyenv)
- Poetry

## Setup

```bash
git clone https://github.com/gen4j/bioimpedance-dx.git
cd bioimpedance-dx
poetry install
```

## Train the model

```bash
poetry run python scripts/train.py
```

Trains a RandomForest classifier on the normalized bioimpedance dataset.
All experiments are tracked with MLflow.

View experiment results:
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Run the API server

```bash
poetry run uvicorn bioimpedance_dx.api.app:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Run tests

```bash
poetry run pytest tests/ -v
```

82 tests covering data validation, inference, and API endpoints.

## Input format

All features must be normalized to [0.0, 1.0]:

| Feature | Description |
|---------|-------------|
| frequency | Measurement frequency (2kHz–99kHz) |
| impedance | Current magnitude I (Ω/cm²) |
| phase | Phase angle Ph (rad) |
| resistance | Resistance R (Ω/cm²) |
| reactance | Imaginary component IM (Ω/cm²) |
| magnitude | Impedance magnitude Mag (Ω/cm²) |

## Model performance (v0.1.0)

| Metric | Value |
|--------|-------|
| Accuracy | 95.7% |
| Macro AUC | 0.9965 |
| Macro Sensitivity | 95.7% |
| Macro Specificity | 98.9% |

---

*This software is under active development toward IEC 62304 compliance.* 