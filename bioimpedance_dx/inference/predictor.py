"""
bioimpedance_dx/inference/predictor.py

The inference module — the bridge between a trained model and the real world.

This module is the only place in the codebase allowed to call model.predict().
Everything else — the API, the CLI, future mobile clients — goes through here.

Why centralize inference?
- Single place to enforce input validation before prediction
- Single place to enforce output formatting
- Single place to add audit logging (required for IEC 62304 / FDA traceability)
- If the model is swapped, only this file changes

Regulatory note: Every prediction made by this module must be logged with:
  - timestamp
  - software version
  - model version
  - raw input features
  - output class + confidence
  - any quality warnings
This audit trail is a post-market surveillance requirement under FDA and EU MDR.
"""

import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from bioimpedance_dx.data import (
    INT_TO_LABEL,
    SEVERITY_DESCRIPTION,
    SEVERITY_ORDER,
    BioimpedanceSample,
    BoneStatus,
)

# Software version — must be updated with every release.
# In a regulatory submission, this string appears in the audit log
# and must match the version in your Software Configuration Item list.
SOFTWARE_VERSION = "0.1.0-dev"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    The structured output of a single inference call.

    Every field here has a clinical or regulatory reason to exist.
    Nothing is returned to the caller that isn't documented.
    """

    # Core prediction
    predicted_class: BoneStatus
    predicted_label: int
    confidence: float                    # probability of predicted class [0.0, 1.0]
    severity_description: str           # human-readable clinical interpretation

    # Full probability distribution across all 5 classes
    # Required for: calibration analysis, uncertainty quantification,
    # ROC curve computation in post-market surveillance
    class_probabilities: dict[str, float]

    # Quality and audit fields
    software_version: str
    model_path: str
    timestamp_utc: str

    # Input echo — logged for full audit trail
    input_features: dict[str, float]

    # Clinical safety flag
    # If confidence is below this threshold, the system should
    # flag the result for human review rather than acting autonomously
    low_confidence_warning: bool = False
    LOW_CONFIDENCE_THRESHOLD: float = field(default=0.70, repr=False)

    def __post_init__(self) -> None:
        self.low_confidence_warning = self.confidence < self.LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        """Serialize to dictionary for API response or logging."""
        return {
            "prediction": {
                "class": self.predicted_class.value,
                "label_index": self.predicted_label,
                "confidence": round(self.confidence, 4),
                "severity_description": self.severity_description,
                "low_confidence_warning": self.low_confidence_warning,
            },
            "probabilities": self.class_probabilities,
            "audit": {
                "software_version": self.software_version,
                "model_path": self.model_path,
                "timestamp_utc": self.timestamp_utc,
                "input_features": self.input_features,
            },
        }

    def __str__(self) -> str:
        warning = " ⚠ LOW CONFIDENCE — REFER FOR CLINICAL REVIEW" \
            if self.low_confidence_warning else ""
        return (
            f"Diagnosis:   {self.predicted_class.value}{warning}\n"
            f"Confidence:  {self.confidence*100:.1f}%\n"
            f"Description: {self.severity_description}\n"
            f"Timestamp:   {self.timestamp_utc}"
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class ModelNotLoadedError(Exception):
    """Raised when inference is attempted before a model is loaded."""


class OsteomyelitisPredictor:
    """
    Loads a trained classifier and runs validated inference.

    Usage:
        predictor = OsteomyelitisPredictor("models/classifier_abc123.pkl")
        sample = BioimpedanceSample(
            frequency=0.5,
            impedance=0.003,
            phase=0.501,
            resistance=0.93,
            reactance=0.73,
            magnitude=0.07,
        )
        result = predictor.predict(sample)
        print(result)
    """

    def __init__(self, model_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to a .pkl file produced by scripts/train.py

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the loaded file is not a valid sklearn classifier.
        """
        self._model_path = str(model_path)
        self._model = self._load_model(model_path)
        self._class_names = [status.value for status in SEVERITY_ORDER]

    def _load_model(self, path: str) -> object:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run 'poetry run python scripts/train.py' to train a model first."
            )

        with open(resolved, "rb") as f:
            model = pickle.load(f)

        # Basic sanity check — the loaded object must have predict/predict_proba
        if not (hasattr(model, "predict") and hasattr(model, "predict_proba")):
            raise ValueError(
                f"Loaded file does not appear to be a valid sklearn classifier: {path}"
            )

        print(f"[Predictor] Model loaded: {path}")
        return model

    def predict(self, sample: BioimpedanceSample) -> PredictionResult:
        """
        Run inference on a single validated bioimpedance measurement.

        Args:
            sample: A validated BioimpedanceSample instance.
                    Input validation happens in BioimpedanceSample —
                    if it reaches here, it passed all boundary checks.

        Returns:
            PredictionResult with prediction, confidence, probabilities,
            and full audit trail fields.
        """
        if self._model is None:
            raise ModelNotLoadedError("No model loaded. Call __init__ with a valid model path.")

        # Convert validated sample to numpy array
        X = sample.to_numpy().reshape(1, -1)   # shape: (1, 6)

        # Run inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = int(self._model.predict(X)[0])
            y_prob = self._model.predict_proba(X)[0]  # shape: (5,)

        # Build output
        predicted_class = INT_TO_LABEL[y_pred]
        confidence = float(y_prob[y_pred])

        class_probabilities = {
            status.value: round(float(prob), 4)
            for status, prob in zip(SEVERITY_ORDER, y_prob)
        }

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_label=y_pred,
            confidence=confidence,
            severity_description=SEVERITY_DESCRIPTION[predicted_class],
            class_probabilities=class_probabilities,
            software_version=SOFTWARE_VERSION,
            model_path=self._model_path,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_features={
                "frequency": sample.frequency,
                "impedance": sample.impedance,
                "phase": sample.phase,
                "resistance": sample.resistance,
                "reactance": sample.reactance,
                "magnitude": sample.magnitude,
            },
        )

    def predict_batch(
        self,
        samples: list[BioimpedanceSample],
    ) -> list[PredictionResult]:
        """
        Run inference on a list of validated samples.
        More efficient than calling predict() in a loop for large batches.
        """
        if not samples:
            return []

        X = np.stack([s.to_numpy() for s in samples])  # shape: (n, 6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_preds = self._model.predict(X)
            y_probs = self._model.predict_proba(X)

        results = []
        for i, sample in enumerate(samples):
            y_pred = int(y_preds[i])
            y_prob = y_probs[i]
            predicted_class = INT_TO_LABEL[y_pred]
            confidence = float(y_prob[y_pred])

            results.append(PredictionResult(
                predicted_class=predicted_class,
                predicted_label=y_pred,
                confidence=confidence,
                severity_description=SEVERITY_DESCRIPTION[predicted_class],
                class_probabilities={
                    status.value: round(float(prob), 4)
                    for status, prob in zip(SEVERITY_ORDER, y_prob)
                },
                software_version=SOFTWARE_VERSION,
                model_path=self._model_path,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                input_features={
                    "frequency": sample.frequency,
                    "impedance": sample.impedance,
                    "phase": sample.phase,
                    "resistance": sample.resistance,
                    "reactance": sample.reactance,
                    "magnitude": sample.magnitude,
                },
            ))

        return results