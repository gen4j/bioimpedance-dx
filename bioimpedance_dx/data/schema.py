"""
bioimpedance_dx/data/schema.py

Data contract for the osteomyelitis bioimpedance dataset.
Every value that enters the ML pipeline must pass through this module.
If it doesn't pass validation, it never reaches the model.

This is a regulatory-grade design decision: the model must never receive
garbage input and produce a confident but meaningless diagnosis.
"""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Label definitions — single source of truth for all class names
# ---------------------------------------------------------------------------

class BoneStatus(str, Enum):
    """
    The five diagnostic classes the model predicts.
    String enum so values serialize cleanly to JSON for the API layer.
    """
    NORMAL = "Normal Tibia Bone Sample"
    TYPE_1 = "Chronic tibia bone sample type1 osteomyelites"
    TYPE_2 = "Chronic tibia bone sample type2 osteomyelites"
    TYPE_3 = "Chronic tibia bone sample type3 osteomyelites"
    TYPE_4 = "Chronic tibia bone sample type4 osteomyelites"


# Ordered from healthy to most severe — used for ordinal reasoning
SEVERITY_ORDER: list[BoneStatus] = [
    BoneStatus.NORMAL,
    BoneStatus.TYPE_1,
    BoneStatus.TYPE_2,
    BoneStatus.TYPE_3,
    BoneStatus.TYPE_4,
]

# Integer encoding for model training
LABEL_TO_INT: dict[BoneStatus, int] = {
    status: idx for idx, status in enumerate(SEVERITY_ORDER)
}

INT_TO_LABEL: dict[int, BoneStatus] = {
    idx: status for status, idx in LABEL_TO_INT.items()
}

# Human-readable severity descriptions for clinical output
SEVERITY_DESCRIPTION: dict[BoneStatus, str] = {
    BoneStatus.NORMAL: "No signs of osteomyelitis detected.",
    BoneStatus.TYPE_1: "Stage 1 chronic osteomyelitis — early infection indicators.",
    BoneStatus.TYPE_2: "Stage 2 chronic osteomyelitis — moderate infection present.",
    BoneStatus.TYPE_3: "Stage 3 chronic osteomyelitis — advanced infection.",
    BoneStatus.TYPE_4: "Stage 4 chronic osteomyelitis — severe chronic infection.",
}


# ---------------------------------------------------------------------------
# Feature schema — Pydantic model for a single bioimpedance measurement
# ---------------------------------------------------------------------------

# Physiologically plausible ranges for normalized features (0.0 to 1.0)
# These bounds enforce that the model only sees valid bioimpedance data.
# Out-of-range values indicate electrode detachment, equipment fault,
# or data corruption — all of which must be caught BEFORE inference.

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "frequency":   (0.0, 1.0),
    "impedance":   (0.0, 1.0),
    "phase":       (0.0, 1.0),
    "resistance":  (0.0, 1.0),
    "reactance":   (0.0, 1.0),
    "magnitude":   (0.0, 1.0),
}


class BioimpedanceSample(BaseModel):
    """
    A single validated bioimpedance measurement.

    All features are normalized to [0.0, 1.0] by the preprocessing pipeline
    before reaching this schema. Raw values must never enter the model.

    Field names map directly to the normalized dataset columns.
    """

    frequency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized measurement frequency (original: 2kHz–99kHz)",
    )
    impedance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized current magnitude I (Ω/cm²)",
    )
    phase: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized phase angle Ph (rad)",
    )
    resistance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized resistance R (Ω/cm²)",
    )
    reactance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized imaginary component IM (Ω/cm²)",
    )
    magnitude: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized impedance magnitude Mag (Ω/cm²)",
    )

    @model_validator(mode="after")
    def check_not_all_zero(self) -> "BioimpedanceSample":
        """
        A sample where every feature is exactly 0.0 is a dead measurement.
        This catches electrode disconnection or acquisition failure.
        """
        values = [
            self.frequency, self.impedance, self.phase,
            self.resistance, self.reactance, self.magnitude,
        ]
        if all(v == 0.0 for v in values):
            raise ValueError(
                "All features are zero — this indicates a failed measurement. "
                "Check electrode contact and acquisition hardware."
            )
        return self

    def to_numpy(self) -> np.ndarray:
        """Return features as a 1D numpy array in model input order."""
        return np.array([
            self.frequency,
            self.impedance,
            self.phase,
            self.resistance,
            self.reactance,
            self.magnitude,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Column name mapping — raw Excel columns → clean Python names
# ---------------------------------------------------------------------------

RAW_COLUMN_MAP: dict[str, str] = {
    "Frequency":       "frequency",
    "I(Ω/cm^2)":      "impedance",
    "I(Ω/cm^2)":      "impedance",
    "Ph(rad)":         "phase",
    "R(Ω/cm^2)":      "resistance",
    "R(Ω/cm^2)":      "resistance",
    "IM(Ω/cm^2":      "reactance",   # note: missing closing paren in source
    "IM(Ω/cm^2":      "reactance",   # note: missing closing paren in source
    "Mag(Ω/cm^2":     "magnitude",   # note: missing closing paren in source
    "Mag(Ω/cm^2":     "magnitude",   # note: missing closing paren in source
    "Status of bone":  "label",
}

FEATURE_COLUMNS: list[str] = [
    "frequency", "impedance", "phase",
    "resistance", "reactance", "magnitude",
]

LABEL_COLUMN: str = "label"


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or fails validation."""


def load_normalized_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the normalized bioimpedance dataset.

    Args:
        path: Path to the normalized Excel file.

    Returns:
        A tuple of (X, y) where:
            X: float32 array of shape (n_samples, 6) — feature matrix
            y: int64 array of shape (n_samples,) — integer class labels

    Raises:
        DataLoadError: If the file cannot be read, columns are missing,
                       labels are unrecognized, or features are out of range.
    """
    # 1. Load raw file
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except FileNotFoundError:
        raise DataLoadError(f"Dataset file not found: {path}")
    except Exception as e:
        raise DataLoadError(f"Failed to read dataset: {e}") from e

    # 2. Rename columns to clean names. RAW_COLUMN_MAP intentionally accepts
    # both common Unicode representations of the ohm symbol.
    df = df.rename(columns=RAW_COLUMN_MAP)

    # 3. Validate expected logical columns exist
    expected_cols = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise DataLoadError(
            f"Dataset is missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # 4. Validate labels — every label must be a known BoneStatus
    known_labels = {status.value for status in BoneStatus}
    unknown_labels = set(df[LABEL_COLUMN].unique()) - known_labels
    if unknown_labels:
        raise DataLoadError(
            f"Dataset contains unrecognized labels: {unknown_labels}\n"
            f"Expected one of: {known_labels}"
        )

    # 5. Drop rows with missing feature values and report
    initial_count = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"[DataLoader] Dropped {dropped} rows with missing feature values.")

    # 6. Validate feature ranges — all normalized values must be in [0.0, 1.0]
    for col in FEATURE_COLUMNS:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < 0.0 or col_max > 1.0:
            raise DataLoadError(
                f"Feature '{col}' contains out-of-range values: "
                f"min={col_min:.4f}, max={col_max:.4f}. "
                f"Expected range: [0.0, 1.0]. "
                f"Ensure you are loading the NORMALIZED dataset, not the raw one."
            )

    # 7. Build feature matrix and label vector
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = np.array([
        LABEL_TO_INT[BoneStatus(label)]
        for label in df[LABEL_COLUMN]
    ], dtype=np.int64)

    print(f"[DataLoader] Loaded {len(X)} samples, {X.shape[1]} features, "
          f"{len(set(y))} classes.")
    print("[DataLoader] Class distribution:")
    for label_int, count in zip(*np.unique(y, return_counts=True)):
        label = INT_TO_LABEL[label_int]
        print(f"  [{label_int}] {label.value}: {count} samples")

    return X, y
