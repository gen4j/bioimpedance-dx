"""
tests/unit/test_schema.py

Unit tests for bioimpedance_dx/data/schema.py

These tests verify:
1. BoneStatus enum has all expected values
2. BioimpedanceSample rejects invalid inputs
3. BioimpedanceSample accepts valid inputs
4. load_normalized_dataset loads and validates correctly

In IEC 62304 Class C software, every safety-critical function
must have documented test cases. The data validator is safety-critical
because it is the first line of defense against bad data reaching the model.
"""

import numpy as np
import pytest

from bioimpedance_dx.data import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    SEVERITY_ORDER,
    BioimpedanceSample,
    BoneStatus,
    load_normalized_dataset,
)
from bioimpedance_dx.data.schema import DataLoadError


# ---------------------------------------------------------------------------
# BoneStatus enum tests
# ---------------------------------------------------------------------------

class TestBoneStatus:

    def test_all_five_classes_exist(self) -> None:
        """There must be exactly 5 diagnostic classes."""
        assert len(BoneStatus) == 5

    def test_normal_class_exists(self) -> None:
        assert BoneStatus.NORMAL.value == "Normal Tibia Bone Sample"

    def test_type1_class_exists(self) -> None:
        assert "type1" in BoneStatus.TYPE_1.value.lower()

    def test_type4_class_exists(self) -> None:
        assert "type4" in BoneStatus.TYPE_4.value.lower()

    def test_severity_order_starts_with_normal(self) -> None:
        """Normal must always be first in severity order."""
        assert SEVERITY_ORDER[0] == BoneStatus.NORMAL

    def test_severity_order_ends_with_type4(self) -> None:
        """Type 4 must always be last — most severe."""
        assert SEVERITY_ORDER[-1] == BoneStatus.TYPE_4

    def test_label_to_int_mapping_is_complete(self) -> None:
        """Every BoneStatus must have an integer encoding."""
        assert len(LABEL_TO_INT) == len(BoneStatus)

    def test_int_to_label_is_inverse_of_label_to_int(self) -> None:
        """Round-trip: label → int → label must return original."""
        for status in BoneStatus:
            idx = LABEL_TO_INT[status]
            assert INT_TO_LABEL[idx] == status

    def test_normal_encodes_to_zero(self) -> None:
        """Normal must encode to 0 — lowest severity index."""
        assert LABEL_TO_INT[BoneStatus.NORMAL] == 0

    def test_type4_encodes_to_four(self) -> None:
        """Type4 must encode to 4 — highest severity index."""
        assert LABEL_TO_INT[BoneStatus.TYPE_4] == 4


# ---------------------------------------------------------------------------
# BioimpedanceSample validation tests
# ---------------------------------------------------------------------------

VALID_SAMPLE_KWARGS = {
    "frequency": 0.5,
    "impedance": 0.003,
    "phase": 0.501,
    "resistance": 0.93,
    "reactance": 0.73,
    "magnitude": 0.07,
}


class TestBioimpedanceSample:

    def test_valid_sample_is_accepted(self) -> None:
        """A sample with all values in [0, 1] must pass validation."""
        sample = BioimpedanceSample(**VALID_SAMPLE_KWARGS)
        assert sample.frequency == 0.5

    def test_boundary_values_zero_and_one_are_accepted(self) -> None:
        """Exact boundary values 0.0 and 1.0 must be valid."""
        sample = BioimpedanceSample(
            frequency=0.0,
            impedance=1.0,
            phase=0.0,
            resistance=1.0,
            reactance=0.5,
            magnitude=0.5,
        )
        assert sample is not None

    def test_frequency_above_one_is_rejected(self) -> None:
        """Values above 1.0 are out of normalized range."""
        with pytest.raises(Exception):
            BioimpedanceSample(**{**VALID_SAMPLE_KWARGS, "frequency": 1.001})

    def test_frequency_below_zero_is_rejected(self) -> None:
        """Negative values are physically impossible in normalized data."""
        with pytest.raises(Exception):
            BioimpedanceSample(**{**VALID_SAMPLE_KWARGS, "frequency": -0.001})

    def test_impedance_out_of_range_is_rejected(self) -> None:
        with pytest.raises(Exception):
            BioimpedanceSample(**{**VALID_SAMPLE_KWARGS, "impedance": 1.5})

    def test_phase_out_of_range_is_rejected(self) -> None:
        with pytest.raises(Exception):
            BioimpedanceSample(**{**VALID_SAMPLE_KWARGS, "phase": -1.0})

    def test_all_zero_sample_is_rejected(self) -> None:
        """
        All-zero sample indicates electrode disconnection or hardware failure.
        This is a safety-critical check — a dead measurement must never
        reach the model and produce a confident but meaningless diagnosis.
        """
        with pytest.raises(Exception, match="failed measurement"):
            BioimpedanceSample(
                frequency=0.0,
                impedance=0.0,
                phase=0.0,
                resistance=0.0,
                reactance=0.0,
                magnitude=0.0,
            )

    def test_to_numpy_returns_correct_shape(self) -> None:
        """Feature array must be shape (6,) for the model."""
        sample = BioimpedanceSample(**VALID_SAMPLE_KWARGS)
        arr = sample.to_numpy()
        assert arr.shape == (6,)

    def test_to_numpy_returns_float32(self) -> None:
        """Model expects float32 input — not float64."""
        sample = BioimpedanceSample(**VALID_SAMPLE_KWARGS)
        arr = sample.to_numpy()
        assert arr.dtype == np.float32

    def test_to_numpy_preserves_feature_order(self) -> None:
        """Feature order must be deterministic and match training order."""
        sample = BioimpedanceSample(**VALID_SAMPLE_KWARGS)
        arr = sample.to_numpy()
        assert arr[0] == pytest.approx(VALID_SAMPLE_KWARGS["frequency"])
        assert arr[1] == pytest.approx(VALID_SAMPLE_KWARGS["impedance"])
        assert arr[2] == pytest.approx(VALID_SAMPLE_KWARGS["phase"])
        assert arr[3] == pytest.approx(VALID_SAMPLE_KWARGS["resistance"])
        assert arr[4] == pytest.approx(VALID_SAMPLE_KWARGS["reactance"])
        assert arr[5] == pytest.approx(VALID_SAMPLE_KWARGS["magnitude"])


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestLoadNormalizedDataset:

    DATA_PATH = "data/raw/Normalized_NEw_data.xlsx"

    def test_loads_correct_number_of_samples(self) -> None:
        X, y = load_normalized_dataset(self.DATA_PATH)
        assert len(X) == 6821

    def test_loads_correct_number_of_features(self) -> None:
        X, y = load_normalized_dataset(self.DATA_PATH)
        assert X.shape[1] == 6

    def test_loads_correct_number_of_classes(self) -> None:
        X, y = load_normalized_dataset(self.DATA_PATH)
        assert len(np.unique(y)) == 5

    def test_X_is_float32(self) -> None:
        X, y = load_normalized_dataset(self.DATA_PATH)
        assert X.dtype == np.float32

    def test_y_is_int64(self) -> None:
        X, y = load_normalized_dataset(self.DATA_PATH)
        assert y.dtype == np.int64

    def test_all_features_in_normalized_range(self) -> None:
        """Every feature value must be in [0.0, 1.0] in the normalized dataset."""
        X, _ = load_normalized_dataset(self.DATA_PATH)
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_labels_in_valid_range(self) -> None:
        """All integer labels must be in [0, 4]."""
        _, y = load_normalized_dataset(self.DATA_PATH)
        assert y.min() == 0
        assert y.max() == 4

    def test_class_balance(self) -> None:
        """
        Dataset is perfectly balanced — each class should have
        approximately equal samples. Allow ±5 sample tolerance.
        """
        _, y = load_normalized_dataset(self.DATA_PATH)
        counts = [int((y == i).sum()) for i in range(5)]
        assert max(counts) - min(counts) <= 5

    def test_missing_file_raises_data_load_error(self) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            load_normalized_dataset("data/raw/nonexistent_file.xlsx")

    def test_X_and_y_have_same_length(self) -> None:
        X, y = load_normalized_dataset(self.DATA_PATH)
        assert len(X) == len(y)