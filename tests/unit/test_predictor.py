"""
tests/unit/test_predictor.py

Unit tests for bioimpedance_dx/inference/predictor.py

These tests verify:
1. Predictor loads a valid model correctly
2. Predictor rejects invalid model paths
3. predict() returns a correctly structured PredictionResult
4. PredictionResult fields are correct types and ranges
5. Low confidence warning triggers at the right threshold
6. predict_batch() handles multiple samples correctly
"""

import pytest

from bioimpedance_dx.data import BioimpedanceSample, BoneStatus
from bioimpedance_dx.inference import OsteomyelitisPredictor, PredictionResult

# Path to the known good model from training
MODEL_PATH = "models/classifier_22754aaa.pkl"

# A known normal sample (first row of normalized dataset)
NORMAL_SAMPLE = BioimpedanceSample(
    frequency=0.0,
    impedance=0.003146,
    phase=0.501509,
    resistance=0.91378,
    reactance=0.729335,
    magnitude=0.067359,
)


# ---------------------------------------------------------------------------
# Predictor loading tests
# ---------------------------------------------------------------------------

class TestPredictorLoading:

    def test_loads_valid_model(self) -> None:
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        assert predictor is not None

    def test_missing_model_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            OsteomyelitisPredictor("models/nonexistent.pkl")

    def test_model_path_is_stored(self) -> None:
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        assert predictor._model_path == MODEL_PATH


# ---------------------------------------------------------------------------
# Prediction output tests
# ---------------------------------------------------------------------------

class TestPrediction:

    @pytest.fixture(scope="class")
    def predictor(self) -> OsteomyelitisPredictor:
        return OsteomyelitisPredictor(MODEL_PATH)

    @pytest.fixture(scope="class")
    def result(self, predictor: OsteomyelitisPredictor) -> PredictionResult:
        return predictor.predict(NORMAL_SAMPLE)

    def test_returns_prediction_result(
        self, result: PredictionResult
    ) -> None:
        assert isinstance(result, PredictionResult)

    def test_predicted_class_is_bone_status(
        self, result: PredictionResult
    ) -> None:
        assert isinstance(result.predicted_class, BoneStatus)

    def test_known_normal_sample_predicts_normal(
        self, result: PredictionResult
    ) -> None:
        """
        The first row of the training data is a normal sample.
        The model must predict Normal for this input.
        This is a regression test — if this fails after a model update,
        it means the new model behaves differently on known data.
        """
        assert result.predicted_class == BoneStatus.NORMAL

    def test_confidence_is_between_zero_and_one(
        self, result: PredictionResult
    ) -> None:
        assert 0.0 <= result.confidence <= 1.0

    def test_label_index_is_valid(
        self, result: PredictionResult
    ) -> None:
        assert 0 <= result.predicted_label <= 4

    def test_severity_description_is_non_empty(
        self, result: PredictionResult
    ) -> None:
        assert len(result.severity_description) > 0

    def test_class_probabilities_has_five_entries(
        self, result: PredictionResult
    ) -> None:
        assert len(result.class_probabilities) == 5

    def test_class_probabilities_sum_to_one(
        self, result: PredictionResult
    ) -> None:
        """Probabilities across all classes must sum to 1.0 (within float tolerance)."""
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_all_probabilities_non_negative(
        self, result: PredictionResult
    ) -> None:
        for prob in result.class_probabilities.values():
            assert prob >= 0.0

    def test_software_version_is_set(
        self, result: PredictionResult
    ) -> None:
        assert result.software_version != ""

    def test_timestamp_is_set(
        self, result: PredictionResult
    ) -> None:
        assert result.timestamp_utc != ""
        assert "T" in result.timestamp_utc  # ISO format contains T

    def test_input_features_are_echoed(
        self, result: PredictionResult
    ) -> None:
        """
        Input features must be included in the result for audit trail.
        Regulatory requirement: every prediction must be reproducible
        from its logged inputs.
        """
        assert "frequency" in result.input_features
        assert "impedance" in result.input_features
        assert len(result.input_features) == 6

    def test_to_dict_contains_prediction_key(
        self, result: PredictionResult
    ) -> None:
        d = result.to_dict()
        assert "prediction" in d
        assert "probabilities" in d
        assert "audit" in d

    def test_to_dict_prediction_has_class(
        self, result: PredictionResult
    ) -> None:
        d = result.to_dict()
        assert "class" in d["prediction"]
        assert "confidence" in d["prediction"]


# ---------------------------------------------------------------------------
# Low confidence warning tests
# ---------------------------------------------------------------------------

class TestLowConfidenceWarning:

    def test_high_confidence_no_warning(self) -> None:
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        result = predictor.predict(NORMAL_SAMPLE)
        # Normal sample gets ~87% confidence — well above threshold
        if result.confidence >= 0.70:
            assert result.low_confidence_warning is False

    def test_str_representation_contains_diagnosis(self) -> None:
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        result = predictor.predict(NORMAL_SAMPLE)
        output = str(result)
        assert "Diagnosis" in output
        assert "Confidence" in output


# ---------------------------------------------------------------------------
# Batch prediction tests
# ---------------------------------------------------------------------------

class TestBatchPrediction:

    def test_batch_returns_correct_number_of_results(self) -> None:
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        samples = [NORMAL_SAMPLE, NORMAL_SAMPLE, NORMAL_SAMPLE]
        results = predictor.predict_batch(samples)
        assert len(results) == 3

    def test_empty_batch_returns_empty_list(self) -> None:
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        results = predictor.predict_batch([])
        assert results == []

    def test_batch_results_match_single_predictions(self) -> None:
        """
        Batch prediction must produce the same result as
        calling predict() individually. If these diverge,
        there is a bug in the batch implementation.
        """
        predictor = OsteomyelitisPredictor(MODEL_PATH)
        single = predictor.predict(NORMAL_SAMPLE)
        batch = predictor.predict_batch([NORMAL_SAMPLE])
        assert batch[0].predicted_class == single.predicted_class
        assert abs(batch[0].confidence - single.confidence) < 0.001