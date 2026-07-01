"""
tests/integration/test_api.py

Integration tests for the FastAPI inference server.

These tests verify the full HTTP request/response cycle:
- Correct status codes
- Correct response structure
- Input validation at the API layer
- Health check reflects real system state

We use FastAPI's TestClient which runs the app in-process
without needing a real server running — fast and deterministic.
"""

import pytest
from fastapi.testclient import TestClient

from bioimpedance_dx.api.app import app
from bioimpedance_dx.data import BoneStatus, SEVERITY_ORDER

# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Create a TestClient for the FastAPI app.
    scope="module" means one client is shared across all tests in this file
    — the model is loaded once, not once per test.
    """
    with TestClient(app) as c:
        yield c


# Valid sample payload — first row of normalized dataset (Normal)
VALID_PAYLOAD = {
    "frequency": 0.0,
    "impedance": 0.003146,
    "phase": 0.501509,
    "resistance": 0.91378,
    "reactance": 0.729335,
    "magnitude": 0.067359,
}


# ---------------------------------------------------------------------------
# /health endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_model_is_loaded(self, client: TestClient) -> None:
        """Model must be loaded at startup — not on first request."""
        response = client.get("/health")
        assert response.json()["model_loaded"] is True

    def test_health_returns_correct_class_count(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["n_classes"] == 5

    def test_health_returns_api_version(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "api_version" in response.json()

    def test_health_returns_all_class_names(self, client: TestClient) -> None:
        response = client.get("/health")
        classes = response.json()["classes"]
        assert len(classes) == 5
        assert "Normal Tibia Bone Sample" in classes


# ---------------------------------------------------------------------------
# /classes endpoint tests
# ---------------------------------------------------------------------------

class TestClassesEndpoint:

    def test_classes_returns_200(self, client: TestClient) -> None:
        response = client.get("/classes")
        assert response.status_code == 200

    def test_classes_returns_five_items(self, client: TestClient) -> None:
        response = client.get("/classes")
        assert len(response.json()["classes"]) == 5

    def test_classes_first_item_is_normal(self, client: TestClient) -> None:
        response = client.get("/classes")
        first = response.json()["classes"][0]
        assert first["index"] == 0
        assert first["value"] == BoneStatus.NORMAL.value

    def test_classes_last_item_is_type4(self, client: TestClient) -> None:
        response = client.get("/classes")
        last = response.json()["classes"][-1]
        assert last["index"] == 4
        assert "type4" in last["value"].lower()

    def test_classes_indices_are_sequential(self, client: TestClient) -> None:
        response = client.get("/classes")
        indices = [c["index"] for c in response.json()["classes"]]
        assert indices == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# /predict endpoint — valid input tests
# ---------------------------------------------------------------------------

class TestPredictEndpointValidInput:

    def test_predict_returns_200_for_valid_input(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_returns_predicted_class(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "predicted_class" in response.json()

    def test_predict_known_normal_sample(self, client: TestClient) -> None:
        """
        The first row of the dataset is a Normal sample.
        The model must classify it as Normal.
        This is a regression test — a model update that changes
        this result must be explicitly reviewed.
        """
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.json()["predicted_class"] == BoneStatus.NORMAL.value

    def test_predict_confidence_is_float_between_0_and_1(
        self, client: TestClient
    ) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        confidence = response.json()["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_returns_five_class_probabilities(
        self, client: TestClient
    ) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        probs = response.json()["class_probabilities"]
        assert len(probs) == 5

    def test_predict_probabilities_sum_to_one(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        probs = response.json()["class_probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_returns_severity_description(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        desc = response.json()["severity_description"]
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_predict_returns_timestamp(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        ts = response.json()["timestamp_utc"]
        assert "T" in ts  # ISO 8601 format

    def test_predict_returns_software_version(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "software_version" in response.json()

    def test_predict_echoes_input_features(self, client: TestClient) -> None:
        """
        Input features must be returned in the response for audit trail.
        Regulatory requirement: every prediction must be reproducible
        from its logged inputs alone.
        """
        response = client.post("/predict", json=VALID_PAYLOAD)
        features = response.json()["input_features"]
        assert len(features) == 6
        assert "frequency" in features
        assert "impedance" in features

    def test_predict_low_confidence_warning_is_bool(
        self, client: TestClient
    ) -> None:
        response = client.post("/predict", json=VALID_PAYLOAD)
        warning = response.json()["low_confidence_warning"]
        assert isinstance(warning, bool)


# ---------------------------------------------------------------------------
# /predict endpoint — invalid input tests
# ---------------------------------------------------------------------------

class TestPredictEndpointInvalidInput:

    def test_frequency_above_one_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "frequency": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_frequency_below_zero_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "frequency": -0.1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "magnitude"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_all_zero_payload_returns_422(self, client: TestClient) -> None:
        """
        All-zero measurement indicates hardware failure.
        Must be rejected before reaching the model.
        """
        zero_payload = {k: 0.0 for k in VALID_PAYLOAD}
        response = client.post("/predict", json=zero_payload)
        assert response.status_code == 422

    def test_string_instead_of_float_returns_422(self, client: TestClient) -> None:
        payload = {**VALID_PAYLOAD, "frequency": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Non-existent routes
# ---------------------------------------------------------------------------

class TestNotFound:

    def test_root_returns_404(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 404

    def test_unknown_route_returns_404(self, client: TestClient) -> None:
        response = client.get("/diagnose")
        assert response.status_code == 404