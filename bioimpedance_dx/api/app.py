"""
bioimpedance_dx/api/app.py

FastAPI inference server for the osteomyelitis bioimpedance classifier.

This is the public interface of the system. Every external client —
mobile app, web dashboard, hospital integration, future cloud backend —
communicates with the model through this API.

Design principles:
- Input validation happens at the Pydantic layer before any business logic
- The predictor is loaded once at startup, not on every request
- Every prediction response includes the full audit trail
- Errors return structured JSON, never raw Python tracebacks
- The /health endpoint allows infrastructure to monitor the service

To run locally:
    poetry run uvicorn bioimpedance_dx.api.app:app --reload --port 8000

Then open: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bioimpedance_dx.data import BioimpedanceSample, BoneStatus, SEVERITY_ORDER
from bioimpedance_dx.inference import OsteomyelitisPredictor, PredictionResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "")
API_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Request / Response schemas
# (separate from the internal BioimpedanceSample — the API schema
#  is what external clients see; the internal schema is what the model sees)
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    """
    A single bioimpedance measurement submitted for diagnosis.
    All values must be normalized to [0.0, 1.0].
    """
    frequency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized measurement frequency (original range: 2kHz–99kHz)",
        examples=[0.0]
    )
    impedance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized current magnitude I (Ω/cm²)",
        examples=[0.003146]
    )
    phase: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized phase angle Ph (rad)",
        examples=[0.501509]
    )
    resistance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized resistance R (Ω/cm²)",
        examples=[0.91378]
    )
    reactance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized imaginary component IM (Ω/cm²)",
        examples=[0.729335]
    )
    magnitude: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized impedance magnitude Mag (Ω/cm²)",
        examples=[0.067359]
    )


class PredictionResponse(BaseModel):
    """Structured diagnosis response returned to the client."""
    predicted_class: str
    label_index: int
    confidence: float
    severity_description: str
    low_confidence_warning: bool
    class_probabilities: dict[str, float]
    software_version: str
    model_path: str
    timestamp_utc: str
    input_features: dict[str, float]


class HealthResponse(BaseModel):
    """Health check response for infrastructure monitoring."""
    status: str
    api_version: str
    model_loaded: bool
    model_path: str
    n_classes: int
    classes: list[str]


class ErrorResponse(BaseModel):
    """Structured error response — never expose raw tracebacks."""
    error: str
    detail: str


# ---------------------------------------------------------------------------
# Application state — model loaded once at startup
# ---------------------------------------------------------------------------

class AppState:
    predictor: OsteomyelitisPredictor | None = None


state = AppState()


def _find_latest_model() -> str:
    """Find the most recently created model file in models/."""
    models_dir = Path("models")
    if not models_dir.exists():
        return ""
    pkl_files = sorted(
        models_dir.glob("*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(pkl_files[0]) if pkl_files else ""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Load the model once when the server starts.
    This avoids loading the model on every request — critical for performance.
    """
    model_path = MODEL_PATH or _find_latest_model()

    if not model_path:
        print("[API] WARNING: No model file found. "
              "Run 'poetry run python scripts/train.py' first.")
        print("[API] Server starting without a loaded model. "
              "/predict will return 503 until a model is available.")
    else:
        try:
            state.predictor = OsteomyelitisPredictor(model_path)
            print(f"[API] Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"[API] ERROR loading model: {e}")
            print("[API] Server starting without a loaded model.")

    yield  # server runs here

    # Cleanup on shutdown
    state.predictor = None
    print("[API] Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Osteomyelitis Bioimpedance Diagnostic API",
    description=(
        "AI-powered bone infection severity classifier based on bioimpedance spectroscopy. "
        "Classifies bone samples into 5 categories: Normal, and 4 stages of chronic osteomyelitis."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# CORS — allows web frontends to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict to specific domains in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns server status and model information. "
                "Use this endpoint for infrastructure monitoring.",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model_loaded=state.predictor is not None,
        model_path=state.predictor._model_path if state.predictor else "none",
        n_classes=len(SEVERITY_ORDER),
        classes=[s.value for s in SEVERITY_ORDER],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify bone infection severity",
    description=(
        "Submit a normalized bioimpedance measurement and receive a diagnosis. "
        "All feature values must be normalized to [0.0, 1.0]. "
        "Returns predicted class, confidence score, full probability distribution, "
        "and a complete audit trail."
    ),
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"description": "Input validation error — feature out of range"},
    },
)
async def predict(request: InferenceRequest) -> PredictionResponse:
    # Check model is loaded
    if state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training pipeline first.",
        )

    # Build validated internal sample
    # This is the second validation layer — the API schema (above) catches
    # range errors, but BioimpedanceSample catches semantic errors
    # (e.g. all-zero measurements indicating hardware failure)
    try:
        sample = BioimpedanceSample(
            frequency=request.frequency,
            impedance=request.impedance,
            phase=request.phase,
            resistance=request.resistance,
            reactance=request.reactance,
            magnitude=request.magnitude,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # Run inference
    result: PredictionResult = state.predictor.predict(sample)

    return PredictionResponse(
        predicted_class=result.predicted_class.value,
        label_index=result.predicted_label,
        confidence=result.confidence,
        severity_description=result.severity_description,
        low_confidence_warning=result.low_confidence_warning,
        class_probabilities=result.class_probabilities,
        software_version=result.software_version,
        model_path=result.model_path,
        timestamp_utc=result.timestamp_utc,
        input_features=result.input_features,
    )


@app.get(
    "/classes",
    summary="List diagnostic classes",
    description="Returns all possible diagnostic classes in severity order.",
)
async def classes() -> dict:
    return {
        "classes": [
            {
                "index": i,
                "value": status.value,
            }
            for i, status in enumerate(SEVERITY_ORDER)
        ]
    }