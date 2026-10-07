"""
FastAPI Backend for Sentiment Analyzer NLP

This backend serves the pre-trained scikit-learn sentiment pipeline (TF-IDF + Logistic Regression).
The model is loaded once on startup and handles prediction requests via a REST API.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# 1. Path & Model Storage Configuration
# ============================================================================

# Resolve the absolute path to the pre-trained ML model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR.parent / "ml" / "models" / "sentiment_pipeline.pkl").resolve()

# Global variable to hold the loaded pipeline in memory
ml_model: Optional[Any] = None


# ============================================================================
# 2. Application Lifespan (Load Model ONCE at Startup)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    - Runs ONCE when the FastAPI server starts: loads the trained ML model into memory.
    - Yields control to handle incoming requests.
    - Runs cleanup when the server shuts down.
    """
    global ml_model
    print(f"[*] Starting FastAPI server...")
    print(f"[*] Loading sentiment model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        error_msg = f"Model file not found at: {MODEL_PATH}"
        print(f"[!] ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)

    try:
        # Load the scikit-learn pipeline (TF-IDF + LogisticRegression)
        ml_model = joblib.load(MODEL_PATH)
        print("[+] Model loaded successfully and ready for inference!")
    except Exception as exc:
        print(f"[!] Failed to load model: {exc}")
        raise exc

    yield  # Application handles incoming requests while yielded

    # Clean up on shutdown
    print("[*] Shutting down FastAPI server and releasing model resources...")
    ml_model = None


# ============================================================================
# 3. FastAPI App Initialization & CORS Setup
# ============================================================================

app = FastAPI(
    title="Sentiment Analyzer NLP API",
    description="REST API for classifying review sentiment using a trained scikit-learn NLP pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 4. Pydantic Request & Response Schemas
# ============================================================================

class PredictRequest(BaseModel):
    """Request schema for sentiment prediction."""
    text: str = Field(
        ...,
        description="The text review to analyze",
        examples=["I absolutely loved this movie!"]
    )

    @field_validator("text")
    @classmethod
    def validate_not_empty_or_whitespace(cls, value: str) -> str:
        """Reject empty strings or strings that contain only whitespace."""
        if not value or not value.strip():
            raise ValueError("Text cannot be empty or contain only whitespace.")
        return value


class PredictResponse(BaseModel):
    """Response schema for sentiment prediction."""
    text: str
    sentiment: str
    confidence: float


class HealthResponse(BaseModel):
    """Response schema for API health check."""
    status: str
    model_loaded: bool


# ============================================================================
# 5. API Endpoints
# ============================================================================

@app.get(
    "/",
    tags=["Root"],
    summary="Root Welcome Endpoint"
)
async def root():
    """Welcome endpoint providing basic API information and links to docs."""
    return {
        "message": "Sentiment Analyzer NLP API is running.",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health Check"],
    summary="Check API and Model Status"
)
async def health_check():
    """
    Returns the operational status of the service and indicates
    whether the ML pipeline has been successfully loaded into memory.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=(ml_model is not None)
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Sentiment Analysis"],
    summary="Predict Sentiment of a Text Review"
)
async def predict_sentiment(request: PredictRequest):
    """
    Classifies the sentiment of the provided text into 'positive' or 'negative'
    along with a confidence score (percentage rounded to 2 decimal places).
    """
    # Guard: Ensure model is loaded
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ML model is not loaded. Please check server logs."
        )

    try:
        # Pass the input text directly into the pipeline as an iterable of strings
        # model.predict() returns class array: e.g. [1] or [0]
        prediction_arr = ml_model.predict([request.text])
        prediction_class = int(prediction_arr[0])

        # model.predict_proba() returns probability matrix: e.g. [[0.05, 0.95]]
        probabilities = ml_model.predict_proba([request.text])
        sample_probs = probabilities[0]

        # Convert prediction: 1 -> "positive", 0 -> "negative"
        sentiment = "positive" if prediction_class == 1 else "negative"

        # Confidence: highest predicted probability multiplied by 100, rounded to 2 decimal places
        highest_probability = float(max(sample_probs))
        confidence = round(highest_probability * 100, 2)

        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during model inference: {str(exc)}"
        )
