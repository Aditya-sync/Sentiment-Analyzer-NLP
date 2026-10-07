"""
FastAPI Backend for Sentiment Analyzer NLP

This backend serves:
1. V1 (Historical Baseline): Pre-trained scikit-learn classical NLP pipeline (TF-IDF + Logistic Regression) on IMDb reviews.
2. V2 (Current Production Model): Fine-tuned RoBERTa Transformer model (FacebookAI/roberta-base) on TweetEval sentiment (3-class: negative / neutral / positive).

Both models are loaded once on startup via the application lifespan context manager.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# 1. Path & Model Storage Configuration
# ============================================================================

# Resolve directory paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# V1 Classical model path
MODEL_PATH = (PROJECT_ROOT / "ml" / "models" / "sentiment_pipeline.pkl").resolve()

# V2 Transformer model path
V2_MODEL_PATH = PROJECT_ROOT / "ml_transformer" / "models" / "roberta-tweeteval"

# Ensure PROJECT_ROOT is on sys.path for importing ml_transformer modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_transformer.src.test_model import (
    load_inference_pipeline,
    predict_sentiment as predict_v2_sentiment,
)

# Global variables for model references in memory
ml_model: Optional[Any] = None
v2_model: Optional[Any] = None
v2_tokenizer: Optional[Any] = None
v2_model_loaded: bool = False


# ============================================================================
# 2. Application Lifespan (Load Models ONCE at Startup)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    - Runs ONCE when the FastAPI server starts: loads V1 and V2 models into memory.
    - Yields control to handle incoming requests.
    - Runs cleanup when the server shuts down.
    """
    global ml_model, v2_model, v2_tokenizer, v2_model_loaded
    print("[*] Starting FastAPI server...")

    # --- Load V1 Classical Model ---
    print(f"[*] Loading V1 sentiment model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        error_msg = f"V1 Model file not found at: {MODEL_PATH}"
        print(f"[!] ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)

    try:
        # Load the scikit-learn pipeline (TF-IDF + LogisticRegression)
        ml_model = joblib.load(MODEL_PATH)
        print("[+] V1 Classical model loaded successfully and ready for inference!")
    except Exception as exc:
        print(f"[!] Failed to load V1 model: {exc}")
        raise exc

    # --- Load V2 Transformer Model ---
    print(f"[*] Loading V2 RoBERTa model from: {V2_MODEL_PATH}")
    try:
        v2_model, v2_tokenizer, _ = load_inference_pipeline(V2_MODEL_PATH)
        v2_model_loaded = True
        print("[+] V2 RoBERTa model loaded successfully and ready for inference!")
    except Exception as exc:
        print(f"[!] WARNING: Failed to load V2 RoBERTa model: {exc}")
        print("[!] V1 will remain operational, but V2 endpoints will be unavailable.")
        v2_model = None
        v2_tokenizer = None
        v2_model_loaded = False

    yield  # Application handles incoming requests while yielded

    # Clean up on shutdown
    print("[*] Shutting down FastAPI server and releasing model resources...")
    ml_model = None
    v2_model = None
    v2_tokenizer = None
    v2_model_loaded = False


# ============================================================================
# 3. FastAPI App Initialization & CORS Setup
# ============================================================================

app = FastAPI(
    title="Sentiment Analyzer NLP API",
    description="REST API for classifying sentiment using Classical NLP (V1) and Transformer (V2) pipelines.",
    version="2.0.0",
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
        description="The text to analyze",
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
    """Response schema for V1 classical sentiment prediction."""
    text: str
    sentiment: str
    confidence: float


class PredictV2Response(BaseModel):
    """Response schema for V2 Transformer sentiment prediction."""
    text: str
    sentiment: str
    confidence: float
    negative_probability: float
    neutral_probability: float
    positive_probability: float


class HealthResponse(BaseModel):
    """Response schema for API health check."""
    status: str
    model_loaded: bool
    v2_model_loaded: bool


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
        "health_check": "/health",
        "v1_endpoint": "/predict",
        "v2_endpoint": "/predict/v2",
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
    whether both ML pipelines have been loaded into memory.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=(ml_model is not None),
        v2_model_loaded=v2_model_loaded,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Sentiment Analysis (V1 - Classical Baseline)"],
    summary="Predict Sentiment using V1 Classical Pipeline (Baseline)"
)
async def predict_sentiment(request: PredictRequest):
    """
    [Baseline / Historical Endpoint]
    Classifies the sentiment of the provided text into 'positive' or 'negative'
    using the scikit-learn pipeline (TF-IDF + Logistic Regression).
    """
    # Guard: Ensure V1 model is loaded
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The V1 ML model is not loaded. Please check server logs."
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
            detail=f"An error occurred during V1 model inference: {str(exc)}"
        )


@app.post(
    "/predict/v2",
    response_model=PredictV2Response,
    tags=["Sentiment Analysis (V2 - Transformer Production)"],
    summary="Predict Sentiment using V2 RoBERTa Transformer Pipeline (Primary)"
)
async def predict_sentiment_v2(request: PredictRequest):
    """
    [Primary Production Endpoint]
    Classifies the sentiment of the provided text into 'negative', 'neutral', or 'positive'
    using the fine-tuned RoBERTa Transformer model, returning individual class probabilities.
    """
    # Guard: Ensure V2 model is loaded
    if not v2_model_loaded or v2_model is None or v2_tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The V2 Transformer model is not loaded. Please check server logs."
        )

    try:
        # Run forward pass inference with real model logits and softmax
        result = predict_v2_sentiment(request.text, v2_model, v2_tokenizer)

        # Convert probabilities to percentages rounded to 2 decimal places
        neg_pct = round(result["negative_prob"] * 100, 2)
        neu_pct = round(result["neutral_prob"] * 100, 2)
        pos_pct = round(result["positive_prob"] * 100, 2)

        # Confidence is the probability percentage of the predicted class
        probs_by_class = {
            "negative": neg_pct,
            "neutral": neu_pct,
            "positive": pos_pct,
        }
        sentiment = result["predicted_class"]
        confidence = probs_by_class.get(sentiment, round(max(neg_pct, neu_pct, pos_pct), 2))

        return PredictV2Response(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            negative_probability=neg_pct,
            neutral_probability=neu_pct,
            positive_probability=pos_pct,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during V2 model inference: {str(exc)}"
        )
