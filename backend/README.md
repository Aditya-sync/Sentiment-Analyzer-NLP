# Sentiment Analyzer NLP - FastAPI Backend

A lightweight, high-performance REST API built with [FastAPI](https://fastapi.tiangolo.com/) that serves a pre-trained scikit-learn Machine Learning pipeline (`TF-IDF Vectorizer` + `Logistic Regression`) for sentiment analysis.

---

## Features

- **Efficient Startup Model Loading**: The model is loaded into memory once on application startup using FastAPI's `lifespan` handler, ensuring zero model-loading overhead during incoming API requests.
- **Strict Input Validation**: Validates requests using Pydantic, instantly rejecting empty or whitespace-only inputs with informative error messages.
- **Standardized Inference**: Emits predictions (`positive` / `negative`) along with a confidence percentage rounded to two decimal places.
- **CORS Configured**: Pre-configured to allow cross-origin requests from the Next.js frontend running locally at `http://localhost:3000`.
- **Automated Interactive Docs**: Swagger UI documentation is available out of the box at `/docs`.

---

## Directory Structure

```text
backend/
├── main.py           # FastAPI application, lifespan handler, schemas, and endpoints
├── requirements.txt  # Python package dependencies
└── README.md         # Setup and testing instructions
```

---

## Setup & Installation

### 1. Create a Python Virtual Environment

From the root project directory or the `backend/` folder, create an isolated virtual environment:

```bash
# Navigate to backend directory
cd backend

# Create a virtual environment named .venv
python3 -m venv .venv
```

Activate the virtual environment:

- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

---

### 2. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 3. Start the FastAPI Server

Run the server with Uvicorn in development mode (with auto-reload enabled):

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

> **Note**: If running from the root project directory instead of `backend/`, run:
> ```bash
> uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
> ```

You should see logs indicating the model was loaded once on startup:

```text
[*] Starting FastAPI server...
[*] Loading sentiment model from: .../ml/models/sentiment_pipeline.pkl
[+] Model loaded successfully and ready for inference!
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Interactive API Documentation (Swagger)

FastAPI automatically generates interactive OpenAPI documentation. Once your server is running, open your browser and navigate to:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

You can test both `/health` and `/predict` directly inside the browser using Swagger UI.

---

## API Endpoints & Testing

### 1. Health Check: `GET /health`

Verifies that the server is operational and confirms whether the model pipeline is loaded in memory.

#### Example Request:
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Example Response (`200 OK`):
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 2. Sentiment Prediction: `POST /predict`

Analyzes the sentiment of a text review.

#### Request Body Schema:
```json
{
  "text": "string (non-empty)"
}
```

#### Successful Request (Positive Sentiment):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this movie!"}'
```

#### Response (`200 OK`):
```json
{
  "text": "I absolutely loved this movie!",
  "sentiment": "positive",
  "confidence": 94.76
}
```

---

#### Successful Request (Negative Sentiment):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was one of the worst movies I have ever watched."}'
```

#### Response (`200 OK`):
```json
{
  "text": "This was one of the worst movies I have ever watched.",
  "sentiment": "negative",
  "confidence": 99.08
}
```

---

#### Validation Error (Empty or Whitespace-Only Text):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "   "}'
```

#### Response (`422 Unprocessable Entity`):
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Value error, Text cannot be empty or contain only whitespace.",
      "input": "   "
    }
  ]
}
```
