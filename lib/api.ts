import { SentimentResult } from "@/types/sentiment";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Service to analyze sentiment from input text via the FastAPI backend.
 * Calls the trained ML pipeline (TF-IDF + Logistic Regression) at POST /predict.
 */
export async function analyzeSentiment(text: string): Promise<SentimentResult> {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Text cannot be empty");
  }

  let data: { text: string; sentiment: "positive" | "negative" | "neutral"; confidence: number };

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: trimmed }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => null);
      throw new Error(err?.detail || `API error: ${response.status} ${response.statusText}`);
    }

    data = await response.json();
  } catch (err: any) {
    if (err.message?.includes("Failed to fetch") || err.name === "TypeError") {
      throw new Error(
        "Could not connect to the ML Backend at http://localhost:8000. Please ensure your FastAPI server is running."
      );
    }
    throw err;
  }

  // Normalize confidence to 0-1 scale if returned as 0-100 percentage
  const confidence = data.confidence > 1 ? data.confidence / 100 : data.confidence;

  const words = trimmed.match(/\b\S+\b/g) || [];
  const characterCount = trimmed.length;
  const wordCount = words.length;

  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    sentiment: data.sentiment,
    confidence,
    text: trimmed,
    wordCount,
    characterCount,
    timestamp: new Date().toISOString(),
  };
}

