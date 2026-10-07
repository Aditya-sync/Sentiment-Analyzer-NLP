import { PredictV2Response, SentimentResult } from "@/types/sentiment";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Service to analyze sentiment from input text via the FastAPI backend.
 * Calls the fine-tuned RoBERTa Transformer sentiment model at POST /predict/v2.
 */
export async function analyzeSentiment(text: string): Promise<SentimentResult> {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Text cannot be empty");
  }

  let data: PredictV2Response;

  try {
    const response = await fetch(`${API_BASE_URL}/predict/v2`, {
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
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.message.includes("Failed to fetch") || err.name === "TypeError") {
        throw new Error(
          "Could not connect to the ML Backend at http://localhost:8000. Please ensure your FastAPI server is running."
        );
      }
      throw err;
    }
    throw new Error("An unexpected network error occurred.");
  }

  // Normalize confidence to 0-1 scale if returned as a percentage (e.g. 99.61 -> 0.9961)
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
    negative_probability: data.negative_probability,
    neutral_probability: data.neutral_probability,
    positive_probability: data.positive_probability,
  };
}
