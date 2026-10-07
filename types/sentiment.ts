export type Sentiment = "positive" | "negative" | "neutral";

/**
 * Expected schema returned by the FastAPI V2 endpoint (POST /predict/v2).
 */
export interface PredictV2Response {
  text: string;
  sentiment: Sentiment;
  confidence: number;
  negative_probability: number;
  neutral_probability: number;
  positive_probability: number;
}

/**
 * Normalized result stored and displayed within the frontend application.
 */
export interface SentimentResult {
  id: string;
  sentiment: Sentiment;
  confidence: number;
  text: string;
  wordCount: number;
  characterCount: number;
  timestamp: string;
  negative_probability?: number;
  neutral_probability?: number;
  positive_probability?: number;
}

export interface SentimentStats {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
}
