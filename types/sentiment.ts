export type Sentiment = "positive" | "negative" | "neutral";

export interface SentimentResult {
  id: string;
  sentiment: Sentiment;
  confidence: number;
  text: string;
  wordCount: number;
  characterCount: number;
  timestamp: string;
}

export interface SentimentStats {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
}
