import { Sentiment } from "@/types/sentiment";

export interface MockRawPrediction {
  sentiment: Sentiment;
  confidence: number;
}

const POSITIVE_KEYWORDS = [
  "love",
  "loved",
  "loving",
  "amazing",
  "great",
  "excellent",
  "good",
  "wonderful",
  "fantastic",
  "awesome",
  "best",
  "delightful",
  "superb",
  "happy",
  "perfect",
];

const NEGATIVE_KEYWORDS = [
  "hate",
  "hated",
  "terrible",
  "bad",
  "worst",
  "awful",
  "poor",
  "horrible",
  "disappointing",
  "disaster",
  "broken",
  "annoying",
  "useless",
  "slow",
  "fail",
  "failed",
];

/**
 * Deterministic hash to calculate reproducible confidence scores for demo purposes.
 */
function getDeterministicSeed(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

/**
 * Deterministic mock sentiment classifier based on keyword scoring.
 * This is strictly a local simulation for frontend testing and does not use any AI model.
 */
export function mockSentimentPrediction(text: string): MockRawPrediction {
  const normalizedText = text.toLowerCase();
  const words = normalizedText.match(/\b\w+\b/g) || [];

  let positiveScore = 0;
  let negativeScore = 0;

  for (const word of words) {
    if (POSITIVE_KEYWORDS.includes(word)) {
      positiveScore++;
    }
    if (NEGATIVE_KEYWORDS.includes(word)) {
      negativeScore++;
    }
  }

  const seed = getDeterministicSeed(text);
  const variance = (seed % 100) / 1000; // 0.000 to 0.099

  let sentiment: Sentiment = "neutral";
  let confidence = 0.72 + variance; // 0.72 - 0.819 for neutral

  if (positiveScore > negativeScore) {
    sentiment = "positive";
    const boost = Math.min(positiveScore * 0.03, 0.08);
    confidence = Math.min(0.985, 0.88 + variance + boost);
  } else if (negativeScore > positiveScore) {
    sentiment = "negative";
    const boost = Math.min(negativeScore * 0.03, 0.08);
    confidence = Math.min(0.985, 0.87 + variance + boost);
  }

  // Round to 3 decimal places (e.g. 0.942)
  confidence = Math.round(confidence * 1000) / 1000;

  return {
    sentiment,
    confidence,
  };
}
