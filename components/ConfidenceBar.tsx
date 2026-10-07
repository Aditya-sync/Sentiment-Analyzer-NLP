"use client";

import React from "react";
import { Sentiment } from "@/types/sentiment";

interface ConfidenceBarProps {
  confidence: number; // 0 to 1, e.g. 0.942
  sentiment: Sentiment;
  showLabel?: boolean;
}

export function ConfidenceBar({
  confidence,
  sentiment,
  showLabel = true,
}: ConfidenceBarProps) {
  const percentage = Math.min(100, Math.max(0, Math.round(confidence * 1000) / 10));

  const getColorClasses = (sentiment: Sentiment) => {
    switch (sentiment) {
      case "positive":
        return {
          bar: "bg-emerald-500",
          text: "text-emerald-400",
        };
      case "negative":
        return {
          bar: "bg-rose-500",
          text: "text-rose-400",
        };
      case "neutral":
      default:
        return {
          bar: "bg-amber-500",
          text: "text-amber-400",
        };
    }
  };

  const colors = getColorClasses(sentiment);

  return (
    <div className="w-full space-y-1.5">
      {showLabel && (
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-slate-400">Confidence Score</span>
          <span className={`font-mono font-semibold ${colors.text}`}>
            {percentage.toFixed(1)}%
          </span>
        </div>
      )}

      {/* Progress Track */}
      <div
        role="progressbar"
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confidence score: ${percentage.toFixed(1)}%`}
        className="h-2 w-full overflow-hidden rounded-full bg-surface-subtle border border-border/80"
      >
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${colors.bar}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
