"use client";

import React, { useState } from "react";
import { Check, X, Minus, Copy, CheckCheck, FileText } from "lucide-react";
import { SentimentResult, Sentiment } from "@/types/sentiment";
import { ConfidenceBar } from "./ConfidenceBar";
import { StatsCard } from "./StatsCard";

interface AnalysisResultProps {
  result: SentimentResult;
}

export function AnalysisResult({ result }: AnalysisResultProps) {
  const [copied, setCopied] = useState(false);

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard fallback
    }
  };

  const getSentimentConfig = (sentiment: Sentiment) => {
    switch (sentiment) {
      case "positive":
        return {
          label: "POSITIVE",
          display: "Positive",
          icon: <Check className="h-5 w-5 stroke-[2.5]" />,
          textClass: "text-emerald-400",
          borderClass: "border-emerald-500/30",
          bgClass: "bg-emerald-500/10",
          cardBorder: "border-emerald-500/20",
          variant: "positive" as const,
        };
      case "negative":
        return {
          label: "NEGATIVE",
          display: "Negative",
          icon: <X className="h-5 w-5 stroke-[2.5]" />,
          textClass: "text-rose-400",
          borderClass: "border-rose-500/30",
          bgClass: "bg-rose-500/10",
          cardBorder: "border-rose-500/20",
          variant: "negative" as const,
        };
      case "neutral":
      default:
        return {
          label: "NEUTRAL",
          display: "Neutral",
          icon: <Minus className="h-5 w-5 stroke-[2.5]" />,
          textClass: "text-amber-400",
          borderClass: "border-amber-500/30",
          bgClass: "bg-amber-500/10",
          cardBorder: "border-amber-500/20",
          variant: "neutral" as const,
        };
    }
  };

  const config = getSentimentConfig(result.sentiment);
  const confidencePercent = (result.confidence * 100).toFixed(1) + "%";

  return (
    <div className="space-y-6">
      {/* Result Card */}
      <div className={`overflow-hidden rounded-2xl border ${config.cardBorder} bg-surface p-6 shadow-sm`}>
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/80 pb-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Analysis Result
          </span>
          <span className="font-mono text-xs text-slate-500" suppressHydrationWarning>
            {new Date(result.timestamp).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
          </span>
        </div>

        {/* Large Sentiment Display */}
        <div className="mt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="text-xs font-medium text-slate-400 mb-1.5">
              Detected Sentiment
            </div>
            <div
              className={`inline-flex items-center gap-2 rounded-xl border px-4 py-2 font-mono text-xl sm:text-2xl font-bold tracking-tight ${config.bgClass} ${config.borderClass} ${config.textClass}`}
            >
              {config.icon}
              <span>{config.label}</span>
            </div>
          </div>

          <div className="sm:text-right">
            <div className="text-xs font-medium text-slate-400 mb-1.5">
              Confidence
            </div>
            <div className={`font-mono text-2xl font-bold ${config.textClass}`}>
              {confidencePercent}
            </div>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mt-6">
          <ConfidenceBar confidence={result.confidence} sentiment={result.sentiment} />
        </div>

        {/* Statistics Cards */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatsCard
            label="Sentiment"
            value={config.display}
            variant={config.variant}
          />
          <StatsCard
            label="Confidence"
            value={confidencePercent}
            variant={config.variant}
          />
          <StatsCard
            label="Characters"
            value={result.characterCount.toLocaleString()}
          />
          <StatsCard
            label="Words"
            value={result.wordCount.toLocaleString()}
          />
        </div>
      </div>

      {/* Submitted / Analyzed Text Section */}
      <div className="rounded-xl border border-border bg-surface-subtle p-5">
        <div className="flex items-center justify-between pb-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <FileText className="h-3.5 w-3.5 text-slate-500" />
            <span>Analyzed Text</span>
          </div>
          <button
            type="button"
            onClick={copyText}
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1 text-xs text-slate-400 transition hover:bg-surface-hover hover:text-slate-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500"
            title="Copy text to clipboard"
          >
            {copied ? (
              <>
                <CheckCheck className="h-3 w-3 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
        <blockquote className="rounded-lg border-l-2 border-slate-600 bg-surface/60 p-4 text-sm leading-relaxed text-slate-200 italic font-sans">
          &ldquo;{result.text}&rdquo;
        </blockquote>
      </div>
    </div>
  );
}
