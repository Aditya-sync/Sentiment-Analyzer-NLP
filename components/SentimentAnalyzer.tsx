"use client";

import React, { useState } from "react";
import { MessageSquareText, AlertCircle } from "lucide-react";
import { TextInput } from "./TextInput";
import { AnalysisResult } from "./AnalysisResult";
import { EmptyState } from "./EmptyState";
import { SentimentResult } from "@/types/sentiment";
import { analyzeSentiment } from "@/lib/api";

interface SentimentAnalyzerProps {
  onNewResult: (result: SentimentResult) => void;
  activeResult: SentimentResult | null;
  setActiveResult: (result: SentimentResult | null) => void;
}

export function SentimentAnalyzer({
  onNewResult,
  activeResult,
  setActiveResult,
}: SentimentAnalyzerProps) {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!inputText.trim() || isLoading) return;

    try {
      setIsLoading(true);
      setErrorMessage(null);
      const result = await analyzeSentiment(inputText);
      setActiveResult(result);
      onNewResult(result);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("An unexpected error occurred during analysis.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setInputText("");
    setActiveResult(null);
    setErrorMessage(null);
  };

  return (
    <div id="analyzer-section" className="space-y-6">
      {/* Primary Input Card */}
      <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
        {/* Card Header */}
        <div className="flex items-start justify-between border-b border-border/80 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-md bg-blue-500/10 text-blue-400">
                <MessageSquareText className="h-3.5 w-3.5" />
              </div>
              <h2 className="text-base font-semibold text-white tracking-tight">
                Analyze Text
              </h2>
            </div>
            <p className="mt-1 text-xs text-slate-400">
              Enter a sentence, review, or message to classify its sentiment.
            </p>
          </div>

          <span className="hidden sm:inline-flex rounded-full border border-border bg-surface-subtle px-2.5 py-0.5 text-[11px] font-mono text-slate-400">
            NLP Engine Ready
          </span>
        </div>

        {/* Text Input Area */}
        <div className="mt-6">
          <TextInput
            value={inputText}
            onChange={(val) => {
              setInputText(val);
              if (errorMessage) setErrorMessage(null);
            }}
            onAnalyze={handleAnalyze}
            onClear={handleClear}
            isLoading={isLoading}
            maxLength={5000}
          />
        </div>

        {/* Error notification if any */}
        {errorMessage && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
        )}
      </div>

      {/* Analysis Output Section */}
      <div>
        {activeResult ? (
          <AnalysisResult result={activeResult} />
        ) : (
          <EmptyState />
        )}
      </div>
    </div>
  );
}
