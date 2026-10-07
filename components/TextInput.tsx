"use client";

import React, { ChangeEvent } from "react";
import { Loader2, Sparkles, RotateCcw, Quote } from "lucide-react";

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  onClear: () => void;
  isLoading: boolean;
  maxLength?: number;
}

const EXAMPLES = [
  {
    label: "Positive example",
    type: "positive",
    text: "I absolutely loved this product. The quality was amazing!",
  },
  {
    label: "Negative example",
    type: "negative",
    text: "The service was terrible and the product stopped working after one day.",
  },
  {
    label: "Neutral example",
    type: "neutral",
    text: "The package arrived yesterday and contains the item I ordered.",
  },
];

export function TextInput({
  value,
  onChange,
  onAnalyze,
  onClear,
  isLoading,
  maxLength = 5000,
}: TextInputProps) {
  const charCount = value.length;
  const isOverLimit = charCount > maxLength;
  const isEmpty = value.trim().length === 0;

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl + Enter or Cmd + Enter to quickly trigger analysis
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      if (!isEmpty && !isLoading && !isOverLimit) {
        e.preventDefault();
        onAnalyze();
      }
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      {/* Label and Character Count */}
      <div className="flex items-center justify-between">
        <label
          htmlFor="sentiment-input-text"
          className="text-xs font-semibold uppercase tracking-wider text-slate-400"
        >
          Input Passage
        </label>
        <span
          className={`font-mono text-xs ${
            isOverLimit
              ? "text-red-400 font-bold"
              : charCount > maxLength * 0.9
              ? "text-amber-400"
              : "text-slate-500"
          }`}
          aria-live="polite"
        >
          {charCount.toLocaleString()} / {maxLength.toLocaleString()}
        </span>
      </div>

      {/* Textarea container */}
      <div className="relative rounded-xl border border-border bg-surface transition-colors focus-within:border-blue-500/80 focus-within:ring-1 focus-within:ring-blue-500/40">
        <textarea
          id="sentiment-input-text"
          rows={8}
          value={value}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          maxLength={maxLength}
          placeholder="Type or paste text here..."
          disabled={isLoading}
          className="min-h-[220px] w-full resize-none rounded-xl bg-transparent p-4 text-sm leading-relaxed text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-60"
        />

        {value && !isLoading && (
          <button
            type="button"
            onClick={onClear}
            className="absolute right-3 bottom-3 inline-flex items-center gap-1 rounded-md border border-border/80 bg-surface-subtle/80 px-2 py-1 text-xs text-slate-400 transition hover:bg-surface-hover hover:text-slate-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500"
            title="Clear text"
            aria-label="Clear current input"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* Example Prompts */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pt-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-slate-500 flex items-center gap-1">
            <Quote className="h-3 w-3" /> Examples:
          </span>
          {EXAMPLES.map((example) => (
            <button
              key={example.label}
              type="button"
              onClick={() => onChange(example.text)}
              disabled={isLoading}
              className="rounded-md border border-border bg-surface-subtle px-2.5 py-1 text-xs font-medium text-slate-300 transition hover:border-slate-600 hover:bg-surface-hover hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 disabled:opacity-50"
            >
              {example.label}
            </button>
          ))}
        </div>
        <span className="hidden lg:inline text-[11px] text-slate-500">
          Press <kbd className="rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-400 font-mono">⌘</kbd> + <kbd className="rounded bg-slate-800 px-1 py-0.5 text-[10px] text-slate-400 font-mono">Enter</kbd> to analyze
        </span>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-3 pt-2">
        <button
          type="button"
          onClick={onAnalyze}
          disabled={isEmpty || isLoading || isOverLimit}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-3.5 w-3.5" />
              <span>Analyze Sentiment</span>
            </>
          )}
        </button>

        <button
          type="button"
          onClick={onClear}
          disabled={isEmpty || isLoading}
          className="rounded-lg border border-border bg-surface-subtle px-4 py-2.5 text-xs font-medium text-slate-300 transition hover:bg-surface-hover hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
