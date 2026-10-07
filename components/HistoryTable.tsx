"use client";

import React from "react";
import { Trash2, Clock, Check, X, Minus, ArrowUpRight } from "lucide-react";
import { SentimentResult, Sentiment } from "@/types/sentiment";
import { EmptyState } from "./EmptyState";

interface HistoryTableProps {
  history: SentimentResult[];
  onClearHistory: () => void;
  onSelectResult?: (result: SentimentResult) => void;
}

function formatRelativeTime(dateString: string): string {
  const diffMs = Date.now() - new Date(dateString).getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 45) return "Just now";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return new Date(dateString).toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

function SentimentBadge({ sentiment }: { sentiment: Sentiment }) {
  switch (sentiment) {
    case "positive":
      return (
        <span className="inline-flex items-center gap-1 rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-400">
          <Check className="h-3 w-3 stroke-[2.5]" />
          Positive
        </span>
      );
    case "negative":
      return (
        <span className="inline-flex items-center gap-1 rounded-md border border-rose-500/25 bg-rose-500/10 px-2 py-0.5 text-xs font-medium text-rose-400">
          <X className="h-3 w-3 stroke-[2.5]" />
          Negative
        </span>
      );
    case "neutral":
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded-md border border-amber-500/25 bg-amber-500/10 px-2 py-0.5 text-xs font-medium text-amber-400">
          <Minus className="h-3 w-3 stroke-[2.5]" />
          Neutral
        </span>
      );
  }
}

export function HistoryTable({
  history,
  onClearHistory,
  onSelectResult,
}: HistoryTableProps) {
  if (history.length === 0) {
    return (
      <EmptyState
        title="No analysis history yet"
        description="Run your first sentiment analysis to populate recent runs in this session."
        icon={<Clock className="h-5 w-5 text-slate-500" />}
      />
    );
  }

  return (
    <div className="rounded-2xl border border-border bg-surface overflow-hidden shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border p-4 sm:px-6">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Recent Analyses
          </span>
          <span className="rounded-full bg-surface-subtle border border-border px-2 py-0.5 text-[11px] font-mono text-slate-400">
            {history.length}
          </span>
        </div>

        <button
          type="button"
          onClick={onClearHistory}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface-subtle px-2.5 py-1 text-xs text-slate-400 transition hover:border-rose-500/30 hover:bg-rose-500/10 hover:text-rose-300 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-rose-500"
        >
          <Trash2 className="h-3 w-3" />
          <span>Clear History</span>
        </button>
      </div>

      {/* Desktop Table View */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-border/80 bg-surface-subtle text-slate-400">
            <tr>
              <th scope="col" className="py-3 px-6 font-medium">
                Sentiment
              </th>
              <th scope="col" className="py-3 px-6 font-medium">
                Text Preview
              </th>
              <th scope="col" className="py-3 px-6 font-medium">
                Confidence
              </th>
              <th scope="col" className="py-3 px-6 font-medium">
                Time
              </th>
              <th scope="col" className="py-3 px-6 text-right font-medium">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            {history.map((item) => (
              <tr
                key={item.id}
                className="group transition-colors hover:bg-surface-subtle/50"
              >
                <td className="py-3.5 px-6 whitespace-nowrap">
                  <SentimentBadge sentiment={item.sentiment} />
                </td>
                <td className="py-3.5 px-6 font-normal text-slate-300 max-w-[340px]">
                  <p className="truncate" title={item.text}>
                    &ldquo;{item.text}&rdquo;
                  </p>
                </td>
                <td className="py-3.5 px-6 font-mono text-slate-300 whitespace-nowrap">
                  {(item.confidence * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-6 text-slate-400 whitespace-nowrap" suppressHydrationWarning>
                  {formatRelativeTime(item.timestamp)}
                </td>
                <td className="py-3.5 px-6 text-right whitespace-nowrap">
                  {onSelectResult && (
                    <button
                      type="button"
                      onClick={() => onSelectResult(item)}
                      className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-blue-400 transition"
                      title="Load this result"
                    >
                      <span>Inspect</span>
                      <ArrowUpRight className="h-3 w-3" />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile Card List View */}
      <div className="divide-y divide-border/60 md:hidden">
        {history.map((item) => (
          <div
            key={item.id}
            onClick={() => onSelectResult && onSelectResult(item)}
            className="p-4 transition-colors hover:bg-surface-subtle/50 cursor-pointer"
          >
            <div className="flex items-center justify-between">
              <SentimentBadge sentiment={item.sentiment} />
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-slate-300">
                  {(item.confidence * 100).toFixed(1)}%
                </span>
                <span className="text-[11px] text-slate-500" suppressHydrationWarning>
                  • {formatRelativeTime(item.timestamp)}
                </span>
              </div>
            </div>
            <p className="mt-2 text-xs text-slate-300 line-clamp-2 leading-relaxed">
              &ldquo;{item.text}&rdquo;
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
