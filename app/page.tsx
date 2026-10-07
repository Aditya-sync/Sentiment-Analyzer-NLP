"use client";

import React, { useState, useMemo } from "react";
import { Navbar } from "@/components/Navbar";
import { SentimentAnalyzer } from "@/components/SentimentAnalyzer";
import { StatsCard } from "@/components/StatsCard";
import { HistoryTable } from "@/components/HistoryTable";
import { Footer } from "@/components/Footer";
import { SentimentResult } from "@/types/sentiment";
import {
  Sparkles,
  Layers,
  ArrowRight,
  Server,
  Code2,
  CheckCircle2,
  Activity,
  ThumbsUp,
  ThumbsDown,
  HelpCircle,
} from "lucide-react";

// Initial realistic demonstration analyses
const INITIAL_HISTORY: SentimentResult[] = [
  {
    id: "init-1",
    sentiment: "positive",
    confidence: 0.942,
    text: "I absolutely loved this product. The quality was amazing!",
    wordCount: 9,
    characterCount: 57,
    timestamp: new Date(Date.now() - 1000 * 60 * 1).toISOString(), // 1 min ago
  },
  {
    id: "init-2",
    sentiment: "negative",
    confidence: 0.971,
    text: "The service was terrible and the product stopped working after one day.",
    wordCount: 12,
    characterCount: 71,
    timestamp: new Date(Date.now() - 1000 * 60 * 2).toISOString(), // 2 mins ago
  },
  {
    id: "init-3",
    sentiment: "neutral",
    confidence: 0.724,
    text: "The package arrived yesterday and contains the item I ordered.",
    wordCount: 10,
    characterCount: 62,
    timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(), // 5 mins ago
  },
];

export default function HomePage() {
  const [history, setHistory] = useState<SentimentResult[]>(INITIAL_HISTORY);
  const [activeResult, setActiveResult] = useState<SentimentResult | null>(
    INITIAL_HISTORY[0]
  );

  // Dynamically calculated statistics based on history
  const stats = useMemo(() => {
    const total = history.length;
    let positive = 0;
    let negative = 0;
    let neutral = 0;

    for (const item of history) {
      if (item.sentiment === "positive") positive++;
      else if (item.sentiment === "negative") negative++;
      else if (item.sentiment === "neutral") neutral++;
    }

    return { total, positive, negative, neutral };
  }, [history]);

  const handleNewResult = (newResult: SentimentResult) => {
    setHistory((prev) => {
      // Prepend and limit to recent 10 items
      const updated = [newResult, ...prev.filter((item) => item.id !== newResult.id)];
      return updated.slice(0, 10);
    });
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background selection:bg-blue-500/20 selection:text-blue-200">
      <Navbar />

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6 md:py-12 space-y-12">
        {/* Compact Hero Section */}
        <section className="text-center space-y-4 pt-2 pb-2">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-subtle px-3 py-1 text-[11px] font-mono font-medium text-slate-300 shadow-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
            <span>ROBERTA V2 • 3-CLASS TRANSFORMER</span>
          </div>

          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl md:text-5xl">
            Understand the sentiment behind your text.
          </h1>

          <p className="mx-auto max-w-2xl text-xs sm:text-sm text-slate-400 leading-relaxed">
            Analyze text using a fine-tuned RoBERTa Transformer model on TweetEval and instantly see its predicted 3-class sentiment and confidence.
          </p>

          <div className="pt-1 flex items-center justify-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-surface-card px-2.5 py-1 text-[11px] text-slate-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Runs locally • No API key required
            </span>
          </div>
        </section>

        {/* Dynamic Statistics Section */}
        <section aria-label="Dynamic Analysis Statistics">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatsCard
              label="Total Analyses"
              value={stats.total}
              icon={<Activity className="h-4 w-4" />}
            />
            <StatsCard
              label="Positive"
              value={stats.positive}
              variant="positive"
              icon={<ThumbsUp className="h-4 w-4" />}
            />
            <StatsCard
              label="Negative"
              value={stats.negative}
              variant="negative"
              icon={<ThumbsDown className="h-4 w-4" />}
            />
            <StatsCard
              label="Neutral"
              value={stats.neutral}
              variant="neutral"
              icon={<HelpCircle className="h-4 w-4" />}
            />
          </div>
        </section>

        {/* Primary Sentiment Analyzer Card */}
        <section>
          <SentimentAnalyzer
            onNewResult={handleNewResult}
            activeResult={activeResult}
            setActiveResult={setActiveResult}
          />
        </section>

        {/* Recent Analyses History Section */}
        <section id="history-section" className="space-y-4">
          <HistoryTable
            history={history}
            onClearHistory={handleClearHistory}
            onSelectResult={(item) => {
              setActiveResult(item);
              const element = document.getElementById("analyzer-section");
              if (element) {
                element.scrollIntoView({ behavior: "smooth" });
              }
            }}
          />
        </section>

        {/* About / How it works Section */}
        <section
          id="about-section"
          className="rounded-2xl border border-border bg-surface p-6 sm:p-8 space-y-8"
        >
          <div>
            <div className="inline-flex items-center gap-1.5 rounded-md bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-400">
              <Sparkles className="h-3 w-3" />
              <span>Architecture</span>
            </div>
            <h3 className="mt-2 text-xl font-bold tracking-tight text-white">
              How it works
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              High-level pipeline of the Natural Language Processing workflow.
            </p>
          </div>

          {/* 3-Step Process */}
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="rounded-xl border border-border bg-surface-subtle p-5">
              <span className="font-mono text-2xl font-black text-blue-500/40">
                01
              </span>
              <h4 className="mt-2 text-sm font-semibold text-white">
                Enter text
              </h4>
              <p className="mt-1 text-xs text-slate-400 leading-relaxed">
                Provide sentences, product reviews, or customer messages up to 5,000 characters.
              </p>
            </div>

            <div className="rounded-xl border border-border bg-surface-subtle p-5">
              <span className="font-mono text-2xl font-black text-blue-500/40">
                02
              </span>
              <h4 className="mt-2 text-sm font-semibold text-white">
                RoBERTa Transformer analyzes it
              </h4>
              <p className="mt-1 text-xs text-slate-400 leading-relaxed">
                The fine-tuned RoBERTa model processes subword tokens through self-attention layers to capture deep contextual semantics.
              </p>
            </div>

            <div className="rounded-xl border border-border bg-surface-subtle p-5">
              <span className="font-mono text-2xl font-black text-blue-500/40">
                03
              </span>
              <h4 className="mt-2 text-sm font-semibold text-white">
                3-class sentiment &amp; confidence
              </h4>
              <p className="mt-1 text-xs text-slate-400 leading-relaxed">
                Outputs positive, neutral, or negative classifications alongside softmax probability distributions.
              </p>
            </div>
          </div>

          {/* Technology Stack with Clean Frontend / Backend Distinction */}
          <div className="border-t border-border/80 pt-6">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">
              Technology Stack &amp; Model Architecture
            </h4>
            <div className="grid gap-4 sm:grid-cols-2">
              {/* Frontend Card */}
              <div className="rounded-xl border border-border bg-surface-subtle/70 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Code2 className="h-4 w-4 text-blue-400" />
                    <span className="text-xs font-semibold text-white">
                      Frontend (Active)
                    </span>
                  </div>
                  <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400 border border-emerald-500/20">
                    Live
                  </span>
                </div>
                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    Next.js (App Router)
                  </span>
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    TypeScript
                  </span>
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    Tailwind CSS
                  </span>
                </div>
              </div>

              {/* Backend Card */}
              <div className="rounded-xl border border-border bg-surface-subtle/70 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Server className="h-4 w-4 text-violet-400" />
                    <span className="text-xs font-semibold text-white">
                      Backend (FastAPI &amp; RoBERTa V2)
                    </span>
                  </div>
                  <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400 border border-emerald-500/20">
                    Live
                  </span>
                </div>
                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    FastAPI
                  </span>
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    Python
                  </span>
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    PyTorch
                  </span>
                  <span className="rounded-md border border-border bg-surface px-2.5 py-1 text-slate-300">
                    RoBERTa V2
                  </span>
                </div>
              </div>
            </div>

            {/* Architecture Evolution Note */}
            <div className="mt-4 rounded-xl border border-border/80 bg-surface/50 p-4 text-xs text-slate-400 space-y-1">
              <span className="font-semibold text-slate-300">Project Evolution:</span>
              <p>
                <strong className="text-slate-300">V2 (Production):</strong> Fine-tuned RoBERTa Transformer on TweetEval (3-class: negative, neutral, positive) served locally via FastAPI.
              </p>
              <p>
                <strong className="text-slate-300">V1 (Baseline):</strong> Classical TF-IDF + Logistic Regression binary classifier on IMDb preserved as a benchmark.
              </p>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
