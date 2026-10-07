"use client";

import React, { useState } from "react";
import { Sparkles, Menu, X, Cpu } from "lucide-react";

export function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const scrollTo = (id: string) => {
    setMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/80 bg-background/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Brand / Logo */}
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-surface-card text-blue-400 shadow-sm">
            <Sparkles className="h-4 w-4" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold tracking-tight text-white">
              Sentiment AI
            </span>
            <span className="rounded border border-border/70 bg-surface-subtle px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-slate-400">
              v2.0
            </span>
          </div>
        </div>

        {/* Center Navigation Links - Desktop */}
        <nav
          aria-label="Main Navigation"
          className="hidden items-center gap-7 md:flex text-xs font-medium text-slate-400"
        >
          <button
            onClick={() => scrollTo("analyzer-section")}
            className="transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 rounded px-1.5 py-1"
          >
            Analyzer
          </button>
          <button
            onClick={() => scrollTo("history-section")}
            className="transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 rounded px-1.5 py-1"
          >
            History
          </button>
          <button
            onClick={() => scrollTo("about-section")}
            className="transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 rounded px-1.5 py-1"
          >
            About
          </button>
        </nav>

        {/* Right Status Indicator */}
        <div className="hidden sm:flex items-center gap-3">
          <div
            className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-subtle px-2.5 py-1 text-xs text-slate-300 shadow-sm"
            title="Connected to the local FastAPI backend running the fine-tuned RoBERTa V2 model."
          >
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
            </span>
        
            <span className="font-mono text-[11px] text-slate-300">
              RoBERTa V2
            </span>
        
            <span className="rounded bg-slate-800/80 px-1 py-0.2 text-[9px] text-emerald-400 uppercase tracking-wider font-semibold">
              LIVE
            </span>
          </div>
        </div>

        {/* Mobile menu trigger */}
        <div className="flex items-center gap-2 md:hidden">
          <div className="flex items-center gap-1.5 rounded-full border border-border bg-surface-subtle px-2 py-0.5 text-[11px] text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span className="text-[10px]">RoBERTa V2</span>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-border bg-surface-subtle text-slate-300 hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500"
            aria-label="Toggle navigation menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="border-b border-border bg-surface px-4 py-3 md:hidden">
          <div className="flex flex-col space-y-2 text-sm text-slate-300">
            <button
              onClick={() => scrollTo("analyzer-section")}
              className="flex items-center gap-2 py-1.5 text-left transition-colors hover:text-white"
            >
              Analyzer
            </button>
            <button
              onClick={() => scrollTo("history-section")}
              className="flex items-center gap-2 py-1.5 text-left transition-colors hover:text-white"
            >
              History
            </button>
            <button
              onClick={() => scrollTo("about-section")}
              className="flex items-center gap-2 py-1.5 text-left transition-colors hover:text-white"
            >
              About
            </button>
            <div className="pt-2 border-t border-border/60 text-xs text-slate-400 flex items-center gap-2">
              <Cpu className="h-3.5 w-3.5 text-emerald-400" />
              <span>Status: RoBERTa V2 · LIVE</span>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
