import React from "react";
import { Sparkles } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-20 border-t border-border/80 bg-background/50 py-8">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 text-xs text-slate-500 sm:flex-row sm:px-6">
        {/* Left */}
        <div className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-blue-400" />
          <span className="font-semibold text-slate-300">Sentiment AI</span>
          <span>© {new Date().getFullYear()}</span>
        </div>

        {/* Center */}
        <div className="flex items-center gap-1.5">
          <span>Built with</span>
          <span className="font-medium text-slate-300">Next.js &amp; Tailwind CSS</span>
        </div>

        {/* Right */}
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          <span>RoBERTa V2 Engine</span>
        </div>
      </div>
    </footer>
  );
}
