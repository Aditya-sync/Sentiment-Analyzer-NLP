import React, { ReactNode } from "react";
import { Terminal } from "lucide-react";

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: ReactNode;
}

export function EmptyState({
  title = "Your analysis will appear here.",
  description = "Enter text above and click \"Analyze Sentiment\" to inspect polarity and confidence.",
  icon,
}: EmptyStateProps) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center rounded-xl border border-dashed border-border/80 bg-surface/40 p-8 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-surface-subtle text-slate-400">
        {icon || <Terminal className="h-5 w-5 text-slate-500" />}
      </div>
      <h4 className="mt-3 text-sm font-medium text-slate-300">{title}</h4>
      <p className="mt-1 max-w-sm text-xs text-slate-500 leading-relaxed">
        {description}
      </p>
    </div>
  );
}
