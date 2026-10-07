import React, { ReactNode } from "react";

interface StatsCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon?: ReactNode;
  variant?: "default" | "positive" | "negative" | "neutral" | "accent";
}

export function StatsCard({
  label,
  value,
  subtext,
  icon,
  variant = "default",
}: StatsCardProps) {
  const getVariantStyles = () => {
    switch (variant) {
      case "positive":
        return "border-emerald-500/20 bg-emerald-950/10 text-emerald-400";
      case "negative":
        return "border-rose-500/20 bg-rose-950/10 text-rose-400";
      case "neutral":
        return "border-amber-500/20 bg-amber-950/10 text-amber-400";
      case "accent":
        return "border-blue-500/20 bg-blue-950/10 text-blue-400";
      default:
        return "border-border bg-surface text-slate-100";
    }
  };

  return (
    <div
      className={`flex flex-col justify-between rounded-xl border p-4 transition-colors ${getVariantStyles()}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {label}
        </span>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="font-mono text-2xl font-bold tracking-tight text-white">
          {value}
        </span>
        {subtext && (
          <span className="text-xs text-slate-400">
            {subtext}
          </span>
        )}
      </div>
    </div>
  );
}
