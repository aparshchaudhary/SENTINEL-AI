import React from 'react';

export const ProbabilityBar = ({ probabilities = {}, showLabels = true }) => {
  const pos = Math.max(0, Math.min(1, probabilities.positive || 0));
  const neg = Math.max(0, Math.min(1, probabilities.negative || 0));
  const neu = Math.max(0, Math.min(1, probabilities.neutral || 0));

  const items = [
    {
      key: 'positive',
      label: 'Positive',
      val: pos,
      pct: (pos * 100).toFixed(1),
      barColor: 'bg-emerald-500',
      bgColor: 'bg-emerald-500/10',
      textColor: 'text-emerald-400',
    },
    {
      key: 'neutral',
      label: 'Neutral',
      val: neu,
      pct: (neu * 100).toFixed(1),
      barColor: 'bg-amber-500',
      bgColor: 'bg-amber-500/10',
      textColor: 'text-amber-400',
    },
    {
      key: 'negative',
      label: 'Negative',
      val: neg,
      pct: (neg * 100).toFixed(1),
      barColor: 'bg-rose-500',
      bgColor: 'bg-rose-500/10',
      textColor: 'text-rose-400',
    },
  ];

  return (
    <div className="space-y-3 w-full">
      {items.map((item) => (
        <div key={item.key} className="space-y-1.5">
          {showLabels && (
            <div className="flex items-center justify-between text-xs font-medium">
              <span className={`tracking-wider uppercase font-semibold ${item.textColor}`}>
                {item.label}
              </span>
              <span className="text-slate-300 font-mono">
                {item.pct}%
              </span>
            </div>
          )}
          <div className="h-2.5 w-full bg-slate-800/80 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
            <div
              className={`h-full rounded-full transition-all duration-700 ease-out ${item.barColor}`}
              style={{ width: `${item.pct}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

export default ProbabilityBar;
