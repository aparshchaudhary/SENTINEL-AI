import React from 'react';

const VARIANTS = {
  positive: {
    bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 light:bg-emerald-50 light:text-emerald-700 light:border-emerald-200',
    dot: 'bg-emerald-400',
  },
  negative: {
    bg: 'bg-rose-500/10 text-rose-400 border-rose-500/20 light:bg-rose-50 light:text-rose-700 light:border-rose-200',
    dot: 'bg-rose-400',
  },
  neutral: {
    bg: 'bg-amber-500/10 text-amber-400 border-amber-500/20 light:bg-amber-50 light:text-amber-700 light:border-amber-200',
    dot: 'bg-amber-400',
  },
  purple: {
    bg: 'bg-purple-500/10 text-purple-400 border-purple-500/20 light:bg-purple-50 light:text-purple-700 light:border-purple-200',
    dot: 'bg-purple-400',
  },
  blue: {
    bg: 'bg-blue-500/10 text-blue-400 border-blue-500/20 light:bg-blue-50 light:text-blue-700 light:border-blue-200',
    dot: 'bg-blue-400',
  },
  slate: {
    bg: 'bg-slate-700/30 text-slate-300 border-slate-700/50 light:bg-slate-100 light:text-slate-700 light:border-slate-300',
    dot: 'bg-slate-400',
  },
};

const SIZES = {
  sm: 'text-[11px] px-2 py-0.5 rounded-md gap-1',
  md: 'text-xs px-2.5 py-1 rounded-lg gap-1.5 font-medium',
  lg: 'text-sm px-3 py-1.5 rounded-xl gap-2 font-medium',
};

export const Badge = ({
  children,
  variant = 'slate',
  size = 'md',
  dot = false,
  className = '',
}) => {
  const v = VARIANTS[variant.toLowerCase()] || VARIANTS.slate;
  const s = SIZES[size] || SIZES.md;

  return (
    <span
      className={`inline-flex items-center border font-medium uppercase tracking-wider ${v.bg} ${s} ${className}`}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${v.dot} shrink-0 animate-pulse`} />}
      {children}
    </span>
  );
};

export default Badge;
