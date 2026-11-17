import React from 'react';
import Card from '../common/Card';

export const MetricsCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  colorScheme = 'purple',
}) => {
  const schemeStyles = {
    purple: {
      iconBg: 'bg-purple-500/10 border-purple-500/20 text-purple-400',
      glow: 'group-hover:border-purple-500/40',
    },
    blue: {
      iconBg: 'bg-blue-500/10 border-blue-500/20 text-blue-400',
      glow: 'group-hover:border-blue-500/40',
    },
    emerald: {
      iconBg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
      glow: 'group-hover:border-emerald-500/40',
    },
    rose: {
      iconBg: 'bg-rose-500/10 border-rose-500/20 text-rose-400',
      glow: 'group-hover:border-rose-500/40',
    },
    amber: {
      iconBg: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
      glow: 'group-hover:border-amber-500/40',
    },
  };

  const scheme = schemeStyles[colorScheme] || schemeStyles.purple;

  return (
    <Card className={`group relative transition-all duration-300 hover:shadow-xl ${scheme.glow}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold tracking-wider uppercase text-slate-400 mb-1">
            {title}
          </p>
          <h3 className="text-2xl font-bold text-white tracking-tight font-mono">
            {value}
          </h3>
          {subtitle && (
            <p className="text-xs text-slate-400 mt-1 flex items-center gap-1.5">
              {subtitle}
            </p>
          )}
        </div>

        {Icon && (
          <div className={`p-3 rounded-xl border shrink-0 ${scheme.iconBg}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      {trend && (
        <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
          <span className="text-slate-400">Trend</span>
          <span className={`font-medium ${trend.isPositive ? 'text-emerald-400' : 'text-slate-300'}`}>
            {trend.value}
          </span>
        </div>
      )}
    </Card>
  );
};

export default MetricsCard;
