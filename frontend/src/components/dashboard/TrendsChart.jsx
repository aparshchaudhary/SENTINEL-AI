import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import Card from '../common/Card';
import EmptyState from '../common/EmptyState';
import { TrendingUp } from 'lucide-react';

const CustomTrendsTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700/80 p-3 rounded-xl shadow-2xl text-xs space-y-1.5 min-w-[140px]">
        <p className="font-semibold text-slate-200 border-b border-slate-800 pb-1 font-mono">
          {label}
        </p>
        {payload.map((entry, idx) => (
          <div key={`item-${idx}`} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 capitalize font-medium" style={{ color: entry.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              {entry.name}
            </span>
            <span className="font-mono text-white font-semibold">{entry.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export const TrendsChart = ({ data = [], title = 'Sentiment Volume Timeline', subtitle }) => {
  const hasData = Array.isArray(data) && data.length > 0 && data.some((d) => (d.total || 0) > 0);

  return (
    <Card title={title} subtitle={subtitle} className="h-full flex flex-col justify-between">
      {!hasData ? (
        <EmptyState
          icon={TrendingUp}
          title="No Timeline Activity"
          description="Timeline trends will appear here as sentiment analyses are processed across dates."
          className="border-none py-14"
        />
      ) : (
        <div className="w-full h-72 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="posGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="negGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="neuGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis
                dataKey="date"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
              />
              <Tooltip content={<CustomTrendsTooltip />} />
              <Legend
                verticalAlign="bottom"
                height={30}
                formatter={(value) => (
                  <span className="text-xs font-medium text-slate-300 capitalize ml-1">
                    {value}
                  </span>
                )}
              />

              <Area
                type="monotone"
                dataKey="positive"
                name="Positive"
                stroke="#10b981"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#posGradient)"
              />
              <Area
                type="monotone"
                dataKey="neutral"
                name="Neutral"
                stroke="#f59e0b"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#neuGradient)"
              />
              <Area
                type="monotone"
                dataKey="negative"
                name="Negative"
                stroke="#ef4444"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#negGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
};

export default TrendsChart;
