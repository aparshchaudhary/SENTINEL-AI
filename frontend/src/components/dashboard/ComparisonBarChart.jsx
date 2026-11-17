import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import Card from '../common/Card';
import EmptyState from '../common/EmptyState';
import { BarChart3 } from 'lucide-react';

const COLORS = {
  Positive: '#10b981',
  Neutral: '#f59e0b',
  Negative: '#ef4444',
};

const CustomBarTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-slate-900 border border-slate-700/80 p-2.5 rounded-xl shadow-xl text-xs">
        <p className="font-semibold text-white capitalize">{data.name}</p>
        <p className="text-slate-300 mt-1 font-mono">
          Total: <span className="font-bold text-white">{data.value.toLocaleString()}</span>
        </p>
      </div>
    );
  }
  return null;
};

export const ComparisonBarChart = ({ data = [], title = 'Comparative Breakdown', subtitle }) => {
  const hasData = Array.isArray(data) && data.length > 0 && data.some((d) => d.value > 0);

  return (
    <Card title={title} subtitle={subtitle} className="h-full flex flex-col justify-between">
      {!hasData ? (
        <EmptyState
          icon={BarChart3}
          title="No Comparison Data"
          description="Volume by class will display here once analyses are stored."
          className="border-none py-12"
        />
      ) : (
        <div className="w-full h-64 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip content={<CustomBarTooltip />} />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`bar-${index}`} fill={COLORS[entry.name] || '#8b5cf6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
};

export default ComparisonBarChart;
