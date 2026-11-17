import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import Card from '../common/Card';
import EmptyState from '../common/EmptyState';
import { PieChart as PieIcon } from 'lucide-react';

const DEFAULT_COLORS = {
  Positive: '#10b981', // emerald-500
  Negative: '#ef4444', // rose-500
  Neutral: '#f59e0b',  // amber-500
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-slate-900 border border-slate-700/80 px-3 py-2 rounded-xl shadow-xl text-xs">
        <p className="font-semibold text-white flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: data.color || DEFAULT_COLORS[data.name] }} />
          {data.name}
        </p>
        <p className="text-slate-300 mt-1">
          Count: <span className="font-mono font-medium text-white">{data.value.toLocaleString()}</span>
        </p>
        <p className="text-slate-400">
          Share: <span className="font-mono text-white">{data.percentage || 0}%</span>
        </p>
      </div>
    );
  }
  return null;
};

export const SentimentPieChart = ({ data = [], title = 'Polarity Distribution', subtitle }) => {
  const hasData = Array.isArray(data) && data.length > 0 && data.some((item) => item.value > 0);

  return (
    <Card title={title} subtitle={subtitle} className="h-full flex flex-col justify-between">
      {!hasData ? (
        <EmptyState
          icon={PieIcon}
          title="No Distribution Data"
          description="Analyze text or upload CSVs to visualize sentiment distribution."
          className="border-none py-12"
        />
      ) : (
        <div className="w-full h-64 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={4}
                dataKey="value"
                nameKey="name"
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.color || DEFAULT_COLORS[entry.name] || '#8b5cf6'}
                    stroke="#0b0f19"
                    strokeWidth={2}
                  />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value, entry) => (
                  <span className="text-xs font-medium text-slate-300 ml-1">
                    {value}
                  </span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
};

export default SentimentPieChart;
