import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layers,
  Smile,
  Frown,
  Zap,
  ArrowRight,
  RefreshCw,
  AlertCircle,
  FileSpreadsheet,
  MessageSquarePlus,
} from 'lucide-react';
import api from '../services/api';
import MetricsCard from '../components/dashboard/MetricsCard';
import SentimentPieChart from '../components/dashboard/SentimentPieChart';
import TrendsChart from '../components/dashboard/TrendsChart';
import ComparisonBarChart from '../components/dashboard/ComparisonBarChart';
import Button from '../components/common/Button';
import Spinner from '../components/common/Spinner';

export const Dashboard = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [distribution, setDistribution] = useState([]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumRes, trendsRes, distRes] = await Promise.all([
        api.getAnalyticsSummary(),
        api.getAnalyticsTrends(),
        api.getAnalyticsDistribution(),
      ]);
      setSummary(sumRes);
      setTrends(trendsRes.timeline || []);
      setDistribution(distRes.distribution || []);
    } catch (err) {
      setError(err.message || 'Failed to load analytics data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const totalAnalyses = summary?.total_analyses || 0;
  const positiveCount = summary?.positive_count || 0;
  const negativeCount = summary?.negative_count || 0;
  const positivePct = summary?.percentages?.positive || 0;
  const negativePct = summary?.percentages?.negative || 0;
  const avgConfidence = summary?.average_confidence
    ? (summary.average_confidence * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            System Executive Dashboard
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Real-time sentiment telemetry, classification metrics, and dataset distribution.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={fetchData}
            isLoading={loading}
            className="flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={() => navigate('/analyzer')}
            className="flex items-center gap-1.5"
          >
            <MessageSquarePlus className="w-3.5 h-3.5" />
            <span>Analyze Text</span>
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1 text-xs">
            <h4 className="font-semibold text-rose-400">Connection Error</h4>
            <p className="text-rose-200/80 mt-0.5">{error}</p>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData} className="text-xs py-1">
            Retry
          </Button>
        </div>
      )}

      {/* Loading Skeleton or Metrics Grid */}
      {loading && !summary ? (
        <div className="py-24 flex items-center justify-center">
          <Spinner size="lg" label="Loading real-time platform telemetry..." />
        </div>
      ) : (
        <>
          {/* Top 4 Metrics Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricsCard
              title="Total Analyses"
              value={totalAnalyses.toLocaleString()}
              subtitle="Lifetime records evaluated"
              icon={Layers}
              colorScheme="blue"
            />
            <MetricsCard
              title="Positive Polarity"
              value={positiveCount.toLocaleString()}
              subtitle={`${positivePct}% of total volume`}
              icon={Smile}
              colorScheme="emerald"
            />
            <MetricsCard
              title="Negative Polarity"
              value={negativeCount.toLocaleString()}
              subtitle={`${negativePct}% of total volume`}
              icon={Frown}
              colorScheme="rose"
            />
            <MetricsCard
              title="Avg Model Confidence"
              value={`${avgConfidence}%`}
              subtitle="Softmax class likelihood"
              icon={Zap}
              colorScheme="purple"
            />
          </div>

          {/* Quick Action Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              onClick={() => navigate('/analyzer')}
              className="p-5 rounded-2xl bg-gradient-to-r from-purple-950/40 to-slate-900 border border-purple-800/30 hover:border-purple-500/50 cursor-pointer transition-all duration-300 group shadow-md"
            >
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <span className="text-[10px] font-mono tracking-widest text-purple-400 uppercase font-semibold">
                    Interactive Inference
                  </span>
                  <h4 className="text-sm font-semibold text-white group-hover:text-purple-300 transition-colors">
                    Single Text Sentiment Classifier
                  </h4>
                  <p className="text-xs text-slate-400">
                    Input custom reviews, tweets, or feedback to see real-time NLP tokens and Softmax probabilities.
                  </p>
                </div>
                <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 group-hover:translate-x-1 transition-transform shrink-0">
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>

            <div
              onClick={() => navigate('/batch')}
              className="p-5 rounded-2xl bg-gradient-to-r from-blue-950/40 to-slate-900 border border-blue-800/30 hover:border-blue-500/50 cursor-pointer transition-all duration-300 group shadow-md"
            >
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <span className="text-[10px] font-mono tracking-widest text-blue-400 uppercase font-semibold">
                    High-Throughput
                  </span>
                  <h4 className="text-sm font-semibold text-white group-hover:text-blue-300 transition-colors">
                    Batch CSV File Processor
                  </h4>
                  <p className="text-xs text-slate-400">
                    Upload datasets with thousands of customer remarks to process classifications and download annotated CSVs.
                  </p>
                </div>
                <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-400 group-hover:translate-x-1 transition-transform shrink-0">
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
              </div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <TrendsChart
                data={trends}
                title="Sentiment Activity Timeline"
                subtitle="Historical aggregation across positive, negative, and neutral classifications"
              />
            </div>
            <div>
              <SentimentPieChart
                data={distribution}
                title="Polarity Distribution"
                subtitle="Proportion of total analyzed dataset records"
              />
            </div>
          </div>

          {/* Secondary Comparative Breakdown */}
          <div>
            <ComparisonBarChart
              data={distribution}
              title="Class Volume Comparison"
              subtitle="Direct count comparison of positive, neutral, and negative items"
            />
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
