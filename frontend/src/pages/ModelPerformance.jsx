import React, { useState, useEffect } from 'react';
import {
  Cpu,
  CheckCircle2,
  AlertCircle,
  Database,
  BarChart,
  Zap,
  Layers,
  Award,
  RefreshCw,
} from 'lucide-react';
import api from '../services/api';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import Spinner from '../components/common/Spinner';

export const ModelPerformance = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModelData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [infoRes, metricsRes] = await Promise.all([
        api.getModelInfo(),
        api.getModelMetrics(),
      ]);
      setModelInfo(infoRes);
      setMetrics(metricsRes);
    } catch (err) {
      setError(err.message || 'Failed to load model diagnostics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelData();
  }, []);

  const classes = ['negative', 'neutral', 'positive'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Model Performance & Diagnostics
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Supervised machine learning validation metrics, confusion matrix, and feature extraction parameters.
          </p>
        </div>

        <Button
          variant="secondary"
          size="sm"
          onClick={fetchModelData}
          isLoading={loading}
          className="flex items-center gap-1.5 self-start sm:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Reload Metrics</span>
        </Button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-3 text-xs text-rose-300">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading && !metrics ? (
        <div className="py-24 flex items-center justify-center">
          <Spinner size="lg" label="Retrieving serialized ML model evaluation data..." />
        </div>
      ) : (
        <>
          {/* Top KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Holdout Accuracy
              </span>
              <h3 className="text-2xl font-bold text-emerald-400 font-mono mt-1">
                {((metrics?.accuracy || 0) * 100).toFixed(2)}%
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                Evaluated on 2,881 test samples
              </p>
            </Card>

            <Card className="p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Macro F1-Score
              </span>
              <h3 className="text-2xl font-bold text-purple-400 font-mono mt-1">
                {(metrics?.f1_macro || 0).toFixed(4)}
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                Unweighted class average
              </p>
            </Card>

            <Card className="p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Weighted F1-Score
              </span>
              <h3 className="text-2xl font-bold text-blue-400 font-mono mt-1">
                {(metrics?.f1_weighted || 0).toFixed(4)}
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                Support-weighted harmonic mean
              </p>
            </Card>

            <Card className="p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                TF-IDF Features
              </span>
              <h3 className="text-2xl font-bold text-amber-400 font-mono mt-1">
                {(modelInfo?.vocabulary_size || 0).toLocaleString()}
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                Unigrams & Bigrams with sublinear TF
              </p>
            </Card>
          </div>

          {/* Model Specification & Architecture */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card
              title="Architecture Specifications"
              subtitle="Algorithm and serialization metadata"
              className="lg:col-span-1"
            >
              <div className="space-y-3 mt-2 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Champion Model</span>
                  <span className="font-mono text-purple-400 font-semibold">
                    {modelInfo?.model_name || 'LogisticRegression'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Regularization (C)</span>
                  <span className="font-mono text-white">0.5</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Multi-class Strategy</span>
                  <span className="font-mono text-white">Multinomial (Softmax)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Class Weights</span>
                  <span className="font-mono text-white">Balanced</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Solver</span>
                  <span className="font-mono text-white">lbfgs</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Training Samples</span>
                  <span className="font-mono text-white">{modelInfo?.training_dataset_size?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-400">Testing Samples</span>
                  <span className="font-mono text-white">{modelInfo?.testing_dataset_size?.toLocaleString()}</span>
                </div>
              </div>
            </Card>

            {/* Per-Class Evaluation Breakdown */}
            <Card
              title="Class-Level Classification Metrics"
              subtitle="Precision, recall, and F1 across each polarity class"
              className="lg:col-span-2"
            >
              <div className="overflow-x-auto mt-2">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                      <th className="py-3 px-4">Class</th>
                      <th className="py-3 px-4 text-right">Precision</th>
                      <th className="py-3 px-4 text-right">Recall</th>
                      <th className="py-3 px-4 text-right">F1-Score</th>
                      <th className="py-3 px-4 text-right">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-sans">
                    {metrics?.class_metrics &&
                      Object.entries(metrics.class_metrics).map(([clsName, vals]) => {
                        const isNeg = clsName === 'negative';
                        const isPos = clsName === 'positive';
                        const colorClass = isPos
                          ? 'text-emerald-400'
                          : isNeg
                          ? 'text-rose-400'
                          : 'text-amber-400';

                        return (
                          <tr key={clsName} className="hover:bg-slate-900/40">
                            <td className={`py-3 px-4 font-semibold uppercase tracking-wider ${colorClass}`}>
                              {clsName}
                            </td>
                            <td className="py-3 px-4 text-right font-mono text-slate-200">
                              {(vals.precision * 100).toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-right font-mono text-slate-200">
                              {(vals.recall * 100).toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-right font-mono font-semibold text-white">
                              {(vals.f1_score ?? vals['f1-score'] ?? 0).toFixed(4)}
                            </td>
                            <td className="py-3 px-4 text-right font-mono text-slate-400">
                              {vals.support ? Number(vals.support).toLocaleString() : 'N/A'}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          {/* Confusion Matrix Section */}
          <Card
            title="Confusion Matrix (Holdout Test Set)"
            subtitle="Rows represent Ground Truth labels; columns represent Model Predictions"
          >
            <div className="overflow-x-auto mt-4 max-w-xl mx-auto">
              <table className="w-full text-center text-xs border border-slate-800">
                <thead>
                  <tr className="bg-slate-900 border-b border-slate-800">
                    <th className="py-3 px-4 text-slate-400 uppercase font-semibold">
                      Actual \ Predicted
                    </th>
                    {classes.map((c) => (
                      <th key={c} className="py-3 px-4 text-slate-300 uppercase font-mono font-semibold">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  {metrics?.confusion_matrix?.map((row, rowIdx) => {
                    const actualClass = classes[rowIdx];
                    return (
                      <tr key={rowIdx}>
                        <td className="py-3 px-4 font-semibold text-slate-300 bg-slate-900/50 uppercase border-r border-slate-800">
                          {actualClass}
                        </td>
                        {row.map((cellValue, colIdx) => {
                          const isDiagonal = rowIdx === colIdx;
                          return (
                            <td
                              key={colIdx}
                              className={`py-3 px-4 ${
                                isDiagonal
                                  ? 'bg-purple-600/20 text-purple-300 font-bold'
                                  : 'text-slate-400 bg-slate-950/40'
                              }`}
                            >
                              {cellValue.toLocaleString()}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
};

export default ModelPerformance;
