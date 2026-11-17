import React, { useState } from 'react';
import { Clock, Zap, CheckCircle2, Copy, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import Card from '../common/Card';
import SentimentBadge from './SentimentBadge';
import ProbabilityBar from './ProbabilityBar';
import Button from '../common/Button';

export const ResultCard = ({ result, className = '' }) => {
  const [copied, setCopied] = useState(false);
  const [showTokens, setShowTokens] = useState(false);

  if (!result) return null;

  const {
    text,
    cleaned_text,
    sentiment,
    confidence = 0,
    probabilities = {},
    processing_time_ms = 0,
    created_at,
  } = result;

  const handleCopy = () => {
    navigator.clipboard.writeText(
      `Text: "${text}"\nSentiment: ${sentiment}\nConfidence: ${(confidence * 100).toFixed(2)}%`
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const confidencePct = (confidence * 100).toFixed(1);

  return (
    <Card className={`relative overflow-hidden border-slate-700/80 ${className}`}>
      {/* Top Banner Accent */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${
        sentiment === 'positive'
          ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
          : sentiment === 'negative'
          ? 'bg-gradient-to-r from-rose-500 to-red-400'
          : 'bg-gradient-to-r from-amber-500 to-yellow-400'
      }`} />

      {/* Header Info */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <SentimentBadge sentiment={sentiment} size="lg" />
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-300 text-xs font-mono font-medium">
            <Zap className="w-3.5 h-3.5 text-purple-400" />
            <span>{confidencePct}% Confidence</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 text-xs text-slate-400 bg-slate-800/60 px-2.5 py-1 rounded-lg border border-slate-700/50 font-mono">
            <Clock className="w-3 h-3 text-slate-400" />
            <span>{processing_time_ms.toFixed(2)} ms</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="p-1.5 text-slate-400 hover:text-white"
            title="Copy Result"
          >
            {copied ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
          </Button>
        </div>
      </div>

      {/* Input Text Box */}
      <div className="mb-6">
        <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
          Analyzed Input
        </label>
        <p className="text-sm text-slate-200 bg-slate-900/60 p-3.5 rounded-xl border border-slate-800/80 leading-relaxed font-sans select-text">
          "{text}"
        </p>
      </div>

      {/* NLP Cleaned Preview Toggle */}
      {cleaned_text && (
        <div className="mb-6">
          <button
            onClick={() => setShowTokens(!showTokens)}
            className="flex items-center justify-between w-full text-xs text-slate-400 hover:text-slate-200 py-1.5 transition-colors"
          >
            <span className="flex items-center gap-1.5 font-medium">
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              NLP Preprocessed Tokens
            </span>
            {showTokens ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showTokens && (
            <div className="mt-2 p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-xs font-mono text-purple-300 break-words leading-relaxed animate-fadeIn">
              {cleaned_text}
            </div>
          )}
        </div>
      )}

      {/* Softmax Probabilities */}
      <div className="pt-2">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Model Softmax Probabilities
        </h4>
        <ProbabilityBar probabilities={probabilities} />
      </div>
    </Card>
  );
};

export default ResultCard;
