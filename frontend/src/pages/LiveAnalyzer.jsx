import React, { useState } from 'react';
import {
  Send,
  Trash2,
  Sparkles,
  AlertCircle,
  HelpCircle,
  History as HistoryIcon,
} from 'lucide-react';
import api from '../services/api';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import ResultCard from '../components/sentiment/ResultCard';

const SAMPLE_TEXTS = [
  {
    label: 'Positive Experience',
    text: 'I absolutely love this airline! The flight crew was outstanding, seats were very comfortable, and we arrived 20 minutes early.',
  },
  {
    label: 'Negative Complaint',
    text: 'Terrible customer service. Our luggage was completely lost, the staff was extremely rude, and no one offered any assistance.',
  },
  {
    label: 'Neutral Statement',
    text: 'The scheduled flight UA104 departs at 4:15 PM from terminal 3, gate B12.',
  },
];

export const LiveAnalyzer = () => {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState([]);

  const maxChars = 5000;
  const charsLeft = maxChars - inputText.length;

  const handleAnalyze = async () => {
    const trimmed = inputText.trim();
    if (!trimmed) {
      setError('Please enter some text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await api.predictSentiment(trimmed);
      setCurrentResult(res);
      setRecentAnalyses((prev) => [res, ...prev.slice(0, 4)]);
    } catch (err) {
      setError(err.message || 'Sentiment prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInputText('');
    setCurrentResult(null);
    setError(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleAnalyze();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          Live Sentiment Analyzer
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Perform real-time NLP classification using the champion Logistic Regression model.
        </p>
      </div>

      {/* Main Grid: Input Form & Result Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Input Form (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <Card
            title="Text Input"
            subtitle="Enter any remark, review, tweet, or message (up to 5,000 characters)"
          >
            {/* Sample Prompts */}
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                Quick Samples:
              </span>
              {SAMPLE_TEXTS.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInputText(sample.text);
                    setError(null);
                  }}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700/80 text-slate-300 hover:text-white hover:border-purple-500/50 transition-colors"
                >
                  {sample.label}
                </button>
              ))}
            </div>

            {/* Textarea */}
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => {
                  setInputText(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={handleKeyDown}
                rows={7}
                maxLength={maxChars}
                placeholder="Type or paste customer text here... (Ctrl+Enter to analyze)"
                className="w-full p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/40 focus:border-purple-500/80 transition-all font-sans leading-relaxed resize-y"
              />

              <div className="flex items-center justify-between mt-2 text-xs">
                <span className="text-slate-500 font-mono">
                  {charsLeft} characters remaining
                </span>
                <span className="text-slate-500 hidden sm:inline text-[11px]">
                  Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono">Ctrl+Enter</kbd>
                </span>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mt-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-2 text-xs text-rose-300">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Actions */}
            <div className="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClear}
                disabled={!inputText && !currentResult}
                className="text-slate-400 hover:text-rose-400"
              >
                <Trash2 className="w-4 h-4 mr-1.5" />
                Clear
              </Button>

              <Button
                variant="primary"
                size="md"
                onClick={handleAnalyze}
                isLoading={loading}
                disabled={!inputText.trim()}
                className="px-6"
              >
                <Send className="w-4 h-4 mr-2" />
                Analyze Sentiment
              </Button>
            </div>
          </Card>
        </div>

        {/* Right Column: Prediction Result (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {currentResult ? (
            <ResultCard result={currentResult} />
          ) : (
            <Card className="text-center py-16 border-dashed border-slate-800">
              <div className="w-14 h-14 mx-auto rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-3 shadow-inner">
                <Sparkles className="w-7 h-7" />
              </div>
              <h3 className="text-sm font-semibold text-white mb-1">
                Awaiting Input
              </h3>
              <p className="text-xs text-slate-400 max-w-xs mx-auto">
                Enter text and click <strong className="text-purple-400">Analyze Sentiment</strong> to inspect class predictions and Softmax confidence.
              </p>
            </Card>
          )}

          {/* Session History Mini-List */}
          {recentAnalyses.length > 1 && (
            <Card
              title="Recent Inferences"
              subtitle="Quick review of queries from this active session"
              className="text-xs"
            >
              <div className="space-y-2 mt-2">
                {recentAnalyses.slice(1).map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() => {
                      setInputText(item.text);
                      setCurrentResult(item);
                    }}
                    className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700 flex items-center justify-between gap-3 cursor-pointer transition-colors"
                  >
                    <p className="truncate text-slate-300 font-sans flex-1">
                      "{item.text}"
                    </p>
                    <span
                      className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md shrink-0 ${
                        item.sentiment === 'positive'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : item.sentiment === 'negative'
                          ? 'bg-rose-500/20 text-rose-400'
                          : 'bg-amber-500/20 text-amber-400'
                      }`}
                    >
                      {item.sentiment}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveAnalyzer;
