import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  Download,
  Clock,
  Zap,
  BarChart2,
  RefreshCw,
  FileText,
  Sparkles,
} from 'lucide-react';
import api from '../services/api';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import SentimentBadge from '../components/sentiment/SentimentBadge';
import Spinner from '../components/common/Spinner';

export const BatchAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [textColumn, setTextColumn] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.name.endsWith('.csv')) {
      setError('Please select a valid CSV file (.csv extension).');
      return;
    }
    setFile(selectedFile);
    setError(null);
    setBatchResult(null);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUploadAndProcess = async () => {
    if (!file) {
      setError('Please select a CSV file first.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setUploadProgress(0);

    try {
      const res = await api.uploadBatchCSV(
        file,
        textColumn.trim() || null,
        (progress) => setUploadProgress(progress)
      );
      setBatchResult(res);
    } catch (err) {
      setError(err.message || 'Batch CSV processing failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLoadDemoData = () => {
    const csvContent = `review_id,text
1,"I had an amazing flight and the flight attendants were delightful and attentive!"
2,"Lost my luggage and the customer service staff was extremely rude and dismissive."
3,"The flight was scheduled for 3:00 PM and departed on time from gate B12."
4,"Best airline experience ever! Highly recommended for long-haul international travel."
5,"Never flying with this airline again. Complete nightmare with a 6-hour delay."
6,"The legroom in economy class was standard, and complimentary water was served."
7,"Seats were surprisingly spacious, in-flight WiFi was super fast, and meals were fresh!"
8,"Terrible experience. Broken entertainment screens, cold food, and seats won't recline."
9,"Can someone confirm which terminal the connecting flight from Denver arrives at?"
10,"The cabin crew was cheerful and courteous throughout the entire 8 hour journey."
11,"Flight canceled at midnight without any hotel accommodation or meal vouchers provided."
12,"Aircraft departed on schedule and arrived at destination 10 minutes ahead of time."
13,"Super comfortable business class lie-flat seats and exceptional champagne service!"
14,"Worst airline ever! They charged me twice for baggage and refused to process a refund."
15,"Boarding began at 14:15 according to passenger boarding pass zone numbers."
16,"The journey was not bad at all, actually quite relaxing and not delayed."
17,"Extremely frustrated with the gate change that was never announced over the intercom."
18,"Luggage carousel 5 is designated for international arrivals from Frankfurt."
19,"Smooth landing, zero turbulence, and luggage was ready at baggage claim in 5 minutes."
20,"Flight crew ignored call buttons for over an hour. Very unprofessional behavior."`;
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const demoFile = new File([blob], 'sample_airline_reviews.csv', { type: 'text/csv' });
    setFile(demoFile);
    setError(null);
    setBatchResult(null);
  };

  const handleReset = () => {
    setFile(null);
    setTextColumn('');
    setBatchResult(null);
    setError(null);
    setUploadProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          Batch CSV Analyzer
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload bulk datasets for high-throughput automated classification and annotated CSV export.
        </p>
      </div>

      {/* Main Upload Area */}
      {!batchResult ? (
        <div className="max-w-2xl mx-auto space-y-4">
          <Card title="Upload CSV Dataset">
            {/* Quick Demo Helper Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-xl bg-purple-500/10 border border-purple-500/20 mb-4 text-xs">
              <div className="flex items-center gap-2 text-purple-300">
                <Sparkles className="w-4 h-4 text-purple-400 shrink-0" />
                <span>Need a test file? Try our curated 20-row airline reviews dataset.</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleLoadDemoData}
                  className="px-2.5 py-1 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium transition-colors shadow-sm"
                >
                  Load Demo File (1-Click)
                </button>
                <a
                  href="/sample_airline_reviews.csv"
                  download="sample_airline_reviews.csv"
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1 font-medium transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download CSV
                </a>
              </div>
            </div>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`p-10 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
                isDragging
                  ? 'border-purple-500 bg-purple-500/10'
                  : file
                  ? 'border-emerald-500/50 bg-emerald-500/5'
                  : 'border-slate-800 hover:border-purple-500/50 hover:bg-slate-900/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => handleFileSelect(e.target.files[0])}
              />

              <div
                className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-3 ${
                  file
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-purple-500/10 text-purple-400'
                }`}
              >
                {file ? <FileSpreadsheet className="w-7 h-7" /> : <UploadCloud className="w-7 h-7" />}
              </div>

              {file ? (
                <div>
                  <h4 className="text-sm font-semibold text-white flex items-center justify-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    {file.name}
                  </h4>
                  <p className="text-xs text-slate-400 mt-1 font-mono">
                    {(file.size / 1024).toFixed(1)} KB &bull; Ready for processing
                  </p>
                </div>
              ) : (
                <div>
                  <h4 className="text-sm font-semibold text-white">
                    Drag and drop your CSV file here, or browse
                  </h4>
                  <p className="text-xs text-slate-400 mt-1">
                    Accepts .csv files containing customer reviews, tweets, or feedback records
                  </p>
                </div>
              )}
            </div>

            {/* Optional Text Column Input */}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                Target Column Name <span className="text-slate-500 font-normal">(Optional)</span>
              </label>
              <input
                type="text"
                value={textColumn}
                onChange={(e) => setTextColumn(e.target.value)}
                placeholder="e.g. text, comment, review (auto-detected if blank)"
                className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-purple-500"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-2 text-xs text-rose-300">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Progress Bar */}
            {isProcessing && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Uploading & Classifying...</span>
                  <span className="font-mono">{uploadProgress}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="mt-5 pt-4 border-t border-slate-800 flex items-center justify-end gap-3">
              {file && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleReset}
                  disabled={isProcessing}
                >
                  Reset
                </Button>
              )}
              <Button
                variant="primary"
                size="md"
                onClick={handleUploadAndProcess}
                isLoading={isProcessing}
                disabled={!file}
              >
                Start Batch Inference
              </Button>
            </div>
          </Card>
        </div>
      ) : (
        /* Results Section */
        <div className="space-y-6 animate-fadeIn">
          {/* Summary Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="text-center p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Processed Records
              </span>
              <h3 className="text-2xl font-bold text-white font-mono mt-1">
                {batchResult.processed_records.toLocaleString()}
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                of {batchResult.total_records.toLocaleString()} total rows
              </p>
            </Card>

            <Card className="text-center p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Positive Ratio
              </span>
              <h3 className="text-2xl font-bold text-emerald-400 font-mono mt-1">
                {batchResult.percentages?.positive || 0}%
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                {batchResult.distribution?.positive || 0} records
              </p>
            </Card>

            <Card className="text-center p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Negative Ratio
              </span>
              <h3 className="text-2xl font-bold text-rose-400 font-mono mt-1">
                {batchResult.percentages?.negative || 0}%
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                {batchResult.distribution?.negative || 0} records
              </p>
            </Card>

            <Card className="text-center p-4">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Processing Time
              </span>
              <h3 className="text-2xl font-bold text-purple-400 font-mono mt-1">
                {batchResult.processing_duration_seconds.toFixed(2)}s
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                Avg {(batchResult.average_confidence * 100).toFixed(1)}% confidence
              </p>
            </Card>
          </div>

          {/* Download Action Banner */}
          <div className="p-5 rounded-2xl bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-xl bg-purple-500/20 text-purple-300">
                <FileSpreadsheet className="w-6 h-6" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Analysis Complete: {batchResult.filename}
                </h4>
                <p className="text-xs text-slate-300">
                  Annotated with <code className="text-purple-300">predicted_sentiment</code> and <code className="text-purple-300">confidence</code>.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="secondary" size="sm" onClick={handleReset}>
                <RefreshCw className="w-4 h-4 mr-1.5" />
                Upload Another
              </Button>

              <a
                href={api.getBatchDownloadUrl(batchResult.job_id)}
                download
                className="inline-flex items-center px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-500 hover:to-blue-500 shadow-md shadow-purple-600/20 transition-all"
              >
                <Download className="w-4 h-4 mr-1.5" />
                Download Annotated CSV
              </a>
            </div>
          </div>

          {/* Preview Table */}
          <Card
            title="Batch Preview"
            subtitle="Initial sample of classified records from the dataset"
          >
            <div className="overflow-x-auto mt-2">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                    <th className="py-3 px-4 w-16">#</th>
                    <th className="py-3 px-4">Input Text</th>
                    <th className="py-3 px-4 w-36">Sentiment</th>
                    <th className="py-3 px-4 w-28 text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-sans">
                  {batchResult.preview_results?.map((row) => (
                    <tr key={row.row_index} className="hover:bg-slate-900/50 transition-colors">
                      <td className="py-3 px-4 text-slate-500 font-mono">
                        {row.row_index}
                      </td>
                      <td className="py-3 px-4 text-slate-200 max-w-md truncate">
                        "{row.text}"
                      </td>
                      <td className="py-3 px-4">
                        <SentimentBadge sentiment={row.sentiment} size="sm" />
                      </td>
                      <td className="py-3 px-4 text-right font-mono text-purple-400 font-medium">
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default BatchAnalyzer;
