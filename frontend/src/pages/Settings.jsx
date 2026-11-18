import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Moon,
  Sun,
  Database,
  Server,
  Code2,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  BookOpen,
  RefreshCw,
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import api, { API_BASE_URL } from '../services/api';
import Card from '../components/common/Card';
import Button from '../components/common/Button';

export const Settings = () => {
  const { theme, toggleTheme } = useTheme();
  const [healthStatus, setHealthStatus] = useState(null);
  const [testingHealth, setTestingHealth] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const apiBaseUrl = API_BASE_URL;

  const checkConnectivity = async () => {
    setTestingHealth(true);
    setTestResult(null);
    try {
      const res = await api.checkHealth();
      setHealthStatus(res);
      setTestResult({
        success: true,
        message: `Connected successfully! Status: ${res.status}, Model: ${res.active_model}`,
      });
    } catch (err) {
      setTestResult({
        success: false,
        message: `Connection failed: ${err.message}`,
      });
    } finally {
      setTestingHealth(false);
    }
  };

  useEffect(() => {
    checkConnectivity();
  }, []);

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          System Settings & Platform Architecture
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Configure client preferences, verify backend telemetry, and review academic system architecture.
        </p>
      </div>

      {/* Appearance Settings */}
      <Card title="Interface Appearance" subtitle="Choose your visual theme preference">
        <div className="flex items-center justify-between p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400">
              {theme === 'dark' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white capitalize">
                {theme} Mode
              </h4>
              <p className="text-xs text-slate-400">
                {theme === 'dark' ? 'Dark navy background with charcoal surfaces' : 'High-contrast light background'}
              </p>
            </div>
          </div>

          <Button variant="secondary" size="sm" onClick={toggleTheme}>
            Switch to {theme === 'dark' ? 'Light' : 'Dark'} Mode
          </Button>
        </div>
      </Card>

      {/* API Connectivity & Backend Diagnostics */}
      <Card
        title="Backend API Connectivity"
        subtitle="REST endpoints linking React frontend to FastAPI"
      >
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
            <div className="flex justify-between items-center py-1 border-b border-slate-800/80">
              <span className="text-slate-400">API Gateway Endpoint</span>
              <code className="text-purple-400 font-mono">{apiBaseUrl}</code>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Database Engine</span>
              <span className="text-slate-200 font-mono">SQLite (SQLAlchemy 2.0 ORM)</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-slate-400">Interactive Documentation</span>
              <a
                href={API_BASE_URL.replace('/api/v1', '/docs')}
                target="_blank"
                rel="noreferrer"
                className="text-blue-400 hover:text-blue-300 font-mono flex items-center gap-1"
              >
                <span>/docs (Swagger UI)</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {testResult && (
            <div
              className={`p-3.5 rounded-xl border flex items-center gap-2.5 text-xs ${
                testResult.success
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
              }`}
            >
              {testResult.success ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              ) : (
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              )}
              <span>{testResult.message}</span>
            </div>
          )}

          <div className="flex justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={checkConnectivity}
              isLoading={testingHealth}
            >
              <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
              Test API Connection
            </Button>
          </div>
        </div>
      </Card>

      {/* Academic Project Credits */}
      <Card
        title="Academic Course Project Information"
        subtitle="CSE276 – Artificial Intelligence Foundations"
      >
        <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-800/30 space-y-3 text-xs">
          <div className="flex items-center gap-2 text-purple-300 font-semibold text-sm">
            <BookOpen className="w-4 h-4" />
            <span>SENTINEL AI Architecture Overview</span>
          </div>
          <p className="text-slate-300 leading-relaxed font-sans">
            This platform demonstrates an end-to-end production AI system developed across 5 rigorous engineering phases:
          </p>
          <ul className="list-disc list-inside text-slate-400 space-y-1 font-mono text-[11px]">
            <li>Phase 1: System Architecture, Scaffolding, and Data Flow Design</li>
            <li>Phase 2: Dataset Acquisition (CrowdFlower US Airlines) & Negation-Aware NLP Cleaning</li>
            <li>Phase 3: TF-IDF Feature Extraction & Multinomial Logistic Regression Model Training</li>
            <li>Phase 4: FastAPI REST Service, SQLAlchemy SQLite Persistence, and Automated Testing</li>
            <li>Phase 5: Modern SaaS React 18 / Vite / Tailwind Dashboard & Visual Telemetry</li>
          </ul>
        </div>
      </Card>
    </div>
  );
};

export default Settings;
