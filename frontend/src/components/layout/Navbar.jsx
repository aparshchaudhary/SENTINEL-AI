import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Sun, Moon, Server, Activity, Menu } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import api from '../../services/api';

const ROUTE_TITLES = {
  '/': 'Executive Dashboard',
  '/analyzer': 'Live Sentiment Analyzer',
  '/batch': 'Batch CSV Processing',
  '/history': 'Analysis History & Logs',
  '/model': 'Model Performance & Diagnostics',
  '/settings': 'System Settings & Architecture',
};

export const Navbar = ({ onMobileMenuToggle }) => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  const [health, setHealth] = useState({ status: 'checking', active_model: 'Detecting...' });

  const currentTitle = ROUTE_TITLES[location.pathname] || 'Sentiment Analysis Platform';

  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const res = await api.checkHealth();
        if (isMounted) {
          setHealth(res);
        }
      } catch (err) {
        if (isMounted) {
          setHealth({ status: 'offline', active_model: 'FastAPI Offline' });
        }
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-16 px-6 bg-surface-dark/80 backdrop-blur-md border-b border-slate-800">
      {/* Left: Mobile trigger & Page Title */}
      <div className="flex items-center gap-4">
        <button
          onClick={onMobileMenuToggle}
          className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 lg:hidden"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          <h1 className="text-base font-semibold text-white tracking-tight">
            {currentTitle}
          </h1>
          <p className="text-[11px] text-slate-400 font-mono hidden sm:block">
            SENTINEL AI &bull; CSE276 AI Foundations
          </p>
        </div>
      </div>

      {/* Right: Status Pill & Theme Toggle */}
      <div className="flex items-center gap-3">
        {/* Backend Status Indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-800 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              health.status === 'healthy'
                ? 'bg-emerald-400 animate-pulse'
                : health.status === 'checking'
                ? 'bg-amber-400'
                : 'bg-rose-500'
            }`}
          />
          <span className="text-slate-300 font-medium hidden md:inline">
            {health.status === 'healthy' ? 'API Connected' : health.status === 'checking' ? 'Connecting...' : 'API Offline'}
          </span>
          {health.active_model && (
            <span className="text-slate-400 text-[10px] font-mono hidden xl:inline border-l border-slate-700/80 pl-2">
              {health.active_model}
            </span>
          )}
        </div>

        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-xl bg-slate-900/80 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-purple-400" />}
        </button>
      </div>
    </header>
  );
};

export default Navbar;
