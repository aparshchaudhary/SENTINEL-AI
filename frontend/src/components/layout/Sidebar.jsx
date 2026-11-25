import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  MessageSquareText,
  FileSpreadsheet,
  History,
  Activity,
  Settings,
  ShieldCheck,
  X,
  ExternalLink,
} from 'lucide-react';
import { API_BASE_URL } from '../../services/api';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Live Analyzer', path: '/analyzer', icon: MessageSquareText },
  { name: 'Batch CSV Analyzer', path: '/batch', icon: FileSpreadsheet },
  { name: 'Analysis History', path: '/history', icon: History },
  { name: 'Model Performance', path: '/model', icon: Activity },
  { name: 'Settings & Info', path: '/settings', icon: Settings },
];

export const Sidebar = ({ isOpen, onClose }) => {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden animate-fadeIn"
          onClick={onClose}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-surface-dark border-r border-slate-800 flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top Header */}
        <div>
          <div className="flex items-center justify-between h-16 px-6 border-b border-slate-800">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-purple-500/20">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <span className="text-sm font-bold tracking-wider text-white uppercase font-sans">
                  SENTINEL <span className="text-purple-400 font-extrabold">AI</span>
                </span>
                <span className="block text-[10px] text-slate-400 font-mono">NLP Analytics</span>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-1 rounded-lg text-slate-400 hover:text-white lg:hidden"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="p-4 space-y-1.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={() => onClose && onClose()}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 ${
                      isActive
                        ? 'bg-purple-600/15 text-purple-400 border border-purple-500/30 shadow-sm font-semibold'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.name}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-800 space-y-3">
          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800/80">
            <div className="flex items-center justify-between text-[11px] mb-1">
              <span className="text-slate-400 font-medium">Champion Model</span>
              <span className="text-emerald-400 font-mono font-bold">79.24%</span>
            </div>
            <p className="text-[10px] text-slate-300 font-mono truncate">
              LogisticRegression (C=1.25, Balanced)
            </p>
          </div>

          <a
            href={API_BASE_URL.replace('/api/v1', '/docs')}
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-between px-3 py-2 text-[11px] text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/60 transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <span>FastAPI Docs</span>
            </span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
