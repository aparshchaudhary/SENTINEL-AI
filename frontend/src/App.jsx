import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import AppLayout from './layouts/AppLayout';

import Dashboard from './pages/Dashboard';
import LiveAnalyzer from './pages/LiveAnalyzer';
import BatchAnalyzer from './pages/BatchAnalyzer';
import History from './pages/History';
import ModelPerformance from './pages/ModelPerformance';
import Settings from './pages/Settings';

export function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="analyzer" element={<LiveAnalyzer />} />
            <Route path="batch" element={<BatchAnalyzer />} />
            <Route path="history" element={<History />} />
            <Route path="model" element={<ModelPerformance />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
