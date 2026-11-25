/**
 * SENTINEL AI – Centralized Axios REST API Client
 * Academic Project: CSE276 – Artificial Intelligence Foundations
 */

import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Accept': 'application/json',
  },
});

// Response interceptor for consistent error extraction
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    let errorMessage = 'An unexpected network error occurred.';
    if (error.response) {
      const data = error.response.data;
      if (data && data.detail) {
        errorMessage = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      } else if (data && data.message) {
        errorMessage = data.message;
      } else {
        errorMessage = `Server returned HTTP ${error.response.status}`;
      }
    } else if (error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'))) {
      errorMessage = 'Request timed out waiting for server response. The server may be warming up. Please try again.';
    } else if (error.request) {
      errorMessage = 'Backend service is waking up or unreachable. Please wait a moment and try again.';
    } else {
      errorMessage = error.message;
    }
    return Promise.reject(new Error(errorMessage));
  }
);

export const api = {
  // 1. Health Diagnostics
  checkHealth: () => apiClient.get('/health'),

  // 2. Single Text Prediction
  predictSentiment: (text) => apiClient.post('/sentiment/predict', { text }),

  // 3. Batch CSV Analysis
  uploadBatchCSV: (file, textColumn = null, onProgress = null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (textColumn) {
      formData.append('text_column', textColumn);
    }

    return apiClient.post('/sentiment/batch-csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
  },

  getBatchDownloadUrl: (jobId) => `${API_BASE_URL}/sentiment/batch-csv/download/${jobId}`,

  // 4. Analytics
  getAnalyticsSummary: () => apiClient.get('/analytics/summary'),
  getAnalyticsTrends: () => apiClient.get('/analytics/trends'),
  getAnalyticsDistribution: () => apiClient.get('/analytics/distribution'),

  // 5. History
  getHistory: (params = {}) => apiClient.get('/sentiment/history', { params }),
  getHistoryById: (id) => apiClient.get(`/sentiment/history/${id}`),
  deleteHistory: (id) => apiClient.delete(`/sentiment/history/${id}`),

  // 6. Model Metadata & Metrics
  getModelInfo: () => apiClient.get('/model/info'),
  getModelMetrics: () => apiClient.get('/model/metrics'),
};

export default api;
