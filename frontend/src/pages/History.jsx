import React, { useState, useEffect, useCallback } from 'react';
import {
  Search,
  Filter,
  Trash2,
  Calendar,
  AlertTriangle,
  RefreshCw,
  Clock,
  Layers,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';
import api from '../services/api';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import SentimentBadge from '../components/sentiment/SentimentBadge';
import Modal from '../components/common/Modal';
import EmptyState from '../components/common/EmptyState';
import Spinner from '../components/common/Spinner';

export const History = () => {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('');

  // Delete Modal State
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        page,
        page_size: pageSize,
      };
      if (sentimentFilter) params.sentiment = sentimentFilter;
      if (searchTerm.trim()) params.search = searchTerm.trim();

      const res = await api.getHistory(params);
      setItems(res.items || []);
      setTotal(res.total || 0);
      setTotalPages(res.total_pages || 1);
    } catch (err) {
      setError(err.message || 'Failed to fetch analysis history.');
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, sentimentFilter, searchTerm]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchHistory();
  };

  const handleFilterChange = (val) => {
    setSentimentFilter(val);
    setPage(1);
  };

  const confirmDelete = async () => {
    if (!itemToDelete) return;
    setIsDeleting(true);
    try {
      await api.deleteHistory(itemToDelete.id);
      setDeleteModalOpen(false);
      setItemToDelete(null);
      // Refresh current page
      fetchHistory();
    } catch (err) {
      alert(`Deletion failed: ${err.message}`);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Analysis History & Logs
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Audit log of all sentiment analyses persisted to SQLite database ({total} total records).
          </p>
        </div>

        <Button
          variant="secondary"
          size="sm"
          onClick={fetchHistory}
          isLoading={loading}
          className="flex items-center gap-1.5 self-start sm:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh</span>
        </Button>
      </div>

      {/* Filter and Search Bar */}
      <Card className="p-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Search Form */}
          <form onSubmit={handleSearchSubmit} className="relative flex-1 w-full">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search input text content..."
              className="w-full pl-10 pr-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-purple-500"
            />
          </form>

          {/* Sentiment Filter Dropdown */}
          <div className="flex items-center gap-3 w-full md:w-auto">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-400 shrink-0" />
              <select
                value={sentimentFilter}
                onChange={(e) => handleFilterChange(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-purple-500"
              >
                <option value="">All Sentiments</option>
                <option value="positive">Positive Only</option>
                <option value="neutral">Neutral Only</option>
                <option value="negative">Negative Only</option>
              </select>
            </div>
          </div>
        </div>
      </Card>

      {/* History Table */}
      <Card className="overflow-hidden p-0">
        {loading && items.length === 0 ? (
          <div className="py-20 flex items-center justify-center">
            <Spinner size="md" label="Loading historical records..." />
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            title="No Records Found"
            description={
              searchTerm || sentimentFilter
                ? 'No analysis matches your active search or filter criteria.'
                : 'No sentiment queries have been run yet. Use the Live Analyzer to create records.'
            }
            className="border-none py-16"
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="bg-slate-900/60 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                  <th className="py-3.5 px-4 w-14">ID</th>
                  <th className="py-3.5 px-4">Input Text</th>
                  <th className="py-3.5 px-4 w-32">Sentiment</th>
                  <th className="py-3.5 px-4 w-24 text-right">Confidence</th>
                  <th className="py-3.5 px-4 w-24 text-right">Latency</th>
                  <th className="py-3.5 px-4 w-36">Timestamp</th>
                  <th className="py-3.5 px-4 w-16 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-sans">
                {items.map((row) => (
                  <tr key={row.id} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-3.5 px-4 text-slate-500 font-mono">
                      #{row.id}
                    </td>
                    <td className="py-3.5 px-4 max-w-sm truncate text-slate-200">
                      "{row.input_text}"
                    </td>
                    <td className="py-3.5 px-4">
                      <SentimentBadge sentiment={row.predicted_sentiment} size="sm" />
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-purple-400 font-medium">
                      {(row.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3.5 px-4 text-right font-mono text-slate-400">
                      {row.processing_time_ms.toFixed(1)}ms
                    </td>
                    <td className="py-3.5 px-4 text-slate-400 text-[11px] font-mono">
                      {new Date(row.created_at).toLocaleString()}
                    </td>
                    <td className="py-3.5 px-4 text-center">
                      <button
                        onClick={() => {
                          setItemToDelete(row);
                          setDeleteModalOpen(true);
                        }}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        title="Delete record"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 bg-slate-900/40 border-t border-slate-800 text-xs text-slate-400">
            <div>
              Showing page <span className="text-white font-mono">{page}</span> of{' '}
              <span className="text-white font-mono">{totalPages}</span> ({total} items)
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                Next
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Confirm Record Deletion"
        footer={
          <>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDeleteModalOpen(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={confirmDelete}
              isLoading={isDeleting}
            >
              Delete Record
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <div className="flex items-center gap-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
            <span>
              This will permanently delete this analysis record from the SQLite database.
            </span>
          </div>

          {itemToDelete && (
            <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 text-xs">
              <span className="text-slate-400 block mb-1">Record #{itemToDelete.id}</span>
              <p className="text-slate-200 italic font-sans">
                "{itemToDelete.input_text}"
              </p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
};

export default History;
