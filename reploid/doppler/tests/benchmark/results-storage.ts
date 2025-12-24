/**
 * Results Storage - Save and Load Benchmark Results
 *
 * Provides utilities to:
 * - Save benchmark results to JSON files
 * - Load and compare historical results
 * - Generate comparison reports
 *
 * @module tests/benchmark/results-storage
 */

import type { BenchmarkResult, BenchmarkSession } from './types.js';
import type { SystemBenchmarkResult } from './system-benchmark.js';

// ============================================================================
// Result File Naming
// ============================================================================

/**
 * Generate a filename for a benchmark result.
 * Format: {suite}_{model}_{timestamp}.json
 */
export function generateResultFilename(result: BenchmarkResult | SystemBenchmarkResult): string {
  const timestamp = result.timestamp.replace(/[:.]/g, '-').slice(0, 19);
  const suite = result.suite;

  let modelId = 'unknown';
  if ('model' in result && result.model) {
    modelId = (result.model.modelName ?? result.model.modelId ?? 'unknown')
      .toLowerCase()
      .replace(/[^a-z0-9]/g, '-')
      .slice(0, 30);
  }

  return `${suite}_${modelId}_${timestamp}.json`;
}

/**
 * Generate a session filename.
 * Format: session_{sessionId}_{timestamp}.json
 */
export function generateSessionFilename(session: BenchmarkSession): string {
  const timestamp = session.startTime.replace(/[:.]/g, '-').slice(0, 19);
  return `session_${session.sessionId}_${timestamp}.json`;
}

// ============================================================================
// Browser Storage (IndexedDB)
// ============================================================================

const DB_NAME = 'doppler_benchmarks';
const DB_VERSION = 1;
const STORE_NAME = 'results';

/**
 * Open the IndexedDB database.
 */
async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('suite', 'suite', { unique: false });
        store.createIndex('modelId', 'modelId', { unique: false });
      }
    };
  });
}

/**
 * Save a benchmark result to IndexedDB.
 */
export async function saveResult(result: BenchmarkResult | SystemBenchmarkResult): Promise<number> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    const entry = {
      ...result,
      modelId: 'model' in result ? result.model?.modelId : undefined,
    };

    const request = store.add(entry);
    request.onsuccess = () => resolve(request.result as number);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load all benchmark results from IndexedDB.
 */
export async function loadAllResults(): Promise<(BenchmarkResult | SystemBenchmarkResult)[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load results filtered by suite.
 */
export async function loadResultsBySuite(suite: 'kernel' | 'pipeline' | 'system'): Promise<(BenchmarkResult | SystemBenchmarkResult)[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('suite');
    const request = index.getAll(suite);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load results for a specific model.
 */
export async function loadResultsByModel(modelId: string): Promise<BenchmarkResult[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('modelId');
    const request = index.getAll(modelId);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Clear all stored results.
 */
export async function clearAllResults(): Promise<void> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.clear();

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// ============================================================================
// JSON Export/Import
// ============================================================================

/**
 * Export results to JSON string.
 */
export function exportToJSON(results: (BenchmarkResult | SystemBenchmarkResult)[]): string {
  return JSON.stringify(results, null, 2);
}

/**
 * Export a single result to JSON string.
 */
export function exportResultToJSON(result: BenchmarkResult | SystemBenchmarkResult): string {
  return JSON.stringify(result, null, 2);
}

/**
 * Import results from JSON string.
 */
export function importFromJSON(json: string): (BenchmarkResult | SystemBenchmarkResult)[] {
  const parsed = JSON.parse(json);
  return Array.isArray(parsed) ? parsed : [parsed];
}

/**
 * Download results as a JSON file (browser only).
 */
export function downloadAsJSON(
  results: BenchmarkResult | SystemBenchmarkResult | (BenchmarkResult | SystemBenchmarkResult)[],
  filename?: string
): void {
  const data = Array.isArray(results) ? results : [results];
  const json = exportToJSON(data);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const defaultFilename = Array.isArray(results)
    ? `benchmark_results_${new Date().toISOString().slice(0, 10)}.json`
    : generateResultFilename(results);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename ?? defaultFilename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// Comparison Utilities
// ============================================================================

export interface ComparisonDelta {
  metric: string;
  baseline: number;
  current: number;
  delta: number;
  deltaPercent: number;
  improved: boolean;
}

/**
 * Compare two pipeline benchmark results.
 */
export function comparePipelineResults(
  baseline: BenchmarkResult,
  current: BenchmarkResult
): ComparisonDelta[] {
  const deltas: ComparisonDelta[] = [];

  const metrics: { key: keyof BenchmarkResult['metrics']; lowerIsBetter: boolean }[] = [
    { key: 'ttft_ms', lowerIsBetter: true },
    { key: 'prefill_ms', lowerIsBetter: true },
    { key: 'decode_ms_total', lowerIsBetter: true },
    { key: 'prefill_tokens_per_sec', lowerIsBetter: false },
    { key: 'decode_tokens_per_sec', lowerIsBetter: false },
    { key: 'gpu_submit_count_prefill', lowerIsBetter: true },
    { key: 'gpu_submit_count_decode', lowerIsBetter: true },
    { key: 'estimated_vram_bytes_peak', lowerIsBetter: true },
  ];

  for (const { key, lowerIsBetter } of metrics) {
    const baseVal = baseline.metrics[key];
    const currVal = current.metrics[key];

    if (typeof baseVal === 'number' && typeof currVal === 'number') {
      const delta = currVal - baseVal;
      const deltaPercent = baseVal !== 0 ? (delta / baseVal) * 100 : 0;
      const improved = lowerIsBetter ? delta < 0 : delta > 0;

      deltas.push({
        metric: key,
        baseline: baseVal,
        current: currVal,
        delta,
        deltaPercent,
        improved,
      });
    }
  }

  return deltas;
}

/**
 * Format comparison as readable string.
 */
export function formatComparison(deltas: ComparisonDelta[]): string {
  const lines = ['=== Benchmark Comparison ===', ''];

  for (const d of deltas) {
    const sign = d.delta >= 0 ? '+' : '';
    const arrow = d.improved ? '✓' : '✗';
    const pct = `${sign}${d.deltaPercent.toFixed(1)}%`;

    lines.push(`${arrow} ${d.metric}: ${d.baseline} → ${d.current} (${pct})`);
  }

  const improved = deltas.filter(d => d.improved).length;
  const regressed = deltas.filter(d => !d.improved).length;

  lines.push('');
  lines.push(`Summary: ${improved} improved, ${regressed} regressed`);

  return lines.join('\n');
}

// ============================================================================
// Session Management
// ============================================================================

/**
 * Create a new benchmark session.
 */
export function createSession(): BenchmarkSession {
  return {
    sessionId: crypto.randomUUID?.() ?? `session_${Date.now()}`,
    startTime: new Date().toISOString(),
    results: [],
  };
}

/**
 * Add a result to a session.
 */
export function addResultToSession(
  session: BenchmarkSession,
  result: BenchmarkResult
): void {
  session.results.push(result);
}

/**
 * Compute session summary.
 */
export function computeSessionSummary(session: BenchmarkSession): BenchmarkSession['summary'] {
  const pipelineResults = session.results.filter(r => r.suite === 'pipeline');

  if (pipelineResults.length === 0) {
    return {
      totalRuns: session.results.length,
      successfulRuns: session.results.length,
      failedRuns: 0,
      avgTtftMs: 0,
      avgDecodeTokensPerSec: 0,
    };
  }

  const ttfts = pipelineResults.map(r => r.metrics.ttft_ms);
  const decodeSpeeds = pipelineResults.map(r => r.metrics.decode_tokens_per_sec);

  return {
    totalRuns: session.results.length,
    successfulRuns: session.results.length,
    failedRuns: 0,
    avgTtftMs: ttfts.reduce((a, b) => a + b, 0) / ttfts.length,
    avgDecodeTokensPerSec: decodeSpeeds.reduce((a, b) => a + b, 0) / decodeSpeeds.length,
  };
}
