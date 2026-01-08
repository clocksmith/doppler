/**
 * Performance Profiler (Tier 2 P0)
 *
 * Real-time profiling utilities to identify performance bottlenecks
 * in the inference pipeline.
 *
 * Usage:
 * 1. Enable profiling: window.DOPPLER_PROFILE = true
 * 2. Run inference
 * 3. View results: log.info('Profile', 'Report', getProfileReport())
 */

import { getDevice } from './device.js';
import { log } from '../debug/index.js';

/** Global profiler state */
let profilingEnabled = false;
/** @type {import('./perf-profiler.js').ProfileEntry[]} */
let profileEntries = [];
let profileStartTime = 0;

/**
 * Check if profiling is enabled
 * @returns {boolean}
 */
export function isProfilingEnabled() {
  if (typeof window !== 'undefined') {
    return Boolean(/** @type {any} */ (window).DOPPLER_PROFILE);
  }
  return profilingEnabled;
}

/**
 * Enable/disable profiling
 * @param {boolean} enabled
 * @returns {void}
 */
export function setProfilingEnabled(enabled) {
  profilingEnabled = enabled;
  if (typeof window !== 'undefined') {
    /** @type {any} */ (window).DOPPLER_PROFILE = enabled;
  }
}

/**
 * Clear all profile entries
 * @returns {void}
 */
export function clearProfile() {
  profileEntries = [];
  profileStartTime = 0;
}

/**
 * Start a new profiling session
 * @returns {void}
 */
export function startProfileSession() {
  clearProfile();
  profileStartTime = performance.now();
}

/**
 * Record a profile entry
 * @param {string} name
 * @param {import('./perf-profiler.js').ProfileEntry['category']} category
 * @param {number} startTime
 * @param {number} endTime
 * @param {Record<string, unknown>} [metadata]
 * @returns {void}
 */
export function recordProfileEntry(
  name,
  category,
  startTime,
  endTime,
  metadata
) {
  if (!isProfilingEnabled()) return;

  profileEntries.push({
    name,
    category,
    startTime,
    endTime,
    duration: endTime - startTime,
    metadata,
  });
}

/**
 * Profile an async operation
 * @template T
 * @param {string} name
 * @param {import('./perf-profiler.js').ProfileEntry['category']} category
 * @param {() => Promise<T>} fn
 * @param {Record<string, unknown>} [metadata]
 * @returns {Promise<T>}
 */
export async function profileAsync(
  name,
  category,
  fn,
  metadata
) {
  if (!isProfilingEnabled()) {
    return fn();
  }

  const startTime = performance.now();
  try {
    const result = await fn();
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, metadata);
    return result;
  } catch (error) {
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, { ...metadata, error: true });
    throw error;
  }
}

/**
 * Profile a sync operation
 * @template T
 * @param {string} name
 * @param {import('./perf-profiler.js').ProfileEntry['category']} category
 * @param {() => T} fn
 * @param {Record<string, unknown>} [metadata]
 * @returns {T}
 */
export function profileSync(
  name,
  category,
  fn,
  metadata
) {
  if (!isProfilingEnabled()) {
    return fn();
  }

  const startTime = performance.now();
  try {
    const result = fn();
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, metadata);
    return result;
  } catch (error) {
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, { ...metadata, error: true });
    throw error;
  }
}

/**
 * Profile a GPU kernel dispatch with queue sync
 * @param {string} name
 * @param {() => void} dispatchFn
 * @param {Record<string, unknown>} [metadata]
 * @returns {Promise<void>}
 */
export async function profileKernel(
  name,
  dispatchFn,
  metadata
) {
  if (!isProfilingEnabled()) {
    dispatchFn();
    return;
  }

  const device = getDevice();
  const startTime = performance.now();

  dispatchFn();

  // Wait for GPU to finish
  await device.queue.onSubmittedWorkDone();

  const endTime = performance.now();
  recordProfileEntry(name, 'kernel', startTime, endTime, metadata);
}

/**
 * Generate profile report
 * @returns {import('./perf-profiler.js').ProfileReport}
 */
export function getProfileReport() {
  const entries = [...profileEntries];
  const totalTime = entries.reduce((sum, e) => sum + e.duration, 0);

  // Calculate category totals
  const kernelEntries = entries.filter(e => e.category === 'kernel');
  const memoryEntries = entries.filter(e => e.category === 'memory');
  const syncEntries = entries.filter(e => e.category === 'sync');
  const otherEntries = entries.filter(e => e.category === 'other');

  const kernelTime = kernelEntries.reduce((sum, e) => sum + e.duration, 0);
  const memoryTime = memoryEntries.reduce((sum, e) => sum + e.duration, 0);
  const syncTime = syncEntries.reduce((sum, e) => sum + e.duration, 0);
  const otherTime = otherEntries.reduce((sum, e) => sum + e.duration, 0);

  // Generate breakdown by operation name
  /** @type {Map<string, { totalTime: number; count: number }>} */
  const byName = new Map();
  for (const entry of entries) {
    const existing = byName.get(entry.name) || { totalTime: 0, count: 0 };
    existing.totalTime += entry.duration;
    existing.count += 1;
    byName.set(entry.name, existing);
  }

  const breakdown = Array.from(byName.entries())
    .map(([name, stats]) => ({
      name,
      totalTime: stats.totalTime,
      count: stats.count,
      avgTime: stats.totalTime / stats.count,
      pctOfTotal: (stats.totalTime / totalTime) * 100,
    }))
    .sort((a, b) => b.totalTime - a.totalTime);

  // Identify bottlenecks
  /** @type {import('./perf-profiler.js').ProfileReport['bottlenecks']} */
  const bottlenecks = [];

  // Check for excessive sync operations
  if (syncEntries.length > entries.length * 0.1) {
    bottlenecks.push({
      name: 'Excessive GPU Syncs',
      impact: syncTime / totalTime,
      suggestion: 'Use CommandRecorder to batch operations and reduce syncs',
    });
  }

  // Check for memory-bound operations
  if (memoryTime > kernelTime) {
    bottlenecks.push({
      name: 'Memory Bandwidth Bound',
      impact: memoryTime / totalTime,
      suggestion: 'Consider kernel fusion to reduce memory traffic',
    });
  }

  // Check for small kernels (overhead-bound)
  const smallKernels = kernelEntries.filter(e => e.duration < 0.1);
  if (smallKernels.length > kernelEntries.length * 0.5) {
    const smallKernelTime = smallKernels.reduce((sum, e) => sum + e.duration, 0);
    bottlenecks.push({
      name: 'Kernel Launch Overhead',
      impact: smallKernelTime / totalTime,
      suggestion: 'Batch small kernels or increase work per kernel',
    });
  }

  // Check for dominant operations
  for (const item of breakdown.slice(0, 3)) {
    if (item.pctOfTotal > 30) {
      bottlenecks.push({
        name: `${item.name} dominates (${item.pctOfTotal.toFixed(1)}%)`,
        impact: item.pctOfTotal / 100,
        suggestion: `Optimize ${item.name} or check if it's using optimal variant`,
      });
    }
  }

  return {
    entries,
    summary: {
      totalTime,
      kernelTime,
      memoryTime,
      syncTime,
      otherTime,
      kernelCount: kernelEntries.length,
      memoryOps: memoryEntries.length,
      syncOps: syncEntries.length,
    },
    breakdown,
    bottlenecks,
  };
}

/**
 * Print profile report via debug log
 * @param {import('./perf-profiler.js').ProfileReport} [report]
 * @returns {void}
 */
export function printProfileReport(report) {
  const r = report || getProfileReport();

  log.info('Profile', '='.repeat(60));
  log.info('Profile', 'PERFORMANCE PROFILE REPORT');
  log.info('Profile', '='.repeat(60));

  log.info('Profile', 'Summary:');
  log.info('Profile', `  Total Time: ${r.summary.totalTime.toFixed(2)}ms`);
  log.info('Profile', `  Kernel Time: ${r.summary.kernelTime.toFixed(2)}ms (${((r.summary.kernelTime / r.summary.totalTime) * 100).toFixed(1)}%)`);
  log.info('Profile', `  Memory Time: ${r.summary.memoryTime.toFixed(2)}ms (${((r.summary.memoryTime / r.summary.totalTime) * 100).toFixed(1)}%)`);
  log.info('Profile', `  Sync Time: ${r.summary.syncTime.toFixed(2)}ms (${((r.summary.syncTime / r.summary.totalTime) * 100).toFixed(1)}%)`);
  log.info('Profile', `  Kernel Count: ${r.summary.kernelCount}`);

  log.info('Profile', 'Top Operations:');
  log.info('Profile', '-'.repeat(60));
  log.info('Profile', 'Operation                    | Time (ms) | Count | % Total');
  log.info('Profile', '-'.repeat(60));

  for (const item of r.breakdown.slice(0, 10)) {
    log.info('Profile',
      `${item.name.padEnd(28)} | ${item.totalTime.toFixed(2).padStart(9)} | ` +
      `${item.count.toString().padStart(5)} | ${item.pctOfTotal.toFixed(1).padStart(7)}%`
    );
  }

  if (r.bottlenecks.length > 0) {
    log.info('Profile', 'Bottlenecks:');
    log.info('Profile', '-'.repeat(60));
    for (const b of r.bottlenecks) {
      log.info('Profile', `  [${(b.impact * 100).toFixed(0)}%] ${b.name}`);
      log.info('Profile', `       Fix: ${b.suggestion}`);
    }
  }

  log.info('Profile', '='.repeat(60));
}

/**
 * Export profile data as JSON
 * @param {import('./perf-profiler.js').ProfileReport} [report]
 * @returns {string}
 */
export function exportProfileJSON(report) {
  return JSON.stringify(report || getProfileReport(), null, 2);
}

/**
 * Analyze decode performance and suggest optimizations
 * @param {number} tokensGenerated
 * @param {number} totalTimeMs
 * @param {number} [targetTokPerSec]
 * @returns {{ currentTokPerSec: number; targetTokPerSec: number; gap: number; suggestions: string[] }}
 */
export function analyzeDecodePerformance(
  tokensGenerated,
  totalTimeMs,
  targetTokPerSec = 40
) {
  const currentTokPerSec = (tokensGenerated / totalTimeMs) * 1000;
  const gap = targetTokPerSec / currentTokPerSec;

  /** @type {string[]} */
  const suggestions = [];

  if (gap > 5) {
    suggestions.push('Critical: Enable CommandRecorder for batched execution');
    suggestions.push('Critical: Verify GEMV kernels are being used for M=1 matmuls');
    suggestions.push('Critical: Check if subgroups are available and enabled');
  }

  if (gap > 3) {
    suggestions.push('Use fused FFN kernel to reduce memory bandwidth');
    suggestions.push('Enable optimized decode attention kernel');
    suggestions.push('Profile individual kernels to find dominant operation');
  }

  if (gap > 1.5) {
    suggestions.push('Consider F16 KV cache to reduce memory traffic');
    suggestions.push('Tune workgroup sizes for your GPU');
    suggestions.push('Check for unnecessary GPU syncs');
  }

  return {
    currentTokPerSec,
    targetTokPerSec,
    gap,
    suggestions,
  };
}
