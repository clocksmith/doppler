/**
 * Performance Guards - Runtime Flags for Expensive Operations
 *
 * Controls performance-critical operations that should be
 * disabled in production or gated behind debug mode.
 */

import { log, trace } from '../debug/index.js';

/**
 * Default configuration
 * - Development: All tracking enabled, readbacks allowed
 * - Production: Tracking disabled, readbacks blocked
 * @type {import('./perf-guards.js').PerfConfig}
 */
const DEFAULT_CONFIG = {
  allowGPUReadback: true, // Default to allowed for backward compatibility
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
};

/**
 * Global performance configuration
 * @type {import('./perf-guards.js').PerfConfig}
 */
let config = { ...DEFAULT_CONFIG };

/**
 * Performance counters for current inference pass
 * @type {{submits: number, allocations: number, readbacks: number, startTime: number}}
 */
let counters = {
  submits: 0,
  allocations: 0,
  readbacks: 0,
  startTime: 0,
};

/**
 * Configure performance guards
 * @param {Partial<import('./perf-guards.js').PerfConfig>} newConfig
 * @returns {void}
 */
export function configurePerfGuards(newConfig) {
  config = { ...config, ...newConfig };
}

/**
 * Get current performance configuration
 * @returns {Readonly<import('./perf-guards.js').PerfConfig>}
 */
export function getPerfConfig() {
  return config;
}

/**
 * Reset performance counters (call at start of inference pass)
 * @returns {void}
 */
export function resetPerfCounters() {
  counters = {
    submits: 0,
    allocations: 0,
    readbacks: 0,
    startTime: performance.now(),
  };
}

/**
 * Get current performance counters
 * @returns {Readonly<{submits: number, allocations: number, readbacks: number, startTime: number}>}
 */
export function getPerfCounters() {
  return counters;
}

/**
 * Increment submit counter
 * @returns {void}
 */
export function trackSubmit() {
  if (config.trackSubmitCount) {
    counters.submits++;
    if (config.logExpensiveOps) {
      trace.perf(`PerfGuard: Submit #${counters.submits}`);
    }
  }
}

/**
 * Increment allocation counter
 * @param {number} size
 * @param {string} [label]
 * @returns {void}
 */
export function trackAllocation(size, label) {
  if (config.trackAllocations) {
    counters.allocations++;
    if (config.logExpensiveOps) {
      trace.buffers(`PerfGuard: Allocation #${counters.allocations}: ${size} bytes (${label || 'unlabeled'})`);
    }
  }
}

/**
 * Check if GPU readback is allowed
 * @param {string} [reason]
 * @returns {boolean}
 * @throws Error if readback is disallowed and strictMode is enabled
 */
export function allowReadback(reason) {
  if (!config.allowGPUReadback) {
    const message = `PerfGuard: GPU readback blocked: ${reason || 'unknown reason'}`;
    if (config.strictMode) {
      throw new Error(message);
    }
    if (config.logExpensiveOps) {
      log.warn('PerfGuard', message);
    }
    return false;
  }

  if (config.trackSubmitCount) {
    counters.readbacks++;
    if (config.logExpensiveOps) {
      trace.perf(`PerfGuard: Readback #${counters.readbacks}: ${reason || 'unknown'}`);
    }
  }

  return true;
}

/**
 * Get performance summary for current pass
 * @returns {string}
 */
export function getPerfSummary() {
  const elapsed = performance.now() - counters.startTime;
  return [
    `Performance Summary (${elapsed.toFixed(1)}ms):`,
    `  Submits: ${counters.submits}`,
    `  Allocations: ${counters.allocations}`,
    `  Readbacks: ${counters.readbacks}`,
  ].join('\n');
}

/**
 * Log performance summary to console
 * @returns {void}
 */
export function logPerfSummary() {
  trace.perf(getPerfSummary());
}

/**
 * Production preset: Disable all tracking, block readbacks
 * @returns {void}
 */
export function enableProductionMode() {
  configurePerfGuards({
    allowGPUReadback: false,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: true,
  });
}

/**
 * Debug preset: Enable all tracking, allow readbacks, log operations
 * @returns {void}
 */
export function enableDebugMode() {
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: true,
    trackAllocations: true,
    logExpensiveOps: true,
    strictMode: false,
  });
}

/**
 * Benchmark preset: Track counters but don't log
 * @returns {void}
 */
export function enableBenchmarkMode() {
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: true,
    trackAllocations: true,
    logExpensiveOps: false,
    strictMode: false,
  });
}
