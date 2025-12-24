/**
 * Performance Guards - Runtime Flags for Expensive Operations
 *
 * Controls performance-critical operations that should be
 * disabled in production or gated behind debug mode.
 */

/**
 * Performance configuration
 */
export interface PerfConfig {
  /** Allow GPU→CPU readbacks (mapAsync). Disable for production. */
  allowGPUReadback: boolean;

  /** Count queue.submit() calls per inference pass */
  trackSubmitCount: boolean;

  /** Count buffer allocations per inference pass */
  trackAllocations: boolean;

  /** Log expensive operations to console */
  logExpensiveOps: boolean;

  /** Throw error on disallowed operations (vs silent no-op) */
  strictMode: boolean;
}

/**
 * Default configuration
 * - Development: All tracking enabled, readbacks allowed
 * - Production: Tracking disabled, readbacks blocked
 */
const DEFAULT_CONFIG: PerfConfig = {
  allowGPUReadback: true, // Default to allowed for backward compatibility
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
};

/**
 * Global performance configuration
 */
let config: PerfConfig = { ...DEFAULT_CONFIG };

/**
 * Performance counters for current inference pass
 */
interface PerfCounters {
  submits: number;
  allocations: number;
  readbacks: number;
  startTime: number;
}

let counters: PerfCounters = {
  submits: 0,
  allocations: 0,
  readbacks: 0,
  startTime: 0,
};

/**
 * Configure performance guards
 */
export function configurePerfGuards(newConfig: Partial<PerfConfig>): void {
  config = { ...config, ...newConfig };
}

/**
 * Get current performance configuration
 */
export function getPerfConfig(): Readonly<PerfConfig> {
  return config;
}

/**
 * Reset performance counters (call at start of inference pass)
 */
export function resetPerfCounters(): void {
  counters = {
    submits: 0,
    allocations: 0,
    readbacks: 0,
    startTime: performance.now(),
  };
}

/**
 * Get current performance counters
 */
export function getPerfCounters(): Readonly<PerfCounters> {
  return counters;
}

/**
 * Increment submit counter
 */
export function trackSubmit(): void {
  if (config.trackSubmitCount) {
    counters.submits++;
    if (config.logExpensiveOps) {
      console.log(`[PerfGuard] Submit #${counters.submits}`);
    }
  }
}

/**
 * Increment allocation counter
 */
export function trackAllocation(size: number, label?: string): void {
  if (config.trackAllocations) {
    counters.allocations++;
    if (config.logExpensiveOps) {
      console.log(`[PerfGuard] Allocation #${counters.allocations}: ${size} bytes (${label || 'unlabeled'})`);
    }
  }
}

/**
 * Check if GPU readback is allowed
 * @throws Error if readback is disallowed and strictMode is enabled
 */
export function allowReadback(reason?: string): boolean {
  if (!config.allowGPUReadback) {
    const message = `[PerfGuard] GPU readback blocked: ${reason || 'unknown reason'}`;
    if (config.strictMode) {
      throw new Error(message);
    }
    if (config.logExpensiveOps) {
      console.warn(message);
    }
    return false;
  }

  if (config.trackSubmitCount) {
    counters.readbacks++;
    if (config.logExpensiveOps) {
      console.log(`[PerfGuard] Readback #${counters.readbacks}: ${reason || 'unknown'}`);
    }
  }

  return true;
}

/**
 * Get performance summary for current pass
 */
export function getPerfSummary(): string {
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
 */
export function logPerfSummary(): void {
  console.log(getPerfSummary());
}

/**
 * Production preset: Disable all tracking, block readbacks
 */
export function enableProductionMode(): void {
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
 */
export function enableDebugMode(): void {
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
 */
export function enableBenchmarkMode(): void {
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: true,
    trackAllocations: true,
    logExpensiveOps: false,
    strictMode: false,
  });
}
