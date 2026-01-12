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
  /** Allow GPU->CPU readbacks (mapAsync). Disable for production. */
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
 * Performance counters for current inference pass
 */
interface PerfCounters {
  submits: number;
  allocations: number;
  readbacks: number;
  startTime: number;
}

/**
 * Configure performance guards
 */
export function configurePerfGuards(newConfig: Partial<PerfConfig>): void;

/**
 * Get current performance configuration
 */
export function getPerfConfig(): Readonly<PerfConfig>;

/**
 * Reset performance counters (call at start of inference pass)
 */
export function resetPerfCounters(): void;

/**
 * Get current performance counters
 */
export function getPerfCounters(): Readonly<PerfCounters>;

/**
 * Increment submit counter
 */
export function trackSubmit(): void;

/**
 * Increment allocation counter
 */
export function trackAllocation(size: number, label?: string): void;

/**
 * Check if GPU readback is allowed
 * @throws Error if readback is disallowed and strictMode is enabled
 */
export function allowReadback(reason?: string, count?: number): boolean;

/**
 * Get performance summary for current pass
 */
export function getPerfSummary(): string;

/**
 * Log performance summary to console
 */
export function logPerfSummary(): void;

/**
 * Production preset: Disable all tracking, block readbacks
 */
export function enableProductionMode(): void;

/**
 * Debug preset: Enable all tracking, allow readbacks, log operations
 */
export function enableDebugMode(): void;

/**
 * Benchmark preset: Track counters but don't log
 */
export function enableBenchmarkMode(): void;
