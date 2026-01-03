/**
 * GPU Profiler - Timestamp-based Performance Profiling
 *
 * Provides GPU-side timing using WebGPU timestamp queries.
 * Falls back to CPU timing when timestamp queries unavailable.
 */

/**
 * Profiling result for a single label
 */
export interface ProfileResult {
  /** Average time in milliseconds */
  avg: number;
  /** Minimum time in milliseconds */
  min: number;
  /** Maximum time in milliseconds */
  max: number;
  /** Number of samples */
  count: number;
  /** Total time in milliseconds */
  total: number;
}

/**
 * GPU Profiler using timestamp queries
 */
export declare class GPUProfiler {
  /**
   * @param device - WebGPU device (uses global if not provided)
   */
  constructor(device?: GPUDevice | null);

  /**
   * Begin timing a labeled region.
   * Uses CPU timing; use writeTimestamp() inside passes for GPU timestamps.
   * @param label - Unique label for this measurement
   */
  begin(label: string): void;

  /**
   * End timing a labeled region
   * @param label - Label started with begin()
   */
  end(label: string): void;

  /**
   * Write timestamp to query set within a compute pass
   * Call this instead of begin/end when inside a pass
   * @param pass - Compute pass encoder
   * @param label - Label for this measurement
   * @param isEnd - true for end timestamp
   */
  writeTimestamp(pass: GPUComputePassEncoder, label: string, isEnd?: boolean): void;

  /**
   * Resolve pending timestamp queries and update results
   * Call this after command buffer submission
   */
  resolve(): Promise<void>;

  /**
   * Get profiling results
   */
  getResults(): Record<string, ProfileResult>;

  /**
   * Get result for a specific label
   * @param label - Label to get result for
   */
  getResult(label: string): ProfileResult | null;

  /**
   * Reset all profiling data
   */
  reset(): void;

  /**
   * Get formatted report string
   */
  getReport(): string;

  /**
   * Check if timestamp queries are available
   */
  isGPUTimingAvailable(): boolean;

  /**
   * Destroy profiler resources
   */
  destroy(): void;
}

/**
 * Get the global profiler instance
 */
export function getProfiler(): GPUProfiler;

/**
 * Create a new profiler instance
 * @param device - Optional GPU device
 */
export function createProfiler(device?: GPUDevice | null): GPUProfiler;

/**
 * Convenience function to time a single operation
 * @param label - Label for the operation
 * @param fn - Async function to time
 */
export function timeOperation<T>(
  label: string,
  fn: () => Promise<T>
): Promise<{ result: T; timeMs: number }>;

export default GPUProfiler;
