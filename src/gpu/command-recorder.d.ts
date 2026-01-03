/**
 * CommandRecorder - Batched GPU Command Recording
 *
 * Enables recording multiple GPU operations into a single command buffer,
 * avoiding per-kernel submit overhead. Manages temporary buffers automatically.
 */

/** Statistics about recorded operations */
export interface RecorderStats {
  opCount: number;
  tempBufferCount: number;
  pooledBufferCount: number;
  submitted: boolean;
}

/** Options for CommandRecorder */
export interface RecorderOptions {
  /** Enable GPU timestamp profiling (requires 'timestamp-query' feature) */
  profile?: boolean;
}

/** Profiling result - maps kernel label to time in milliseconds */
export type ProfileTimings = Record<string, number>;

/**
 * CommandRecorder wraps a GPUCommandEncoder and manages temporary resources.
 */
export declare class CommandRecorder {
  readonly device: GPUDevice;
  readonly label: string;

  /**
   * @param device - GPU device (auto-detected if not provided)
   * @param label - Label for debugging
   * @param options - Recorder options (profiling, etc.)
   */
  constructor(device?: GPUDevice | null, label?: string, options?: RecorderOptions);

  /**
   * Check if profiling is enabled and available.
   */
  isProfilingEnabled(): boolean;

  /**
   * Create a temporary buffer that will be destroyed after submit.
   * Use for uniform buffers and other per-operation temporaries.
   *
   * @param size - Buffer size in bytes
   * @param usage - Buffer usage flags
   * @param label - Buffer label for debugging
   * @returns GPUBuffer
   */
  createTempBuffer(size: number, usage: GPUBufferUsageFlags, label?: string): GPUBuffer;

  /**
   * Create an indirect dispatch buffer initialized with workgroup counts.
   * Buffer usage includes STORAGE so GPU kernels can update counts.
   */
  createIndirectDispatchBuffer(
    workgroups?: [number, number, number] | Uint32Array,
    label?: string
  ): GPUBuffer;

  /**
   * Update an indirect dispatch buffer with new workgroup counts.
   */
  writeIndirectDispatchBuffer(
    buffer: GPUBuffer,
    workgroups: [number, number, number] | Uint32Array,
    offset?: number
  ): void;

  /**
   * Create a uniform buffer, write data, and track for cleanup.
   * Uses content-addressed caching for identical uniform data.
   *
   * @param data - Data to write
   * @param label - Buffer label
   * @returns GPUBuffer
   */
  createUniformBuffer(data: ArrayBuffer | ArrayBufferView, label?: string): GPUBuffer;

  /**
   * Begin a compute pass on the encoder.
   * When profiling is enabled, injects timestampWrites to measure GPU execution time.
   *
   * @param label - Pass label for debugging (used as key in profile results)
   * @returns GPUComputePassEncoder
   */
  beginComputePass(label?: string): GPUComputePassEncoder;

  /**
   * Get the raw encoder for advanced use cases.
   * @returns GPUCommandEncoder
   */
  getEncoder(): GPUCommandEncoder;

  /**
   * Track a buffer for cleanup after submit.
   * Pooled buffers are released back to the pool; non-pooled buffers are destroyed.
   *
   * @param buffer - Buffer to clean up after submit
   */
  trackTemporaryBuffer(buffer: GPUBuffer): void;

  /**
   * Submit all recorded commands and clean up temporary buffers.
   * After calling this, the recorder cannot be reused.
   */
  submit(): void;

  /**
   * Submit and wait for GPU to complete (useful for debugging/profiling).
   * Also flushes the uniform cache's pending destruction queue to clean up
   * any evicted buffers that were referenced by this command buffer.
   * @returns Promise that resolves when GPU work is done
   */
  submitAndWait(): Promise<void>;

  /**
   * Get statistics about recorded operations.
   * @returns Statistics object
   */
  getStats(): RecorderStats;

  /**
   * Get the submit completion latency in milliseconds (null if not resolved yet).
   */
  getSubmitLatencyMs(): number | null;

  /**
   * Abort recording without submitting (cleanup only).
   * Use if an error occurs during recording.
   */
  abort(): void;

  /**
   * Resolve profiling timestamps and return per-kernel timings.
   * Must be called after submit() and GPU work is done.
   *
   * Returns a map of kernel label to execution time in milliseconds.
   * Labels with multiple invocations are aggregated (e.g., 'matmul' across all layers).
   *
   * @returns Promise resolving to timing map, or null if profiling not enabled
   */
  resolveProfileTimings(): Promise<ProfileTimings | null>;

  /**
   * Get a formatted profiling report.
   * Must be called after resolveProfileTimings().
   *
   * @param timings - Timings from resolveProfileTimings()
   * @returns Formatted string report
   */
  static formatProfileReport(timings: ProfileTimings): string;
}

/**
 * Create a new CommandRecorder.
 * @param label - Label for debugging
 * @param options - Recorder options
 * @returns CommandRecorder instance
 */
export function createCommandRecorder(label?: string, options?: RecorderOptions): CommandRecorder;

/**
 * Create a profiling-enabled CommandRecorder.
 * Falls back to non-profiling if timestamp-query not available.
 *
 * @param label - Label for debugging
 * @returns CommandRecorder with profiling enabled
 */
export function createProfilingRecorder(label?: string): CommandRecorder;

export default CommandRecorder;
