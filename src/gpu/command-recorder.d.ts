/**
 * CommandRecorder - Batched GPU Command Recording
 *
 * Enables recording multiple GPU operations into a single command buffer,
 * avoiding per-kernel submit overhead. Manages temporary buffers automatically.
 */

/** Options for CommandRecorder */
export interface RecorderOptions {
  /** Enable GPU timestamp profiling (requires 'timestamp-query' feature) */
  profile?: boolean;
  /** Collect per-label operation counts for diagnostics. Defaults to true. */
  recordLabels?: boolean;
  /** Collect per-dispatch geometry for evidence ledgers. Defaults to true. */
  recordDispatches?: boolean;
  /** Aggregate identical label/kind/geometry records in-place to reduce observer overhead. */
  aggregateDispatches?: boolean;
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
   */
  createUniformBuffer(data: ArrayBuffer | ArrayBufferView, label?: string): GPUBuffer;

  /**
   * Begin a compute pass on the encoder.
   * When profiling is enabled, injects timestampWrites to measure GPU execution time.
   *
   */
  beginComputePass(label?: string): GPUComputePassEncoder;

  /**
   * Close the active coalesced compute pass, if one is open.
   */
  closeActiveComputePass(): void;

  /**
   * Record a simple dispatch. Non-profiling recorders coalesce consecutive
   * dispatches into one compute pass until a raw encoder boundary.
   */
  recordDispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: [number, number, number],
    label?: string
  ): void;

  /**
   * Record an indirect dispatch with the same pass coalescing behavior.
   */
  recordDispatchIndirect(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    indirectBuffer: GPUBuffer,
    indirectOffset?: number,
    label?: string
  ): void;

  /**
   * Get the raw encoder for advanced use cases.
   */
  getEncoder(): GPUCommandEncoder;

  /**
   * Track a buffer for cleanup after submit.
   * Pooled buffers are released back to the pool; non-pooled buffers are destroyed.
   *
   */
  trackTemporaryBuffer(buffer: GPUBuffer): void;

  /**
   * Schedule async work that must run after GPU submission completes but before
   * recorder-owned cleanup makes the captured data unavailable.
   */
  enqueueCompletionTask(task: () => Promise<void> | void): void;

  /** Submit the recorded prefix and continue recording without readback or cleanup. */
  submitPartial(): void;

  /**
   * Submit all recorded commands and clean up temporary buffers.
   * After calling this, the recorder cannot be reused.
   */
  submit(options?: { cleanup?: 'queue' | 'deferred' }): void;

  /**
   * Finalize a deferred-cleanup submit after another wait condition
   * has already established GPU completion.
   */
  completeDeferredCleanup(options?: { discardPooled?: boolean }): Promise<void>;

  /**
   * Submit and wait for GPU to complete (useful for debugging/profiling).
   * Also flushes the uniform cache's pending destruction queue to clean up
   * any evicted buffers that were referenced by this command buffer.
   */
  submitAndWait(): Promise<void>;

  /**
   * Get statistics about recorded operations.
   */
  getStats(): {
    opCount: number;
    opLabelCounts: Record<string, number>;
    dispatches: Array<{
      label: string;
      kind: 'direct' | 'indirect';
      workgroups: number[] | null;
      count?: number;
    }>;
    computePassCount: number;
    tempBufferCount: number;
    pooledBufferCount: number;
    submitted: boolean;
    submissionCount: number;
  };

  /**
   * Final submit to completion in milliseconds (null until resolved).
   */
  getSubmitLatencyMs(): number | null;

  /**
   * Abort the unsubmitted suffix. Cleanup waits for any submitted prefix.
   */
  abort(): Promise<void> | null;

  /**
   * Resolve profiling timestamps and return per-kernel timings.
   * Must be called after submit() and GPU work is done.
   *
   * Returns a map of kernel label to execution time in milliseconds.
   * Labels with multiple invocations are aggregated (e.g., 'matmul' across all layers).
   *
   */
  resolveProfileTimings(): Promise<ProfileTimings | null>;

  /**
   * Get a formatted profiling report.
   * Must be called after resolveProfileTimings().
   *
   */
  static formatProfileReport(timings: ProfileTimings): string;
}

/**
 * Create a new CommandRecorder.
 */
export function createCommandRecorder(
  label?: string,
  options?: RecorderOptions,
  device?: GPUDevice | null
): CommandRecorder;

/**
 * Create a profiling-enabled CommandRecorder.
 * Falls back to non-profiling if timestamp-query not available.
 *
 */
export function createProfilingRecorder(
  label?: string,
  device?: GPUDevice | null,
  options?: Omit<RecorderOptions, 'profile'>
): CommandRecorder;

export default CommandRecorder;
