/**
 * Kernel Pipeline Tracer - systematic debugging for GPU inference.
 *
 * @module inference/pipelines/text/kernel-trace
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Snapshot of a tensor's statistics (no full data, just stats).
 */
export interface TensorSnapshot {
  shape: number[];
  dtype: string;
  stats: {
    min: number;
    max: number;
    maxAbs: number;
    mean: number;
    std: number;
  };
  sample: number[];
  hasNaN: boolean;
  hasInf: boolean;
}

/**
 * A single step in the kernel pipeline.
 */
export interface KernelStep {
  name: string;
  label: string;
  layer: number;
  inputs: TensorSnapshot[];
  output: TensorSnapshot;
  variant?: string;
  timeMs?: number;
}

/**
 * Detected anomaly in the pipeline.
 */
export interface Anomaly {
  type: 'nan' | 'inf' | 'explosion' | 'collapse';
  severity: 'critical' | 'warning';
  stepIdx: number;
  step: KernelStep;
  message: string;
  factor?: number;
}

/**
 * Options for enabling tracing.
 */
export interface TraceOptions {
  layers?: number[];
  breakOnAnomaly?: boolean;
  explosionThreshold?: number;
  collapseThreshold?: number;
  maxSteps?: number;
}

// ============================================================================
// Tensor Snapshot Utility
// ============================================================================

/**
 * Create a snapshot of a GPU tensor (requires readback - expensive!).
 */
export function snapshotTensor(
  buffer: GPUBuffer,
  shape?: number[],
  dtype?: string
): Promise<TensorSnapshot>;

/**
 * Create a snapshot from a Float32Array (CPU data).
 */
export function snapshotFromArray(
  arr: Float32Array,
  shape: number[],
  dtype?: string
): TensorSnapshot;

// ============================================================================
// KernelTrace Class
// ============================================================================

/**
 * Global kernel pipeline tracer.
 */
declare class KernelTrace {
  get enabled(): boolean;
  enable(options?: TraceOptions): void;
  disable(): void;
  clear(): void;
  shouldTraceLayer(layerIdx: number): boolean;
  recordStep(step: KernelStep): void;
  findAnomaly(): Anomaly | null;
  getAnomalies(): Anomaly[];
  lastStep(): KernelStep | null;
  getLastNSteps(n: number): KernelStep[];
  getSteps(): KernelStep[];
  getTimeline(): string;
  toJSON(): string;
  dumpLastNSteps(n?: number): void;
}

/**
 * Global kernel trace instance.
 */
export const kernelTrace: KernelTrace;

// ============================================================================
// Convenience: Record Step Helper
// ============================================================================

/**
 * Helper to record a step if tracing is enabled.
 */
export function traceStep(
  name: string,
  label: string,
  layer: number,
  outputBuffer: GPUBuffer,
  outputShape: number[],
  options?: {
    inputs?: GPUBuffer[];
    inputShapes?: number[][];
    variant?: string;
    timeMs?: number;
  }
): Promise<void>;

/**
 * Sync version for when you already have the data as Float32Array.
 */
export function traceStepSync(
  name: string,
  label: string,
  layer: number,
  outputData: Float32Array,
  outputShape: number[],
  options?: {
    variant?: string;
    timeMs?: number;
  }
): void;
