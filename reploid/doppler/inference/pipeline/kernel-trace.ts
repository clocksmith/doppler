/**
 * Kernel Pipeline Tracer - systematic debugging for GPU inference.
 *
 * The inference pipeline is a deterministic sequence of kernels:
 * embed → [layer0: norm → qkv → rope → attn → ffn] → ... → final_norm → lm_head → sample
 *
 * Each kernel takes inputs and produces outputs. When output is garbage, ONE of these
 * kernels has wrong output. This tracer helps find which one, fast.
 *
 * Usage:
 *   // Enable tracing
 *   kernelTrace.enable({ breakOnAnomaly: true });
 *
 *   // In kernel wrappers, call recordStep (automatic with --trace flag)
 *   kernelTrace.recordStep({ name: 'matmul', label: 'q_proj', ... });
 *
 *   // After inference, analyze
 *   const anomaly = kernelTrace.findAnomaly();
 *   console.log(kernelTrace.getTimeline());
 *
 * @module inference/pipeline/kernel-trace
 */

import { readBuffer } from '../../gpu/buffer-pool.js';

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
  sample: number[];  // First 8 values
  hasNaN: boolean;
  hasInf: boolean;
}

/**
 * A single step in the kernel pipeline.
 */
export interface KernelStep {
  name: string;           // "matmul", "rmsnorm", "attention", "rope"
  label: string;          // "layer0.q_proj", "layer0.input_norm"
  layer: number;          // -1 for non-layer ops (embed, final_norm)
  inputs: TensorSnapshot[];
  output: TensorSnapshot;
  variant?: string;       // "gemv_subgroup", "dequant_f16", etc.
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
  factor?: number;  // For explosion: how much values increased
}

/**
 * Options for enabling tracing.
 */
export interface TraceOptions {
  /** Only trace these layer indices (empty = all) */
  layers?: number[];
  /** Break (throw) on first anomaly detected */
  breakOnAnomaly?: boolean;
  /** Explosion threshold: warn if values increase by this factor */
  explosionThreshold?: number;
  /** Collapse threshold: warn if maxAbs falls below this */
  collapseThreshold?: number;
  /** Max steps to record (circular buffer) */
  maxSteps?: number;
}

// ============================================================================
// Tensor Snapshot Utility
// ============================================================================

/**
 * Create a snapshot of a GPU tensor (requires readback - expensive!).
 */
export async function snapshotTensor(
  buffer: GPUBuffer,
  shape?: number[],
  dtype: string = 'f32'
): Promise<TensorSnapshot> {
  try {
    const data = await readBuffer(buffer);
    const arr = new Float32Array(data);
    return snapshotFromArray(arr, shape ?? [arr.length], dtype);
  } catch (e) {
    // Return empty snapshot on error
    return {
      shape: shape ?? [0],
      dtype,
      stats: { min: 0, max: 0, maxAbs: 0, mean: 0, std: 0 },
      sample: [],
      hasNaN: false,
      hasInf: false,
    };
  }
}

/**
 * Create a snapshot from a Float32Array (CPU data).
 */
export function snapshotFromArray(
  arr: Float32Array,
  shape: number[],
  dtype: string = 'f32'
): TensorSnapshot {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let nanCount = 0;
  let infCount = 0;
  let validCount = 0;

  // Only iterate over valid elements based on shape, not full buffer (pool may have padding)
  const numElements = shape.reduce((a, b) => a * b, 1);
  const limit = Math.min(arr.length, numElements);
  for (let i = 0; i < limit; i++) {
    const v = arr[i];
    if (Number.isNaN(v)) {
      nanCount++;
    } else if (!Number.isFinite(v)) {
      infCount++;
    } else {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      sumSq += v * v;
      validCount++;
    }
  }

  const mean = validCount > 0 ? sum / validCount : 0;
  const variance = validCount > 0 ? (sumSq / validCount) - (mean * mean) : 0;
  const std = Math.sqrt(Math.max(0, variance));
  const maxAbs = Math.max(Math.abs(min), Math.abs(max));

  return {
    shape,
    dtype,
    stats: { min, max, maxAbs, mean, std },
    sample: Array.from(arr.slice(0, 8)),
    hasNaN: nanCount > 0,
    hasInf: infCount > 0,
  };
}

// ============================================================================
// KernelTrace Class
// ============================================================================

/**
 * Global kernel pipeline tracer.
 */
class KernelTrace {
  private steps: KernelStep[] = [];
  private _enabled: boolean = false;
  private options: TraceOptions = {};
  private anomalies: Anomaly[] = [];

  /**
   * Check if tracing is enabled.
   */
  get enabled(): boolean {
    return this._enabled;
  }

  /**
   * Enable tracing with options.
   */
  enable(options: TraceOptions = {}): void {
    this._enabled = true;
    this.options = {
      layers: options.layers ?? [],
      breakOnAnomaly: options.breakOnAnomaly ?? false,
      explosionThreshold: options.explosionThreshold ?? 10,
      collapseThreshold: options.collapseThreshold ?? 1e-6,
      maxSteps: options.maxSteps ?? 5000,
    };
    this.steps = [];
    this.anomalies = [];
    console.log('[TRACE] Kernel tracing enabled', this.options);
  }

  /**
   * Disable tracing and clear data.
   */
  disable(): void {
    this._enabled = false;
    this.steps = [];
    this.anomalies = [];
  }

  /**
   * Clear recorded steps without disabling.
   */
  clear(): void {
    this.steps = [];
    this.anomalies = [];
  }

  /**
   * Check if a layer should be traced.
   */
  shouldTraceLayer(layerIdx: number): boolean {
    if (!this._enabled) return false;
    if (!this.options.layers?.length) return true;
    return this.options.layers.includes(layerIdx);
  }

  /**
   * Record a kernel step.
   */
  recordStep(step: KernelStep): void {
    if (!this._enabled) return;

    // Check layer filter
    if (step.layer >= 0 && !this.shouldTraceLayer(step.layer)) return;

    // Add to steps (circular buffer)
    if (this.steps.length >= (this.options.maxSteps ?? 5000)) {
      this.steps.shift();
    }
    this.steps.push(step);

    // Check for anomalies
    const anomaly = this.detectAnomaly(step, this.steps.length - 1);
    if (anomaly) {
      this.anomalies.push(anomaly);
      this.logAnomaly(anomaly);

      if (this.options.breakOnAnomaly) {
        throw new Error(`[TRACE] Anomaly detected: ${anomaly.message}`);
      }
    }
  }

  /**
   * Detect anomaly in a step's output.
   */
  private detectAnomaly(step: KernelStep, stepIdx: number): Anomaly | null {
    const { output } = step;

    // Critical: NaN
    if (output.hasNaN) {
      return {
        type: 'nan',
        severity: 'critical',
        stepIdx,
        step,
        message: `NaN detected in ${step.label}`,
      };
    }

    // Critical: Inf
    if (output.hasInf) {
      return {
        type: 'inf',
        severity: 'critical',
        stepIdx,
        step,
        message: `Inf detected in ${step.label}`,
      };
    }

    // Warning: Value explosion (compare to previous step)
    if (stepIdx > 0) {
      const prevStep = this.steps[stepIdx - 1];
      const prevMaxAbs = prevStep.output.stats.maxAbs;
      const currMaxAbs = output.stats.maxAbs;
      const threshold = this.options.explosionThreshold ?? 10;

      if (prevMaxAbs > 0 && currMaxAbs > prevMaxAbs * threshold) {
        return {
          type: 'explosion',
          severity: 'warning',
          stepIdx,
          step,
          message: `Value explosion at ${step.label}: ${prevMaxAbs.toFixed(2)} → ${currMaxAbs.toFixed(2)} (${(currMaxAbs / prevMaxAbs).toFixed(1)}x)`,
          factor: currMaxAbs / prevMaxAbs,
        };
      }
    }

    // Warning: Collapse to zeros
    const collapseThreshold = this.options.collapseThreshold ?? 1e-6;
    if (output.stats.maxAbs < collapseThreshold && output.shape.reduce((a, b) => a * b, 1) > 0) {
      return {
        type: 'collapse',
        severity: 'warning',
        stepIdx,
        step,
        message: `Value collapse at ${step.label}: maxAbs=${output.stats.maxAbs.toExponential(2)}`,
      };
    }

    return null;
  }

  /**
   * Log an anomaly to console with visual markers.
   */
  private logAnomaly(anomaly: Anomaly): void {
    const marker = anomaly.severity === 'critical' ? '🚨' : '⚠️';
    console.log(`${marker} [TRACE] ${anomaly.message}`);
    console.log(`   Step: ${anomaly.step.name} (${anomaly.step.label})`);
    console.log(`   Output: shape=${JSON.stringify(anomaly.step.output.shape)}`);
    console.log(`   Stats: min=${anomaly.step.output.stats.min.toFixed(4)}, max=${anomaly.step.output.stats.max.toFixed(4)}, maxAbs=${anomaly.step.output.stats.maxAbs.toFixed(4)}`);
    console.log(`   Sample: [${anomaly.step.output.sample.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`);
  }

  /**
   * Find the first anomaly in the recorded steps.
   */
  findAnomaly(): Anomaly | null {
    return this.anomalies.length > 0 ? this.anomalies[0] : null;
  }

  /**
   * Get all anomalies.
   */
  getAnomalies(): Anomaly[] {
    return [...this.anomalies];
  }

  /**
   * Get the last recorded step.
   */
  lastStep(): KernelStep | null {
    return this.steps.length > 0 ? this.steps[this.steps.length - 1] : null;
  }

  /**
   * Get the last N steps.
   */
  getLastNSteps(n: number): KernelStep[] {
    return this.steps.slice(-n);
  }

  /**
   * Get all recorded steps.
   */
  getSteps(): KernelStep[] {
    return [...this.steps];
  }

  /**
   * Format pipeline timeline as string.
   */
  getTimeline(): string {
    if (this.steps.length === 0) return '[TRACE] No steps recorded';

    const lines: string[] = [];
    lines.push('┌─────────────────────────────────────────────────────────────────┐');
    lines.push('│ KERNEL PIPELINE TRACE                                           │');
    lines.push('├─────────────────────────────────────────────────────────────────┤');

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      const anomaly = this.anomalies.find(a => a.stepIdx === i);

      // Format: name.label [shape] min=X max=Y
      const shapeStr = `[${step.output.shape.join(',')}]`;
      const statsStr = `min=${step.output.stats.min.toFixed(2)} max=${step.output.stats.max.toFixed(2)}`;
      const labelPadded = step.label.padEnd(20).slice(0, 20);
      const shapePadded = shapeStr.padEnd(15).slice(0, 15);

      let line = `│ ${labelPadded} ${shapePadded} ${statsStr}`;

      // Add anomaly marker
      if (anomaly) {
        const marker = anomaly.severity === 'critical' ? '🚨' : '⚠️';
        line += ` ${marker} ${anomaly.type.toUpperCase()}`;
      }

      // Pad to box width
      line = line.padEnd(66) + '│';
      lines.push(line);
    }

    lines.push('└─────────────────────────────────────────────────────────────────┘');

    // Anomaly summary
    if (this.anomalies.length > 0) {
      lines.push('');
      const firstAnomaly = this.anomalies[0];
      const marker = firstAnomaly.severity === 'critical' ? '🚨' : '⚠️';
      lines.push(`${marker} ANOMALY DETECTED at ${firstAnomaly.step.label}`);
      lines.push(`   ${firstAnomaly.message}`);
    }

    return lines.join('\n');
  }

  /**
   * Export trace as JSON for comparison/analysis.
   */
  toJSON(): string {
    return JSON.stringify({
      steps: this.steps,
      anomalies: this.anomalies,
      options: this.options,
    }, null, 2);
  }

  /**
   * Dump the last N steps with full details to console.
   */
  dumpLastNSteps(n: number = 5): void {
    const steps = this.getLastNSteps(n);
    console.log(`[TRACE] Last ${steps.length} steps:`);
    for (const step of steps) {
      console.log(`  ${step.label}:`);
      console.log(`    name: ${step.name}, variant: ${step.variant ?? 'default'}`);
      console.log(`    output: ${JSON.stringify(step.output.shape)}, maxAbs=${step.output.stats.maxAbs.toFixed(4)}`);
      console.log(`    sample: [${step.output.sample.slice(0, 8).map(v => v.toFixed(4)).join(', ')}]`);
      if (step.output.hasNaN) console.log(`    ⚠️ HAS NaN`);
      if (step.output.hasInf) console.log(`    ⚠️ HAS Inf`);
    }
  }
}

// ============================================================================
// Global Singleton
// ============================================================================

/**
 * Global kernel trace instance.
 * Enable with: kernelTrace.enable({ breakOnAnomaly: true })
 */
export const kernelTrace = new KernelTrace();

// ============================================================================
// Convenience: Record Step Helper
// ============================================================================

/**
 * Helper to record a step if tracing is enabled.
 * Use this in kernel wrappers for minimal overhead when tracing is off.
 */
export async function traceStep(
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
): Promise<void> {
  if (!kernelTrace.enabled) return;
  if (layer >= 0 && !kernelTrace.shouldTraceLayer(layer)) return;

  const output = await snapshotTensor(outputBuffer, outputShape);

  // Snapshot inputs if provided (expensive - only do if tracing)
  const inputs: TensorSnapshot[] = [];
  if (options?.inputs && options?.inputShapes) {
    for (let i = 0; i < options.inputs.length; i++) {
      const snap = await snapshotTensor(options.inputs[i], options.inputShapes[i]);
      inputs.push(snap);
    }
  }

  kernelTrace.recordStep({
    name,
    label,
    layer,
    inputs,
    output,
    variant: options?.variant,
    timeMs: options?.timeMs,
  });
}

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
): void {
  if (!kernelTrace.enabled) return;
  if (layer >= 0 && !kernelTrace.shouldTraceLayer(layer)) return;

  const output = snapshotFromArray(outputData, outputShape);

  kernelTrace.recordStep({
    name,
    label,
    layer,
    inputs: [],
    output,
    variant: options?.variant,
    timeMs: options?.timeMs,
  });
}
