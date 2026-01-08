/**
 * Kernel Pipeline Tracer - systematic debugging for GPU inference.
 *
 * The inference pipeline is a deterministic sequence of kernels:
 * embed -> [layer0: norm -> qkv -> rope -> attn -> ffn] -> ... -> final_norm -> lm_head -> sample
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
 *   log.info('KernelTrace', kernelTrace.getTimeline());
 *
 * @module inference/pipeline/kernel-trace
 */

import { readBuffer } from '../../gpu/buffer-pool.js';
import { log, trace } from '../../debug/index.js';

// ============================================================================
// Tensor Snapshot Utility
// ============================================================================

/**
 * Create a snapshot of a GPU tensor (requires readback - expensive!).
 * @param {GPUBuffer} buffer
 * @param {number[]} [shape]
 * @param {string} [dtype]
 * @returns {Promise<import('./kernel-trace.js').TensorSnapshot>}
 */
export async function snapshotTensor(buffer, shape, dtype = 'f32') {
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
 * @param {Float32Array} arr
 * @param {number[]} shape
 * @param {string} [dtype]
 * @returns {import('./kernel-trace.js').TensorSnapshot}
 */
export function snapshotFromArray(arr, shape, dtype = 'f32') {
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
  constructor() {
    /** @type {import('./kernel-trace.js').KernelStep[]} */
    this._steps = [];
    /** @type {boolean} */
    this._enabled = false;
    /** @type {import('./kernel-trace.js').TraceOptions} */
    this._options = {};
    /** @type {import('./kernel-trace.js').Anomaly[]} */
    this._anomalies = [];
  }

  /**
   * Check if tracing is enabled.
   * @returns {boolean}
   */
  get enabled() {
    return this._enabled;
  }

  /**
   * Enable tracing with options.
   * @param {import('./kernel-trace.js').TraceOptions} [options]
   */
  enable(options = {}) {
    this._enabled = true;
    this._options = {
      layers: options.layers ?? [],
      breakOnAnomaly: options.breakOnAnomaly ?? false,
      explosionThreshold: options.explosionThreshold ?? 10,
      collapseThreshold: options.collapseThreshold ?? 1e-6,
      maxSteps: options.maxSteps ?? 5000,
    };
    this._steps = [];
    this._anomalies = [];
    log.info('Trace', 'Kernel tracing enabled', this._options);
  }

  /**
   * Disable tracing and clear data.
   */
  disable() {
    this._enabled = false;
    this._steps = [];
    this._anomalies = [];
  }

  /**
   * Clear recorded steps without disabling.
   */
  clear() {
    this._steps = [];
    this._anomalies = [];
  }

  /**
   * Check if a layer should be traced.
   * @param {number} layerIdx
   * @returns {boolean}
   */
  shouldTraceLayer(layerIdx) {
    if (!this._enabled) return false;
    if (!this._options.layers?.length) return true;
    return this._options.layers.includes(layerIdx);
  }

  /**
   * Record a kernel step.
   * @param {import('./kernel-trace.js').KernelStep} step
   */
  recordStep(step) {
    if (!this._enabled) return;

    // Check layer filter
    if (step.layer >= 0 && !this.shouldTraceLayer(step.layer)) return;

    // Add to steps (circular buffer)
    if (this._steps.length >= (this._options.maxSteps ?? 5000)) {
      this._steps.shift();
    }
    this._steps.push(step);

    // Check for anomalies
    const anomaly = this._detectAnomaly(step, this._steps.length - 1);
    if (anomaly) {
      this._anomalies.push(anomaly);
      this._logAnomaly(anomaly);

      if (this._options.breakOnAnomaly) {
        throw new Error(`[TRACE] Anomaly detected: ${anomaly.message}`);
      }
    }
  }

  /**
   * Detect anomaly in a step's output.
   * @param {import('./kernel-trace.js').KernelStep} step
   * @param {number} stepIdx
   * @returns {import('./kernel-trace.js').Anomaly | null}
   */
  _detectAnomaly(step, stepIdx) {
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
      const prevStep = this._steps[stepIdx - 1];
      const prevMaxAbs = prevStep.output.stats.maxAbs;
      const currMaxAbs = output.stats.maxAbs;
      const threshold = this._options.explosionThreshold ?? 10;

      if (prevMaxAbs > 0 && currMaxAbs > prevMaxAbs * threshold) {
        return {
          type: 'explosion',
          severity: 'warning',
          stepIdx,
          step,
          message: `Value explosion at ${step.label}: ${prevMaxAbs.toFixed(2)} -> ${currMaxAbs.toFixed(2)} (${(currMaxAbs / prevMaxAbs).toFixed(1)}x)`,
          factor: currMaxAbs / prevMaxAbs,
        };
      }
    }

    // Warning: Collapse to zeros
    const collapseThreshold = this._options.collapseThreshold ?? 1e-6;
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
   * Log an anomaly via the debug module with visual markers.
   * @param {import('./kernel-trace.js').Anomaly} anomaly
   */
  _logAnomaly(anomaly) {
    const marker = anomaly.severity === 'critical' ? '[CRITICAL]' : '[WARNING]';
    const logFn = anomaly.severity === 'critical' ? log.error : log.warn;
    logFn('Trace', `${marker} ${anomaly.message}`);
    trace.kernels(`Step: ${anomaly.step.name} (${anomaly.step.label})`);
    trace.kernels(`Output: shape=${JSON.stringify(anomaly.step.output.shape)}`);
    trace.kernels(`Stats: min=${anomaly.step.output.stats.min.toFixed(4)}, max=${anomaly.step.output.stats.max.toFixed(4)}, maxAbs=${anomaly.step.output.stats.maxAbs.toFixed(4)}`);
    trace.kernels(`Sample: [${anomaly.step.output.sample.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`);
  }

  /**
   * Find the first anomaly in the recorded steps.
   * @returns {import('./kernel-trace.js').Anomaly | null}
   */
  findAnomaly() {
    return this._anomalies.length > 0 ? this._anomalies[0] : null;
  }

  /**
   * Get all anomalies.
   * @returns {import('./kernel-trace.js').Anomaly[]}
   */
  getAnomalies() {
    return [...this._anomalies];
  }

  /**
   * Get the last recorded step.
   * @returns {import('./kernel-trace.js').KernelStep | null}
   */
  lastStep() {
    return this._steps.length > 0 ? this._steps[this._steps.length - 1] : null;
  }

  /**
   * Get the last N steps.
   * @param {number} n
   * @returns {import('./kernel-trace.js').KernelStep[]}
   */
  getLastNSteps(n) {
    return this._steps.slice(-n);
  }

  /**
   * Get all recorded steps.
   * @returns {import('./kernel-trace.js').KernelStep[]}
   */
  getSteps() {
    return [...this._steps];
  }

  /**
   * Format pipeline timeline as string.
   * @returns {string}
   */
  getTimeline() {
    if (this._steps.length === 0) return '[TRACE] No steps recorded';

    /** @type {string[]} */
    const lines = [];
    lines.push('+-----------------------------------------------------------------+');
    lines.push('| KERNEL PIPELINE TRACE                                           |');
    lines.push('+-----------------------------------------------------------------+');

    for (let i = 0; i < this._steps.length; i++) {
      const step = this._steps[i];
      const anomaly = this._anomalies.find(a => a.stepIdx === i);

      // Format: name.label [shape] min=X max=Y
      const shapeStr = `[${step.output.shape.join(',')}]`;
      const statsStr = `min=${step.output.stats.min.toFixed(2)} max=${step.output.stats.max.toFixed(2)}`;
      const labelPadded = step.label.padEnd(20).slice(0, 20);
      const shapePadded = shapeStr.padEnd(15).slice(0, 15);

      let line = `| ${labelPadded} ${shapePadded} ${statsStr}`;

      // Add anomaly marker
      if (anomaly) {
        const marker = anomaly.severity === 'critical' ? '[!]' : '[*]';
        line += ` ${marker} ${anomaly.type.toUpperCase()}`;
      }

      // Pad to box width
      line = line.padEnd(66) + '|';
      lines.push(line);
    }

    lines.push('+-----------------------------------------------------------------+');

    // Anomaly summary
    if (this._anomalies.length > 0) {
      lines.push('');
      const firstAnomaly = this._anomalies[0];
      const marker = firstAnomaly.severity === 'critical' ? '[CRITICAL]' : '[WARNING]';
      lines.push(`${marker} ANOMALY DETECTED at ${firstAnomaly.step.label}`);
      lines.push(`   ${firstAnomaly.message}`);
    }

    return lines.join('\n');
  }

  /**
   * Export trace as JSON for comparison/analysis.
   * @returns {string}
   */
  toJSON() {
    return JSON.stringify({
      steps: this._steps,
      anomalies: this._anomalies,
      options: this._options,
    }, null, 2);
  }

  /**
   * Dump the last N steps with full details via debug log/trace.
   * @param {number} [n]
   */
  dumpLastNSteps(n = 5) {
    const steps = this.getLastNSteps(n);
    log.info('Trace', `Last ${steps.length} steps:`);
    for (const step of steps) {
      trace.kernels(`  ${step.label}:`);
      trace.kernels(`    name: ${step.name}, variant: ${step.variant ?? 'default'}`);
      trace.kernels(`    output: ${JSON.stringify(step.output.shape)}, maxAbs=${step.output.stats.maxAbs.toFixed(4)}`);
      trace.kernels(`    sample: [${step.output.sample.slice(0, 8).map(v => v.toFixed(4)).join(', ')}]`);
      if (step.output.hasNaN) log.warn('Trace', `    [!] HAS NaN`);
      if (step.output.hasInf) log.warn('Trace', `    [!] HAS Inf`);
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
 * @param {string} name
 * @param {string} label
 * @param {number} layer
 * @param {GPUBuffer} outputBuffer
 * @param {number[]} outputShape
 * @param {{ inputs?: GPUBuffer[]; inputShapes?: number[][]; variant?: string; timeMs?: number }} [options]
 * @returns {Promise<void>}
 */
export async function traceStep(name, label, layer, outputBuffer, outputShape, options) {
  if (!kernelTrace.enabled) return;
  if (layer >= 0 && !kernelTrace.shouldTraceLayer(layer)) return;

  const output = await snapshotTensor(outputBuffer, outputShape);

  // Snapshot inputs if provided (expensive - only do if tracing)
  /** @type {import('./kernel-trace.js').TensorSnapshot[]} */
  const inputs = [];
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
 * @param {string} name
 * @param {string} label
 * @param {number} layer
 * @param {Float32Array} outputData
 * @param {number[]} outputShape
 * @param {{ variant?: string; timeMs?: number }} [options]
 */
export function traceStepSync(name, label, layer, outputData, outputShape, options) {
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
