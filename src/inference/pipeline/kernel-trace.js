

import { log, trace } from '../../debug/index.js';
import { snapshotTensor as snapshotTensorImpl, snapshotFromArray as snapshotFromArrayImpl } from '../../debug/tensor.js';

// ============================================================================
// Tensor Snapshot Utility
// ============================================================================


export async function snapshotTensor(buffer, shape, dtype = 'f32') {
  return snapshotTensorImpl(buffer, shape, dtype);
}


export function snapshotFromArray(arr, shape, dtype = 'f32') {
  return snapshotFromArrayImpl(arr, shape, dtype);
}

// ============================================================================
// KernelTrace Class
// ============================================================================


class KernelTrace {
  constructor() {
    
    this._steps = [];
    
    this._enabled = false;
    
    this._options = {};
    
    this._anomalies = [];
  }

  
  get enabled() {
    return this._enabled;
  }

  
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

  
  disable() {
    this._enabled = false;
    this._steps = [];
    this._anomalies = [];
  }

  
  clear() {
    this._steps = [];
    this._anomalies = [];
  }

  
  shouldTraceLayer(layerIdx) {
    if (!this._enabled) return false;
    if (!this._options.layers?.length) return true;
    return this._options.layers.includes(layerIdx);
  }

  
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

  
  _logAnomaly(anomaly) {
    const marker = anomaly.severity === 'critical' ? '[CRITICAL]' : '[WARNING]';
    const logFn = anomaly.severity === 'critical' ? log.error : log.warn;
    logFn('Trace', `${marker} ${anomaly.message}`);
    trace.kernels(`Step: ${anomaly.step.name} (${anomaly.step.label})`);
    trace.kernels(`Output: shape=${JSON.stringify(anomaly.step.output.shape)}`);
    trace.kernels(`Stats: min=${anomaly.step.output.stats.min.toFixed(4)}, max=${anomaly.step.output.stats.max.toFixed(4)}, maxAbs=${anomaly.step.output.stats.maxAbs.toFixed(4)}`);
    trace.kernels(`Sample: [${anomaly.step.output.sample.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`);
  }

  
  findAnomaly() {
    return this._anomalies.length > 0 ? this._anomalies[0] : null;
  }

  
  getAnomalies() {
    return [...this._anomalies];
  }

  
  lastStep() {
    return this._steps.length > 0 ? this._steps[this._steps.length - 1] : null;
  }

  
  getLastNSteps(n) {
    return this._steps.slice(-n);
  }

  
  getSteps() {
    return [...this._steps];
  }

  
  getTimeline() {
    if (this._steps.length === 0) return '[TRACE] No steps recorded';

    
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

  
  toJSON() {
    return JSON.stringify({
      steps: this._steps,
      anomalies: this._anomalies,
      options: this._options,
    }, null, 2);
  }

  
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


export const kernelTrace = new KernelTrace();

// ============================================================================
// Convenience: Record Step Helper
// ============================================================================


export async function traceStep(name, label, layer, outputBuffer, outputShape, options) {
  if (!kernelTrace.enabled) return;
  if (layer >= 0 && !kernelTrace.shouldTraceLayer(layer)) return;

  const output = await snapshotTensor(outputBuffer, outputShape);

  // Snapshot inputs if provided (expensive - only do if tracing)
  
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
