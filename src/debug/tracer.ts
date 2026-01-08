/**
 * Trace Logging Interface
 *
 * @module debug/tracer
 */

import {
  type TraceCategory,
  TRACE_CATEGORIES,
} from './types.js';
import {
  enabledTraceCategories,
  traceLayerFilter,
  traceDecodeStep,
  traceMaxDecodeSteps,
} from './state.js';
import { storeLog } from './logger.js';

/**
 * Check if a trace category is enabled.
 */
export function isTraceEnabled(category: TraceCategory, layerIdx?: number): boolean {
  if (!enabledTraceCategories.has(category)) return false;

  // Check layer filter
  if (layerIdx !== undefined && traceLayerFilter.length > 0) {
    if (!traceLayerFilter.includes(layerIdx)) return false;
  }

  // Check decode step limit
  if (traceMaxDecodeSteps > 0 && traceDecodeStep > traceMaxDecodeSteps) {
    return false;
  }

  return true;
}

/**
 * Format a trace message with category tag.
 */
export function formatTraceMessage(category: TraceCategory, message: string, layerIdx?: number): string {
  const timestamp = performance.now().toFixed(1);
  const layerTag = layerIdx !== undefined ? `L${layerIdx}:` : '';
  return `[${timestamp}ms][TRACE:${category}] ${layerTag}${message}`;
}

/**
 * Trace logging interface - only logs if category is enabled.
 */
export const trace = {
  /**
   * Trace model loading operations.
   */
  loader(message: string, data?: unknown): void {
    if (!isTraceEnabled('loader')) return;
    const formatted = formatTraceMessage('loader', message);
    storeLog('TRACE:loader', 'Loader', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace kernel execution.
   */
  kernels(message: string, data?: unknown): void {
    if (!isTraceEnabled('kernels')) return;
    const formatted = formatTraceMessage('kernels', message);
    storeLog('TRACE:kernels', 'Kernels', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace logit computation.
   */
  logits(message: string, data?: unknown): void {
    if (!isTraceEnabled('logits')) return;
    const formatted = formatTraceMessage('logits', message);
    storeLog('TRACE:logits', 'Logits', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace embedding layer.
   */
  embed(message: string, data?: unknown): void {
    if (!isTraceEnabled('embed')) return;
    const formatted = formatTraceMessage('embed', message);
    storeLog('TRACE:embed', 'Embed', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace attention computation.
   */
  attn(layerIdx: number, message: string, data?: unknown): void {
    if (!isTraceEnabled('attn', layerIdx)) return;
    const formatted = formatTraceMessage('attn', message, layerIdx);
    storeLog('TRACE:attn', `Attn:L${layerIdx}`, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace feed-forward network.
   */
  ffn(layerIdx: number, message: string, data?: unknown): void {
    if (!isTraceEnabled('ffn', layerIdx)) return;
    const formatted = formatTraceMessage('ffn', message, layerIdx);
    storeLog('TRACE:ffn', `FFN:L${layerIdx}`, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace KV cache operations.
   */
  kv(layerIdx: number, message: string, data?: unknown): void {
    if (!isTraceEnabled('kv', layerIdx)) return;
    const formatted = formatTraceMessage('kv', message, layerIdx);
    storeLog('TRACE:kv', `KV:L${layerIdx}`, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace token sampling.
   */
  sample(message: string, data?: unknown): void {
    if (!isTraceEnabled('sample')) return;
    const formatted = formatTraceMessage('sample', message);
    storeLog('TRACE:sample', 'Sample', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace buffer stats (expensive - requires GPU readback).
   */
  buffers(message: string, data?: unknown): void {
    if (!isTraceEnabled('buffers')) return;
    const formatted = formatTraceMessage('buffers', message);
    storeLog('TRACE:buffers', 'Buffers', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace performance timing.
   */
  perf(message: string, data?: unknown): void {
    if (!isTraceEnabled('perf')) return;
    const formatted = formatTraceMessage('perf', message);
    storeLog('TRACE:perf', 'Perf', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },
};
