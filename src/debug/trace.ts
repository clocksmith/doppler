/**
 * DOPPLER Debug Module - Trace Logging Interface
 *
 * Category-based tracing for detailed subsystem debugging.
 *
 * @module debug/trace
 */

import { getRuntimeConfig } from '../config/runtime.js';
import {
  type TraceCategory,
  enabledTraceCategories,
  traceLayerFilter,
  traceDecodeStep,
  traceMaxDecodeSteps,
  logHistory,
} from './config.js';

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Check if a trace category is enabled.
 */
function isEnabled(category: TraceCategory, layerIdx?: number): boolean {
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
function formatTraceMessage(category: TraceCategory, message: string, layerIdx?: number): string {
  const timestamp = performance.now().toFixed(1);
  const layerTag = layerIdx !== undefined ? `L${layerIdx}:` : '';
  return `[${timestamp}ms][TRACE:${category}] ${layerTag}${message}`;
}

/**
 * Store trace log in history.
 */
function storeTrace(category: TraceCategory, module: string, message: string, data?: unknown): void {
  logHistory.push({
    time: Date.now(),
    perfTime: performance.now(),
    level: `TRACE:${category}`,
    module,
    message,
    data,
  });

  const maxHistory = getRuntimeConfig().debug.logHistory.maxLogHistoryEntries;
  if (logHistory.length > maxHistory) {
    logHistory.shift();
  }
}

// ============================================================================
// Trace Interface
// ============================================================================

/**
 * Trace logging interface - only logs if category is enabled.
 */
export const trace = {
  /**
   * Trace model loading operations.
   */
  loader(message: string, data?: unknown): void {
    if (!isEnabled('loader')) return;
    const formatted = formatTraceMessage('loader', message);
    storeTrace('loader', 'Loader', message, data);
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
    if (!isEnabled('kernels')) return;
    const formatted = formatTraceMessage('kernels', message);
    storeTrace('kernels', 'Kernels', message, data);
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
    if (!isEnabled('logits')) return;
    const formatted = formatTraceMessage('logits', message);
    storeTrace('logits', 'Logits', message, data);
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
    if (!isEnabled('embed')) return;
    const formatted = formatTraceMessage('embed', message);
    storeTrace('embed', 'Embed', message, data);
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
    if (!isEnabled('attn', layerIdx)) return;
    const formatted = formatTraceMessage('attn', message, layerIdx);
    storeTrace('attn', `Attn:L${layerIdx}`, message, data);
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
    if (!isEnabled('ffn', layerIdx)) return;
    const formatted = formatTraceMessage('ffn', message, layerIdx);
    storeTrace('ffn', `FFN:L${layerIdx}`, message, data);
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
    if (!isEnabled('kv', layerIdx)) return;
    const formatted = formatTraceMessage('kv', message, layerIdx);
    storeTrace('kv', `KV:L${layerIdx}`, message, data);
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
    if (!isEnabled('sample')) return;
    const formatted = formatTraceMessage('sample', message);
    storeTrace('sample', 'Sample', message, data);
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
    if (!isEnabled('buffers')) return;
    const formatted = formatTraceMessage('buffers', message);
    storeTrace('buffers', 'Buffers', message, data);
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
    if (!isEnabled('perf')) return;
    const formatted = formatTraceMessage('perf', message);
    storeTrace('perf', 'Perf', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },
};
