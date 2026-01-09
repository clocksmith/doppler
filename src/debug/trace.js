/**
 * DOPPLER Debug Module - Trace Logging Interface
 *
 * Category-based tracing for detailed subsystem debugging.
 *
 * @module debug/trace
 */

import {
  enabledTraceCategories,
  traceLayerFilter,
  traceDecodeStep,
  traceMaxDecodeSteps,
  logHistory,
  getLogHistoryLimit,
} from './config.js';

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Check if a trace category is enabled.
 */
function isEnabled(category, layerIdx) {
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
function formatTraceMessage(category, message, layerIdx) {
  const timestamp = performance.now().toFixed(1);
  const layerTag = layerIdx !== undefined ? `L${layerIdx}:` : '';
  return `[${timestamp}ms][TRACE:${category}] ${layerTag}${message}`;
}

/**
 * Store trace log in history.
 */
function storeTrace(category, module, message, data) {
  logHistory.push({
    time: Date.now(),
    perfTime: performance.now(),
    level: `TRACE:${category}`,
    module,
    message,
    data,
  });

  const maxHistory = getLogHistoryLimit();
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
  loader(message, data) {
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
  kernels(message, data) {
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
  logits(message, data) {
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
  embed(message, data) {
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
  attn(layerIdx, message, data) {
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
  ffn(layerIdx, message, data) {
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
  kv(layerIdx, message, data) {
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
  sample(message, data) {
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
  buffers(message, data) {
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
  perf(message, data) {
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
