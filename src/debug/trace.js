import {
  enabledTraceCategories,
  traceLayerFilter,
  traceDecodeStep,
  traceMaxDecodeSteps,
  logHistory,
  getLogHistoryLimit,
} from './config.js';

function isEnabled(category, layerIdx) {
  if (!enabledTraceCategories.has(category)) return false;

  if (layerIdx !== undefined && traceLayerFilter.length > 0) {
    if (!traceLayerFilter.includes(layerIdx)) return false;
  }

  if (traceMaxDecodeSteps > 0 && traceDecodeStep > traceMaxDecodeSteps) {
    return false;
  }

  return true;
}

function formatTraceMessage(category, message, layerIdx) {
  const timestamp = performance.now().toFixed(1);
  const layerTag = layerIdx !== undefined ? `L${layerIdx}:` : '';
  return `[${timestamp}ms][TRACE:${category}] ${layerTag}${message}`;
}

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

export const trace = {
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
