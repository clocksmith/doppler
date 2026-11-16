import { cloneGraph, deriveKernelEntry } from './dtype.js';

const PREFILL_ATTENTION_PAIRS = new Map([
  ['attention_streaming_f16kv.wgsl', 'attention_small_f16kv.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_streaming_f16kv.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_small_f16.wgsl'],
  ['attention_small_f16.wgsl', 'attention_streaming_f16.wgsl'],
]);

/** @param {Parameters<typeof import('../execution-graph-transforms.js').swapPrefillAttention>[0]} graph
 * @param {string} kernelKey */
function graphUsesKernelKeyInPrefill(graph, kernelKey) {
  for (const entry of graph.prefill || []) {
    if (Array.isArray(entry)) {
      if (entry[1] === kernelKey) {
        return true;
      }
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.steps)) {
      continue;
    }
    for (const step of entry.steps) {
      if (Array.isArray(step) && step[1] === kernelKey) {
        return true;
      }
    }
  }
  return false;
}

/*
 * Swap prefill attention kernel between streaming and small variants.
 *
 * The `opts` parameter specifies the direction:
 *   { from: 'attention_streaming_f16kv.wgsl', to: 'attention_small_f16kv.wgsl' }
 *
 * If `from`/`to` are not provided, uses the bidirectional pair map.
 * Returns null if no matching prefill attention kernel is found.
 *
 */
/** @type {typeof import('../execution-graph-transforms.js').swapPrefillAttention} */
export function swapPrefillAttention(graph, ctx, opts) {
  const from = opts?.from;
  const to = opts?.to;

  const result = cloneGraph(graph);
  let changed = false;

  for (const [key, entry] of Object.entries(result.kernels)) {
    let target;

    if (from && to) {
      // Explicit direction: only swap if the kernel matches `from`
      if (entry.kernel === from) {
        target = to;
      }
    } else {
      // Bidirectional: use pair map
      target = PREFILL_ATTENTION_PAIRS.get(entry.kernel);
    }

    if (target !== undefined) {
      const usedInPrefill = graphUsesKernelKeyInPrefill(graph, key);
      if (usedInPrefill) {
        result.kernels[key] = deriveKernelEntry(entry, target, entry.entry);
        changed = true;
      }
    }
  }

  return changed ? result : null;
}

// =============================================================================
// Transform: useHead256PrefillAttention
// =============================================================================

/*
 * Promote small-tile prefill attention onto the fixed 256-dim shared-block kernel.
 *
 */
/** @type {typeof import('../execution-graph-transforms.js').useHead256SmallPrefillAttention} */
export function useHead256SmallPrefillAttention(graph, ctx) {
  return swapPrefillAttention(graph, ctx, {
    from: 'attention_small_f16kv.wgsl',
    to: 'attention_head256_f16kv.wgsl',
  });
}

/*
 * Promote prefill attention onto the fixed 256-dim shared-block kernel.
 *
 */
/** @type {typeof import('../execution-graph-transforms.js').useHead256PrefillAttention} */
export function useHead256PrefillAttention(graph, ctx) {
  let current = graph;
  let changed = false;

  const smallRemap = useHead256SmallPrefillAttention(current, ctx);
  if (smallRemap) {
    current = smallRemap;
    changed = true;
  }

  const streamingRemap = swapPrefillAttention(current, ctx, {
    from: 'attention_streaming_f16kv.wgsl',
    to: 'attention_head256_f16kv.wgsl',
  });
  if (streamingRemap) {
    current = streamingRemap;
    changed = true;
  }

  return changed ? current : null;
}
