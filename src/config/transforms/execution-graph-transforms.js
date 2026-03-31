// =============================================================================
// Execution Graph Transforms
// =============================================================================
//
// Pure functions that take an execution-v1 graph (as stamped in the manifest)
// and return a modified copy. Replaces the kernel path registry system.
//
// Each transform: (graph, ctx) => newGraph | null
// Returns null if not applicable (no-op).
// Must be pure — no mutation, return new objects.

// =============================================================================
// Helpers
// =============================================================================

/**
 * Deep-clone an execution graph.
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph}
 */
function cloneGraph(graph) {
  return structuredClone(graph);
}

/**
 * Shader files that require subgroups even though "subgroup" is not in the filename.
 * Online attention kernels use subgroup reductions internally.
 */
const SUBGROUP_REQUIRING_FILES = new Set([
  'attention_decode_online_f16kv.wgsl',
  'attention_decode_online_f16.wgsl',
]);

/**
 * Check whether a kernel entry requires subgroup support.
 * @param {{ kernel: string }} kernelEntry
 * @returns {boolean}
 */
function isSubgroupKernel(kernelEntry) {
  if (typeof kernelEntry.kernel !== 'string') return false;
  return kernelEntry.kernel.includes('subgroup') || SUBGROUP_REQUIRING_FILES.has(kernelEntry.kernel);
}

function requiresNoSubgroupFallback(kernelEntry) {
  if (typeof kernelEntry?.kernel !== 'string') return false;
  return isSubgroupKernel(kernelEntry) || kernelEntry.kernel.startsWith('fused_matmul_q4');
}

/**
 * Find all kernel keys in the graph whose `kernel` file matches the given filename.
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {string} filename
 * @returns {string[]}
 */
function findKernelKeysByFile(graph, filename) {
  const keys = [];
  for (const [key, entry] of Object.entries(graph.kernels)) {
    if (entry.kernel === filename) {
      keys.push(key);
    }
  }
  return keys;
}

/**
 * Check whether any kernel in the graph uses the given shader file.
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {string} filename
 * @returns {boolean}
 */
function hasKernelFile(graph, filename) {
  return findKernelKeysByFile(graph, filename).length > 0;
}

/**
 * Create a new kernel entry with the digest cleared (shader changed).
 * @param {object} base - original kernel entry
 * @param {string} newFile - new shader filename
 * @param {string} newEntry - new entry point name
 * @param {object|null} [constants] - override constants (null to remove)
 * @returns {object}
 */
function deriveKernelEntry(base, newFile, newEntry, constants) {
  const derived = { ...base, kernel: newFile, entry: newEntry, digest: null };
  if (constants === null) {
    delete derived.constants;
  } else if (constants !== undefined) {
    derived.constants = { ...constants };
  }
  return derived;
}

/**
 * Derive a non-colliding kernel key name.
 * @param {object} kernels - existing kernels dict
 * @param {string} baseKey - original key
 * @param {string} suffix - suffix to append
 * @returns {string}
 */
function deriveKernelKey(kernels, baseKey, suffix) {
  const candidate = `${baseKey}${suffix}`;
  if (!kernels[candidate]) {
    return candidate;
  }
  let counter = 2;
  while (kernels[`${candidate}_${counter}`]) {
    counter++;
  }
  return `${candidate}_${counter}`;
}

/**
 * Replace kernel key references in step tuples.
 * @param {Array<Array>} steps
 * @param {Map<string, string>} keyMap - oldKey → newKey
 * @returns {Array<Array>}
 */
function remapStepKeys(steps, keyMap) {
  return steps.map((step) => {
    const kernelKey = step[1];
    const replacement = keyMap.get(kernelKey);
    if (replacement !== undefined) {
      const newStep = [...step];
      newStep[1] = replacement;
      return newStep;
    }
    return step;
  });
}

/**
 * Check whether a step tuple's kernel key resolves to the given shader file.
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {Array} step
 * @param {string} filename
 * @returns {boolean}
 */
function stepUsesFile(graph, step, filename) {
  const kernelKey = step[1];
  const entry = graph.kernels[kernelKey];
  return entry != null && entry.kernel === filename;
}

/**
 * Find the first kernel key used by matching ops in a phase whose shader file
 * satisfies the provided predicate.
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {Array<Array>} steps
 * @param {ReadonlySet<string>} ops
 * @param {(entry: { kernel: string, entry: string }) => boolean} predicate
 * @returns {string | null}
 */
function findPhaseKernelKey(graph, steps, ops, predicate) {
  for (const step of steps || []) {
    if (!ops.has(step[0])) {
      continue;
    }
    const entry = graph.kernels[step[1]];
    if (entry && predicate(entry)) {
      return step[1];
    }
  }
  return null;
}

/**
 * Find an existing kernel key by shader file and entry point.
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {string} filename
 * @param {string} entryPoint
 * @returns {string | null}
 */
function findKernelKeyByFileAndEntry(graph, filename, entryPoint) {
  for (const [key, entry] of Object.entries(graph.kernels)) {
    if (entry.kernel === filename && entry.entry === entryPoint) {
      return key;
    }
  }
  return null;
}

function normalizeLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

function isLinearAttentionLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net';
}

function buildGroupedLayerEntries(baseStep, targetLayers, replacementKernelKey) {
  const groupedEntries = [];
  if (!Array.isArray(baseStep) || baseStep.length < 2) {
    return groupedEntries;
  }

  const totalLayers = targetLayers.allLayers;
  const targeted = targetLayers.matchingLayers;
  const remaining = totalLayers.filter((layerIdx) => !targeted.includes(layerIdx));

  if (remaining.length > 0) {
    groupedEntries.push({
      layers: remaining,
      steps: [baseStep],
    });
  }
  if (targeted.length > 0) {
    const replacement = [...baseStep];
    replacement[1] = replacementKernelKey;
    groupedEntries.push({
      layers: targeted,
      steps: [replacement],
    });
  }

  return groupedEntries;
}

function replacePhaseStepKernelKey(steps, op, replacementKernelKey) {
  if (!Array.isArray(steps) || steps.length === 0 || !replacementKernelKey) {
    return { steps, changed: false };
  }
  let changed = false;
  const nextSteps = steps.map((step) => {
    if (!Array.isArray(step) || step[0] !== op) {
      return step;
    }
    if (step[1] === replacementKernelKey) {
      return step;
    }
    const replacement = [...step];
    replacement[1] = replacementKernelKey;
    changed = true;
    return replacement;
  });
  return { steps: nextSteps, changed };
}

function deriveKernelEntryWithPrecision(base, precision) {
  return {
    ...base,
    precision: {
      ...(base.precision ?? {}),
      ...precision,
    },
  };
}

function deriveLinearDecodeF16KernelEntry(base) {
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4.wgsl' && base.entry === 'main_multicol') {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_multicol_f16a.wgsl', 'main_multicol_f16a'),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (
    (base.kernel === 'fused_matmul_q4_multicol_f16.wgsl' && base.entry === 'main_multicol_f16')
    || (base.kernel === 'fused_matmul_q4_multicol_f16a.wgsl' && base.entry === 'main_multicol_f16a')
  ) {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

function deriveLmHeadDecodeF16KernelEntry(base) {
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'matmul_gemv_subgroup.wgsl' && base.entry === 'main_multicol') {
    return {
      ...deriveKernelEntry(base, 'matmul_gemv_subgroup_f16a.wgsl', 'main_multicol'),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (base.kernel === 'matmul_gemv_subgroup_f16a.wgsl' && base.entry === 'main_multicol') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

// =============================================================================
// Transform: removeSubgroups
// =============================================================================

/**
 * Remove subgroup shader dependencies from decode and postLayer steps.
 * Prefill steps are left untouched (they already use tiled matmul).
 *
 * Returns null if the graph has no subgroup kernels.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function removeSubgroups(graph, ctx) {
  const hasAnyFallbackKernel = Object.values(graph.kernels).some(requiresNoSubgroupFallback);
  if (!hasAnyFallbackKernel) {
    return null;
  }

  const result = cloneGraph(graph);
  const keyMap = new Map();
  const isF16Activation = ctx.activationDtype === 'f16';

  // Build replacement kernel entries for each subgroup or fused-Q4K kernel
  // reference found in decode, prefill, and postLayer steps.
  const decodeKeys = new Set((result.decode || []).map((s) => s[1]));
  const prefillKeys = new Set((result.prefill || []).map((s) => s[1]));
  const postLayerKeys = new Set((result.postLayer || []).map((s) => s[1]));
  const relevantKeys = new Set([...decodeKeys, ...prefillKeys, ...postLayerKeys]);

  for (const key of relevantKeys) {
    const entry = result.kernels[key];
    if (!entry || !requiresNoSubgroupFallback(entry)) {
      continue;
    }

    const isPostLayer = postLayerKeys.has(key) && !decodeKeys.has(key);
    const isMulticol = entry.entry === 'main_multicol';
    const isLmHead = isPostLayer || isMulticol;

    let newFile;
    let newEntry = 'main';
    let newConstants = undefined;

    if (entry.kernel === 'matmul_gemv_subgroup.wgsl') {
      if (isLmHead) {
        // lm_head: multicol → plain matmul, remove MULTICOL constants
        newFile = 'matmul_f16w_f32a.wgsl';
        newConstants = null;
      } else {
        // decode projections: vec4 → tiled matmul
        newFile = 'matmul_f16w_f32a_tiled.wgsl';
      }
    } else if (entry.kernel === 'matmul_gemv_subgroup_f16a.wgsl') {
      if (isLmHead) {
        newFile = isF16Activation ? 'matmul_f16.wgsl' : 'matmul_f16w_f32a.wgsl';
        newConstants = null;
      } else {
        newFile = isF16Activation ? 'matmul_f16.wgsl' : 'matmul_f16w_f32a_tiled.wgsl';
      }
    } else if (entry.kernel === 'attention_decode_online_f16kv.wgsl') {
      // f16kv online uses f32 Q; if activations are f16, fall back to all-f16 chunked
      newFile = isF16Activation
        ? 'attention_decode_chunked_f16.wgsl'
        : 'attention_decode_chunked_f16kv.wgsl';
      newEntry = entry.entry;
    } else if (entry.kernel === 'attention_decode_online_f16.wgsl') {
      newFile = 'attention_decode_chunked_f16.wgsl';
      newEntry = entry.entry;
    } else if (entry.kernel.startsWith('fused_matmul_q4')) {
      newFile = isF16Activation ? 'matmul_f16_tiled.wgsl' : 'matmul_f16w_f32a_tiled.wgsl';
      newConstants = null;
    } else {
      // Unknown subgroup kernel — skip
      continue;
    }

    const newKey = deriveKernelKey(result.kernels, key, '_nosg');
    result.kernels[newKey] = deriveKernelEntry(entry, newFile, newEntry, newConstants);
    keyMap.set(key, newKey);
  }

  if (keyMap.size === 0) {
    return null;
  }

  // Remap decode, prefill, and postLayer steps; leave preLayer untouched
  result.decode = remapStepKeys(result.decode || [], keyMap);
  result.prefill = remapStepKeys(result.prefill || [], keyMap);
  result.postLayer = remapStepKeys(result.postLayer || [], keyMap);

  return result;
}

// =============================================================================
// Transform: widenToF32Activations
// =============================================================================

/**
 * Activation-only widening: f16-activation shaders → f32-activation variants
 * that still use f16 for weights and KV cache. Requires shader-f16 for weight
 * and KV buffer reads.
 * @type {ReadonlyMap<string, string>}
 */
const F16_TO_F32_ACTIVATION_MAP = new Map([
  ['rmsnorm_f16.wgsl', 'rmsnorm.wgsl'],
  ['rope_f16.wgsl', 'rope.wgsl'],
  ['residual_f16.wgsl', 'residual.wgsl'],
  ['gelu_f16.wgsl', 'gelu.wgsl'],
  ['sample_f16.wgsl', 'sample.wgsl'],
  ['gather_f16.wgsl', 'gather.wgsl'],
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_gemv_subgroup.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f16w_f32a.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f16w_f32a_tiled.wgsl'],
  ['attention_decode_online_f16.wgsl', 'attention_decode_online_f16kv.wgsl'],
  ['attention_decode_chunked_f16.wgsl', 'attention_decode_chunked_f16kv.wgsl'],
  ['attention_small_f16.wgsl', 'attention_small_f16kv.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_streaming_f16kv.wgsl'],
]);

/**
 * Full f32 widening: every shader that uses `enable f16;` is replaced with a
 * pure-f32 equivalent. Used when the GPU cannot compile any f16 WGSL at all.
 * Covers f16-activation, f16-weight (f16w), and f16-KV (f16kv) kernels.
 * @type {ReadonlyMap<string, string>}
 */
const FULL_F32_SHADER_MAP = new Map([
  // f16-activation utility kernels → f32
  ['rmsnorm_f16.wgsl', 'rmsnorm.wgsl'],
  ['rope_f16.wgsl', 'rope.wgsl'],
  ['residual_f16.wgsl', 'residual.wgsl'],
  ['gelu_f16.wgsl', 'gelu.wgsl'],
  ['sample_f16.wgsl', 'sample.wgsl'],
  ['gather_f16.wgsl', 'gather.wgsl'],
  // f16-activation matmul → f32
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f32.wgsl'],
  // f16-weight + f32-activation matmul → f32
  ['matmul_gemv_subgroup.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16w_f32a.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16w_f32a_tiled.wgsl', 'matmul_f32.wgsl'],
  // f16-activation attention → f32
  ['attention_decode_online_f16.wgsl', 'attention_decode.wgsl'],
  ['attention_decode_chunked_f16.wgsl', 'attention_decode.wgsl'],
  ['attention_small_f16.wgsl', 'attention_small.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_streaming.wgsl'],
  // f16kv attention (f32 Q, f16 KV) → f32
  ['attention_decode_online_f16kv.wgsl', 'attention_decode.wgsl'],
  ['attention_decode_chunked_f16kv.wgsl', 'attention_decode.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_small.wgsl'],
  ['attention_streaming_f16kv.wgsl', 'attention_streaming.wgsl'],
]);

/**
 * Widen all f16-activation shaders to f32-activation equivalents.
 *
 * Returns null if the graph contains fused_ffn_f16.wgsl (no direct f32
 * equivalent exists) or if no f16 activation shaders are present.
 *
 * NOTE: The caller is responsible for also updating session.activationDtype
 * to reflect the widened dtype.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function widenToF32Activations(graph, ctx) {
  // Bail out if fused f16 FFN is present — no direct f32 equivalent
  if (hasKernelFile(graph, 'fused_ffn_f16.wgsl')) {
    return null;
  }

  // When the GPU cannot compile any f16 WGSL (hasF16=false), use the full f32
  // map that also covers f16-weight and f16-KV kernels. Otherwise use the
  // activation-only map that preserves f16 weights/KV for precision fallback.
  const shaderMap = ctx.capabilities?.hasF16 === false
    ? FULL_F32_SHADER_MAP
    : F16_TO_F32_ACTIVATION_MAP;

  const hasTargetShader = Object.values(graph.kernels).some(
    (entry) => shaderMap.has(entry.kernel)
  );
  if (!hasTargetShader) {
    return null;
  }

  const result = cloneGraph(graph);

  for (const [key, entry] of Object.entries(result.kernels)) {
    const replacement = shaderMap.get(entry.kernel);
    if (replacement !== undefined) {
      result.kernels[key] = deriveKernelEntry(entry, replacement, entry.entry);
    }
  }

  return result;
}

// =============================================================================
// Transform: swapPrefillAttention
// =============================================================================

/** @type {ReadonlyMap<string, string>} */
const PREFILL_ATTENTION_PAIRS = new Map([
  ['attention_streaming_f16kv.wgsl', 'attention_small_f16kv.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_streaming_f16kv.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_small_f16.wgsl'],
  ['attention_small_f16.wgsl', 'attention_streaming_f16.wgsl'],
]);

/**
 * Swap prefill attention kernel between streaming and small variants.
 *
 * The `opts` parameter specifies the direction:
 *   { from: 'attention_streaming_f16kv.wgsl', to: 'attention_small_f16kv.wgsl' }
 *
 * If `from`/`to` are not provided, uses the bidirectional pair map.
 * Returns null if no matching prefill attention kernel is found.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @param {{ from?: string, to?: string }} [opts]
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
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
      // Verify this kernel is actually used in a prefill step
      const usedInPrefill = (graph.prefill || []).some((step) => step[1] === key);
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

/**
 * Promote prefill attention onto the fixed 256-dim shared-block kernel.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useHead256PrefillAttention(graph, ctx) {
  return (
    swapPrefillAttention(graph, ctx, {
      from: 'attention_small_f16kv.wgsl',
      to: 'attention_head256_f16kv.wgsl',
    })
    || swapPrefillAttention(graph, ctx, {
      from: 'attention_streaming_f16kv.wgsl',
      to: 'attention_head256_f16kv.wgsl',
    })
  );
}

// =============================================================================
// Transform: widenProjectionWeightsToF32
// =============================================================================

/** @type {ReadonlySet<string>} */
const PROJECTION_MATMUL_FILES = new Set([
  'matmul_gemv_subgroup.wgsl',
  'matmul_gemv_subgroup_f16a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16w_f32a.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);

/**
 * Known layer projection ops. Only these are widened; lm_head and embed are
 * excluded.
 * @type {ReadonlySet<string>}
 */
const LAYER_PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
]);

const DENSE_Q4_PREFILL_FILES = new Set([
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);

function resolveDensePrefillProjectionKernel(ctx) {
  return ctx.activationDtype === 'f16'
    ? 'matmul_f16.wgsl'
    : 'matmul_f16w_f32a.wgsl';
}

/**
 * Replace projection matmul kernels with f32 weight variants.
 *
 * Applies only to layer projection steps (q/k/v/o/gate/up/down), NOT lm_head
 * or embed.
 *
 * Returns null if no applicable projection kernels are found.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function widenProjectionWeightsToF32(graph, ctx) {
  // Collect kernel keys used by layer projection steps across all phases
  const projectionKernelKeys = new Set();
  const allPhases = ['preLayer', 'decode', 'prefill', 'postLayer'];

  for (const phase of allPhases) {
    const steps = graph[phase];
    if (!Array.isArray(steps)) {
      continue;
    }
    for (const step of steps) {
      const op = step[0];
      const kernelKey = step[1];
      if (LAYER_PROJECTION_OPS.has(op) && kernelKey) {
        projectionKernelKeys.add(kernelKey);
      }
    }
  }

  if (projectionKernelKeys.size === 0) {
    return null;
  }

  // Check whether any of those keys reference a swappable matmul
  const keysToSwap = new Set();
  for (const key of projectionKernelKeys) {
    const entry = graph.kernels[key];
    if (entry && PROJECTION_MATMUL_FILES.has(entry.kernel)) {
      keysToSwap.add(key);
    }
  }

  if (keysToSwap.size === 0) {
    return null;
  }

  const result = cloneGraph(graph);

  for (const key of keysToSwap) {
    const entry = result.kernels[key];
    result.kernels[key] = deriveKernelEntry(entry, 'matmul_f32.wgsl', 'main');
  }

  return result;
}

// =============================================================================
// Transform: remapDenseQ4KPrefillToQ4Native
// =============================================================================

/**
 * Replace dense prefill projection kernels with Q4-native prefill variants.
 *
 * This applies only when the graph already exposes a compatible fused Q4 decode
 * projection kernel. All prefill layer projections are remapped to the shared-A
 * batched multicol Q4 prefill kernel so the transformed path remains valid for
 * `M > 1` prefill workloads.
 *
 * Returns null when the graph does not have the required dense-prefill + Q4
 * decode shape.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function remapDenseQ4KPrefillToQ4Native(graph, ctx) {
  const densePrefillProjectionSteps = (graph.prefill || []).filter((step) => {
    if (!LAYER_PROJECTION_OPS.has(step[0])) {
      return false;
    }
    const entry = graph.kernels[step[1]];
    return entry != null && DENSE_Q4_PREFILL_FILES.has(entry.kernel);
  });
  if (densePrefillProjectionSteps.length === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  const existingSharedKey = findKernelKeyByFileAndEntry(
    result,
    'fused_matmul_q4_batched_multicol_shared.wgsl',
    'main'
  );
  let sharedKey = existingSharedKey;
  if (!sharedKey) {
    const q4DecodeKey = findPhaseKernelKey(
      graph,
      graph.decode || [],
      LAYER_PROJECTION_OPS,
      (entry) => entry.kernel === 'fused_matmul_q4.wgsl'
    );
    if (!q4DecodeKey) {
      return null;
    }
    const q4DecodeEntry = result.kernels[q4DecodeKey];
    sharedKey = deriveKernelKey(result.kernels, q4DecodeKey, '_prefill_shared');
    result.kernels[sharedKey] = deriveKernelEntry(
      q4DecodeEntry,
      'fused_matmul_q4_batched_multicol_shared.wgsl',
      'main',
      null
    );
  }

  let changed = false;
  result.prefill = (result.prefill || []).map((step) => {
    const op = step[0];
    if (!LAYER_PROJECTION_OPS.has(op)) {
      return step;
    }
    const entry = result.kernels[step[1]];
    if (!entry || !DENSE_Q4_PREFILL_FILES.has(entry.kernel)) {
      return step;
    }

    const replacementKey = sharedKey;
    if (replacementKey === step[1]) {
      return step;
    }
    changed = true;
    const next = [...step];
    next[1] = replacementKey;
    return next;
  });

  return changed ? result : null;
}

// =============================================================================
// Transform: remapQ4KPrefillToDense
// =============================================================================

/**
 * Replace fused Q4K prefill projection kernels with dense tiled variants.
 *
 * Decode remains unchanged so the runtime can keep using fused Q4K decode while
 * the loader exposes mixed dense+Q4K materializations for prefill.
 *
 * Returns null when the graph has no fused Q4K prefill projection kernels.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function remapQ4KPrefillToDense(graph, ctx) {
  const q4PrefillProjectionSteps = (graph.prefill || []).filter((step) => {
    if (!LAYER_PROJECTION_OPS.has(step[0])) {
      return false;
    }
    const entry = graph.kernels[step[1]];
    return entry != null && entry.kernel.startsWith('fused_matmul_q4');
  });
  if (q4PrefillProjectionSteps.length === 0) {
    return null;
  }

  const denseKernelFile = resolveDensePrefillProjectionKernel(ctx);
  const result = cloneGraph(graph);
  let denseKey = findKernelKeyByFileAndEntry(result, denseKernelFile, 'main');
  if (!denseKey) {
    const sourceKey = q4PrefillProjectionSteps[0][1];
    const sourceEntry = result.kernels[sourceKey];
    denseKey = deriveKernelKey(result.kernels, sourceKey, '_prefill_dense');
    result.kernels[denseKey] = deriveKernelEntry(
      sourceEntry,
      denseKernelFile,
      'main',
      null
    );
  }

  let changed = false;
  result.prefill = (result.prefill || []).map((step) => {
    if (!LAYER_PROJECTION_OPS.has(step[0])) {
      return step;
    }
    const entry = result.kernels[step[1]];
    if (!entry || !entry.kernel.startsWith('fused_matmul_q4')) {
      return step;
    }
    if (step[1] === denseKey) {
      return step;
    }
    changed = true;
    const next = [...step];
    next[1] = denseKey;
    return next;
  });

  return changed ? result : null;
}

// =============================================================================
// Transform: useLinearDecodeProjectionF16
// =============================================================================

/**
 * Remap the linear-attention q_proj decode step onto the f16-activation fused
 * Q4 kernel for linear-attention layers only. Full-attention layers keep the
 * manifest-wide f32 activation contract.
 *
 * Only q_proj is remapped.  o_proj is intentionally excluded: the o_proj
 * output enters the residual stream directly, and f16 truncation there
 * accumulates across the 18 linear-attention layers in the Qwen 3.5 pattern,
 * corrupting the logit distribution (empirically verified: degenerate
 * repetitive output under greedy decode).  q_proj f16 is safe because the
 * linear attention core absorbs the f16 input into its f32 internal state.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useLinearDecodeProjectionF16(graph, ctx) {
  const layerTypes = Array.isArray(ctx.layerTypes) ? ctx.layerTypes : null;
  if (!layerTypes || layerTypes.length === 0) {
    return null;
  }

  const matchingLayers = layerTypes
    .map((layerType, layerIdx) => ({ layerType, layerIdx }))
    .filter(({ layerType }) => isLinearAttentionLayerType(layerType))
    .map(({ layerIdx }) => layerIdx);
  if (matchingLayers.length === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  const targetLayers = {
    allLayers: layerTypes.map((_, layerIdx) => layerIdx),
    matchingLayers,
  };
  const qProjIndex = (result.decode || []).findIndex((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  if (qProjIndex === -1) {
    return null;
  }
  const qProjStep = result.decode[qProjIndex];
  const qProjKernelKey = qProjStep[1];
  const qProjKernel = result.kernels[qProjKernelKey];
  if (!qProjKernel) {
    return null;
  }

  const derivedEntry = deriveLinearDecodeF16KernelEntry(qProjKernel);
  if (!derivedEntry) {
    return null;
  }

  const derivedKey = deriveKernelKey(result.kernels, qProjKernelKey, '_linear_f16out');
  result.kernels[derivedKey] = derivedEntry;
  const groupedEntries = buildGroupedLayerEntries(qProjStep, targetLayers, derivedKey);
  if (groupedEntries.length === 0) {
    return null;
  }
  result.decode = [
    ...result.decode.slice(0, qProjIndex),
    ...groupedEntries,
    ...result.decode.slice(qProjIndex + 1),
  ];

  return result;
}

// =============================================================================
// Transform: remapQ4KDecodeToGemv
// =============================================================================

/**
 * Replace fused Q4K decode projection kernels with GEMV subgroup variants.
 *
 * When Q4K weights have f16 materializations (mixed/dense loader mode), the
 * GEMV subgroup kernel on pre-dequantized f16 weights is significantly faster
 * than the fused Q4K kernel for M=1 decode (empirically 2.3x on Apple M-series).
 *
 * After this transform no decode kernels reference fused_matmul_q4*, which
 * signals the loader to use dense materialization (f16 only — no Q4K buffer
 * retained in GPU memory, reducing peak VRAM).
 *
 * Only layer projection ops are remapped.  Non-matmul ops (rmsnorm, rope,
 * attention, residual, activation) are left untouched.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function remapQ4KDecodeToGemv(graph, ctx) {
  if (ctx.activationDtype === 'f16') {
    return null;
  }

  const decodeSteps = graph.decode || [];
  const fusedDecodeKeys = new Set();
  for (const step of decodeSteps) {
    if (!Array.isArray(step)) continue;
    const kernelKey = step[1];
    const entry = graph.kernels[kernelKey];
    if (entry && entry.kernel.startsWith('fused_matmul_q4')) {
      fusedDecodeKeys.add(kernelKey);
    }
  }
  if (fusedDecodeKeys.size === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  const keyMap = new Map();

  for (const key of fusedDecodeKeys) {
    const newKey = deriveKernelKey(result.kernels, key, '_gemv');
    result.kernels[newKey] = deriveKernelEntry(
      result.kernels[key],
      'matmul_gemv_subgroup.wgsl',
      'main_multicol',
      null
    );
    keyMap.set(key, newKey);
  }

  result.decode = remapStepKeys(result.decode, keyMap);
  return result;
}

// =============================================================================
// Transform: remapQ4KDecodeAttentionToGemv (diagnostic)
// =============================================================================

const ATTENTION_PROJECTION_OPS = new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj']);

/**
 * Replace fused Q4K ATTENTION-ONLY decode projection kernels with GEMV
 * subgroup variants, leaving FFN projections (gate/up/down_proj) untouched.
 *
 * Diagnostic transform for isolating whether the GEMV correctness regression
 * originates in the attention or FFN projection path.  Because FFN ops keep
 * their fused Q4K kernels, `isKernelPathFusedQ4K` stays true and the weight
 * loader remains in mixed-materialization mode.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function remapQ4KDecodeAttentionToGemv(graph, ctx) {
  if (ctx.activationDtype === 'f16') {
    return null;
  }

  const decodeSteps = graph.decode || [];
  const attnFusedKeys = new Set();
  for (const step of decodeSteps) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (!ATTENTION_PROJECTION_OPS.has(op)) continue;
    const kernelKey = step[1];
    const entry = graph.kernels[kernelKey];
    if (entry && entry.kernel.startsWith('fused_matmul_q4')) {
      attnFusedKeys.add(kernelKey);
    }
  }
  if (attnFusedKeys.size === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  const keyMap = new Map();

  for (const key of attnFusedKeys) {
    const newKey = deriveKernelKey(result.kernels, key, '_attn_gemv');
    result.kernels[newKey] = deriveKernelEntry(
      result.kernels[key],
      'matmul_gemv_subgroup.wgsl',
      'main_multicol',
      null
    );
    keyMap.set(key, newKey);
  }

  // Only remap attention projection steps, leave FFN steps unchanged.
  result.decode = result.decode.map((step) => {
    if (!Array.isArray(step)) return step;
    if (!ATTENTION_PROJECTION_OPS.has(step[0])) return step;
    const replacement = keyMap.get(step[1]);
    if (replacement !== undefined) {
      const newStep = [...step];
      newStep[1] = replacement;
      return newStep;
    }
    return step;
  });

  return result;
}

// =============================================================================
// Transform: remapQ4KDecodeFFNToGemv (diagnostic)
// =============================================================================

const FFN_PROJECTION_OPS = new Set(['gate_proj', 'up_proj', 'down_proj']);

/**
 * Replace fused Q4K FFN-ONLY decode projection kernels with GEMV subgroup
 * variants, leaving attention projections (q/k/v/o_proj) as fused Q4K.
 *
 * Diagnostic complement to `remapQ4KDecodeAttentionToGemv`.  Together these
 * two transforms isolate whether the GEMV decode regression originates in
 * the attention or FFN projection path.  Because attention ops keep their
 * fused Q4K kernels, `isKernelPathFusedQ4K` stays true and the weight loader
 * remains in mixed-materialization mode.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function remapQ4KDecodeFFNToGemv(graph, ctx) {
  if (ctx.activationDtype === 'f16') {
    return null;
  }

  const decodeSteps = graph.decode || [];
  const ffnFusedKeys = new Set();
  for (const step of decodeSteps) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (!FFN_PROJECTION_OPS.has(op)) continue;
    const kernelKey = step[1];
    const entry = graph.kernels[kernelKey];
    if (entry && entry.kernel.startsWith('fused_matmul_q4')) {
      ffnFusedKeys.add(kernelKey);
    }
  }
  if (ffnFusedKeys.size === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  const keyMap = new Map();

  for (const key of ffnFusedKeys) {
    const newKey = deriveKernelKey(result.kernels, key, '_ffn_gemv');
    result.kernels[newKey] = deriveKernelEntry(
      result.kernels[key],
      'matmul_gemv_subgroup.wgsl',
      'main_multicol',
      null
    );
    keyMap.set(key, newKey);
  }

  // Only remap FFN projection steps, leave attention steps unchanged.
  result.decode = result.decode.map((step) => {
    if (!Array.isArray(step)) return step;
    if (!FFN_PROJECTION_OPS.has(step[0])) return step;
    const replacement = keyMap.get(step[1]);
    if (replacement !== undefined) {
      const newStep = [...step];
      newStep[1] = replacement;
      return newStep;
    }
    return step;
  });

  return result;
}

// =============================================================================
// Transform: useQwenDecodeF16Matmuls
// =============================================================================

/**
 * Narrow selected Qwen decode matmuls onto explicit f16-input/f16-output
 * kernels while keeping the manifest-wide f32 activation contract intact.
 *
 * This transform is intentionally selective:
 * - FFN gate/up decode matmuls switch to f16a so decode can bypass the slow
 *   fused-q4k FFN path when capability policy opts in.
 * - LM head decode switches to the subgroup f16a GEMV path.
 *
 * FFN down remains on the f32-output contract so the layer residual path stays
 * numerically aligned with the manifest-owned activation dtype.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useQwenDecodeF16Matmuls(graph, ctx) {
  const modelId = typeof ctx.modelId === 'string' ? ctx.modelId.trim() : '';
  if (modelId !== 'qwen-3-5-0-8b-q4k-ehaf16') {
    return null;
  }

  const result = cloneGraph(graph);
  let changed = false;

  for (const op of ['gate_proj', 'up_proj']) {
    const stepIndex = (result.decode || []).findIndex((entry) => Array.isArray(entry) && entry[0] === op);
    if (stepIndex === -1) {
      continue;
    }
    const step = result.decode[stepIndex];
    const kernelKey = step[1];
    const kernelEntry = result.kernels[kernelKey];
    if (!kernelEntry) {
      continue;
    }
    const derivedEntry = deriveLinearDecodeF16KernelEntry(kernelEntry);
    if (!derivedEntry) {
      continue;
    }
    const derivedKey = deriveKernelKey(result.kernels, kernelKey, '_decode_f16out');
    result.kernels[derivedKey] = derivedEntry;
    const replacement = [...step];
    replacement[1] = derivedKey;
    result.decode = [
      ...result.decode.slice(0, stepIndex),
      replacement,
      ...result.decode.slice(stepIndex + 1),
    ];
    changed = true;
  }

  const postLayerResult = replacePhaseStepKernelKey(
    result.postLayer ?? [],
    'lm_head',
    (() => {
      const lmHeadStep = (result.postLayer || []).find((entry) => Array.isArray(entry) && entry[0] === 'lm_head');
      if (!lmHeadStep) {
        return null;
      }
      const lmHeadKernelKey = lmHeadStep[1];
      const lmHeadKernel = result.kernels[lmHeadKernelKey];
      if (!lmHeadKernel) {
        return null;
      }
      const derivedEntry = deriveLmHeadDecodeF16KernelEntry(lmHeadKernel);
      if (!derivedEntry) {
        return null;
      }
      const derivedKey = deriveKernelKey(result.kernels, lmHeadKernelKey, '_decode_f16out');
      result.kernels[derivedKey] = derivedEntry;
      return derivedKey;
    })()
  );
  if (postLayerResult.changed) {
    result.postLayer = postLayerResult.steps;
    changed = true;
  }

  return changed ? result : null;
}

// =============================================================================
// Composition
// =============================================================================

/**
 * Compose multiple transforms into a single transform function.
 *
 * Each transform is applied sequentially. If a transform returns null
 * (not applicable), the graph passes through unchanged.
 *
 * @param {Array<(graph: import('./execution-graph-transforms.js').ExecutionGraph, ctx: import('./execution-graph-transforms.js').TransformContext) => import('./execution-graph-transforms.js').ExecutionGraph | null>} transforms
 * @returns {(graph: import('./execution-graph-transforms.js').ExecutionGraph, ctx: import('./execution-graph-transforms.js').TransformContext) => import('./execution-graph-transforms.js').ExecutionGraph}
 */
export function composeTransforms(...transforms) {
  return (graph, ctx) => {
    let current = graph;
    for (const transform of transforms) {
      const result = transform(current, ctx);
      if (result !== null && result !== undefined) {
        current = result;
      }
    }
    return current;
  };
}

// =============================================================================
// Registry
// =============================================================================

/** @type {Readonly<Record<string, Function>>} */
export const TRANSFORMS = Object.freeze({
  removeSubgroups,
  widenToF32Activations,
  swapPrefillAttention,
  useHead256PrefillAttention,
  widenProjectionWeightsToF32,
  remapDenseQ4KPrefillToQ4Native,
  remapQ4KPrefillToDense,
  useLinearDecodeProjectionF16,
  remapQ4KDecodeToGemv,
  remapQ4KDecodeAttentionToGemv,
  remapQ4KDecodeFFNToGemv,
  useQwenDecodeF16Matmuls,
  composeTransforms,
});
