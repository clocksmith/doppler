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

const KERNEL_FILE_PRECISION_PATCHES = new Map([
  ['matmul_gemv_subgroup_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['matmul_f16.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['matmul_f16_tiled.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['silu_f16.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4_multicol_f16.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4_multicol_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4_batched_f16acc_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['fused_matmul_q4.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['fused_matmul_q4_batched.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['fused_matmul_q4_batched_multicol_shared.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['fused_matmul_q4_widetile_f16a.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['matmul_gemv_subgroup.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['matmul_f16w_f32a.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['matmul_f16w_f32a_tiled.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['matmul_f32.wgsl', { inputDtype: 'f32', outputDtype: 'f32' }],
  ['gather_f16_f16_out.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
  ['gather_f16_vec4_f16_out.wgsl', { inputDtype: 'f16', outputDtype: 'f16' }],
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
  const precision = deriveKernelPrecision(base, newFile);
  if (precision) {
    derived.precision = precision;
  } else {
    delete derived.precision;
  }
  return derived;
}

function deriveMappedKernelEntry(base, newFile) {
  const newEntry = newFile === 'matmul_f32.wgsl' ? 'main' : base.entry;
  return deriveKernelEntry(base, newFile, newEntry);
}

function deriveKernelPrecision(base, newFile) {
  const precision = base.precision ? { ...base.precision } : {};
  const precisionPatch = KERNEL_FILE_PRECISION_PATCHES.get(newFile);
  if (precisionPatch) {
    Object.assign(precision, precisionPatch);
  }
  if (!String(newFile).startsWith('attention')) {
    return Object.keys(precision).length > 0 ? precision : null;
  }
  if (newFile.includes('_f16kv')) {
    precision.activationDtype = 'f32';
    precision.kvDtype = 'f16';
    return precision;
  }
  if (newFile.includes('_f16')) {
    precision.activationDtype = 'f16';
    precision.kvDtype = 'f16';
    return precision;
  }
  precision.activationDtype = 'f32';
  precision.kvDtype = 'f32';
  return precision;
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

function isFullAttentionLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'full_attention'
    || normalized === 'full'
    || normalized === 'global'
    || normalized === 'standard';
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

function findPhaseStep(steps, op) {
  if (!Array.isArray(steps) || !op) {
    return null;
  }
  for (const entry of steps) {
    if (Array.isArray(entry)) {
      if (entry[0] === op) {
        return entry;
      }
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.steps)) {
      continue;
    }
    const nested = findPhaseStep(entry.steps, op);
    if (nested) {
      return nested;
    }
  }
  return null;
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

const ATTENTION_F16KV_TO_F16_MAP = new Map([
  ['attention_decode_online_f16kv.wgsl', 'attention_decode_online_f16.wgsl'],
  ['attention_decode_chunked_f16kv.wgsl', 'attention_decode_chunked_f16.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_small_f16.wgsl'],
  ['attention_streaming_f16kv.wgsl', 'attention_streaming_f16.wgsl'],
  ['attention_head512_f16kv.wgsl', 'attention_head512_f16.wgsl'],
]);

function deriveF16AttentionKernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const replacement = ATTENTION_F16KV_TO_F16_MAP.get(base.kernel);
  if (!replacement) {
    return null;
  }
  return deriveKernelEntryWithPrecision(
    deriveKernelEntry(base, replacement, base.entry),
    { activationDtype: 'f16', kvDtype: 'f16' }
  );
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

function deriveDenseDecodeF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'matmul_gemv_subgroup.wgsl') {
    return {
      ...deriveKernelEntry(base, 'matmul_gemv_subgroup_f16a.wgsl', base.entry ?? 'main'),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (base.kernel === 'matmul_gemv_subgroup_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

function deriveQ4DecodeF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4.wgsl') {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_multicol_f16a.wgsl', 'main_multicol_f16a', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (
    base.kernel === 'fused_matmul_q4_multicol_f16.wgsl'
    || base.kernel === 'fused_matmul_q4_multicol_f16a.wgsl'
  ) {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

function deriveQ4PrefillF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  if (base.kernel.startsWith('fused_matmul_q4_batched')) {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_batched_f16a.wgsl', 'main_batched_f16a', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  return null;
}

function deriveQ4WideTilePrefillF16KernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (base.kernel === 'fused_matmul_q4_widetile.wgsl') {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_widetile_f16a.wgsl', 'main', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  if (base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(base, precision);
  }
  return null;
}

function deriveQ4PrefillF16AccumKernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f16',
    outputDtype: 'f16',
  };
  if (
    base.kernel === 'fused_matmul_q4_widetile.wgsl'
    || base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl'
    || base.kernel.startsWith('fused_matmul_q4_batched')
  ) {
    return {
      ...deriveKernelEntry(base, 'fused_matmul_q4_batched_f16acc_f16a.wgsl', 'main_batched_f16acc_f16a', null),
      precision: {
        ...(base.precision ?? {}),
        ...precision,
      },
    };
  }
  return null;
}

function deriveQ4DecodeF32ActivationKernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const precision = {
    inputDtype: 'f32',
    outputDtype: 'f32',
  };
  if (
    base.kernel === 'fused_matmul_q4_multicol_f16.wgsl'
    || base.kernel === 'fused_matmul_q4_multicol_f16a.wgsl'
  ) {
    return deriveKernelEntryWithPrecision(
      deriveKernelEntry(base, 'fused_matmul_q4.wgsl', 'main_multicol', null),
      precision
    );
  }
  if (base.kernel === 'fused_matmul_q4_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(
      deriveKernelEntry(base, 'fused_matmul_q4.wgsl', 'main', null),
      precision
    );
  }
  return null;
}

function deriveQ4PrefillF32ActivationKernelEntry(base) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  if (
    base.kernel !== 'fused_matmul_q4_batched_f16.wgsl'
    && base.kernel !== 'fused_matmul_q4_batched_f16a.wgsl'
    && base.kernel !== 'fused_matmul_q4_widetile_f16a.wgsl'
  ) {
    return null;
  }
  if (base.kernel === 'fused_matmul_q4_widetile_f16a.wgsl') {
    return deriveKernelEntryWithPrecision(
      deriveKernelEntry(base, 'fused_matmul_q4_widetile.wgsl', 'main', null),
      { inputDtype: 'f32', outputDtype: 'f32' }
    );
  }
  return deriveKernelEntryWithPrecision(
    deriveKernelEntry(base, 'fused_matmul_q4_batched.wgsl', 'main_batched', null),
    { inputDtype: 'f32', outputDtype: 'f32' }
  );
}

function replacePhaseStepEntries(steps, op, replacementEntries) {
  if (!Array.isArray(steps) || steps.length === 0 || !Array.isArray(replacementEntries) || replacementEntries.length === 0) {
    return { steps, changed: false };
  }
  const stepIndex = steps.findIndex((entry) => Array.isArray(entry) && entry[0] === op);
  if (stepIndex === -1) {
    return { steps, changed: false };
  }
  return {
    steps: [
      ...steps.slice(0, stepIndex),
      ...replacementEntries,
      ...steps.slice(stepIndex + 1),
    ],
    changed: true,
  };
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
  ['silu_f16.wgsl', 'silu.wgsl'],
  ['sample_f16.wgsl', 'sample.wgsl'],
  ['gather_f16.wgsl', 'gather.wgsl'],
  ['gather_f16_f16_out.wgsl', 'gather.wgsl'],
  ['gather_f16_vec4_f16_out.wgsl', 'gather.wgsl'],
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_gemv_subgroup.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f16w_f32a.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f16w_f32a_tiled.wgsl'],
  ['attention_decode_online_f16.wgsl', 'attention_decode_online_f16kv.wgsl'],
  ['attention_decode_chunked_f16.wgsl', 'attention_decode_chunked_f16kv.wgsl'],
  ['attention_small_f16.wgsl', 'attention_small_f16kv.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_streaming_f16kv.wgsl'],
  ['attention_head512_f16.wgsl', 'attention_head512_f16kv.wgsl'],
]);

/**
 * Activation-only narrowing: f32-activation shaders that still consume f16
 * weights/KV are rewritten onto the matching f16-activation lane.
 *
 * This is the inverse of `F16_TO_F32_ACTIVATION_MAP` and is used when a
 * runtime session explicitly requests f16 activations for an execution-v1
 * graph that was authored with conservative f32 activation defaults.
 * @type {ReadonlyMap<string, string>}
 */
const F32_TO_F16_ACTIVATION_MAP = new Map(
  Array.from(F16_TO_F32_ACTIVATION_MAP.entries(), ([from, to]) => [to, from])
);
F32_TO_F16_ACTIVATION_MAP.set('gather.wgsl', 'gather_f16.wgsl');
F32_TO_F16_ACTIVATION_MAP.set('attention_head256_f16kv.wgsl', 'attention_small_f16.wgsl');
// head512 pure-f16 prefill is model-scoped below; keep generic Gemma 4 E2B
// f16 requests fail-closed until that path has its own evidence.
F32_TO_F16_ACTIVATION_MAP.delete('attention_head512_f16kv.wgsl');

function hasExplicitF32ActivationContract(entry) {
  const precision = entry?.precision;
  if (!precision || typeof precision !== 'object') {
    return false;
  }
  return precision.activationDtype === 'f32'
    || precision.inputDtype === 'f32'
    || precision.outputDtype === 'f32';
}

/**
 * Correctness fallback: preserve f16 weights where possible, but widen both
 * activations and KV-cache interactions onto the stable f32 execution lane.
 * Used for alternate-plan recovery after finiteness failure.
 * @type {ReadonlyMap<string, string>}
 */
const F16_TO_F32_CORRECTNESS_FALLBACK_MAP = new Map([
  ['rmsnorm_f16.wgsl', 'rmsnorm.wgsl'],
  ['rope_f16.wgsl', 'rope.wgsl'],
  ['residual_f16.wgsl', 'residual.wgsl'],
  ['gelu_f16.wgsl', 'gelu.wgsl'],
  ['silu_f16.wgsl', 'silu.wgsl'],
  ['sample_f16.wgsl', 'sample.wgsl'],
  ['gather_f16.wgsl', 'gather.wgsl'],
  ['gather_f16_f16_out.wgsl', 'gather.wgsl'],
  ['gather_f16_vec4_f16_out.wgsl', 'gather.wgsl'],
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_gemv_subgroup.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f16w_f32a.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f16w_f32a_tiled.wgsl'],
  ['attention_decode_online_f16.wgsl', 'attention_streaming.wgsl'],
  ['attention_decode_chunked_f16.wgsl', 'attention_streaming.wgsl'],
  ['attention_small_f16.wgsl', 'attention_small.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_streaming.wgsl'],
  ['attention_decode_online_f16kv.wgsl', 'attention_streaming.wgsl'],
  ['attention_decode_chunked_f16kv.wgsl', 'attention_streaming.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_small.wgsl'],
  ['attention_streaming_f16kv.wgsl', 'attention_streaming.wgsl'],
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
  ['silu_f16.wgsl', 'silu.wgsl'],
  ['sample_f16.wgsl', 'sample.wgsl'],
  ['gather_f16.wgsl', 'gather.wgsl'],
  ['gather_f16_f16_out.wgsl', 'gather.wgsl'],
  ['gather_f16_vec4_f16_out.wgsl', 'gather.wgsl'],
  // f16-activation matmul → f32
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f32.wgsl'],
  // f16-weight + f32-activation matmul → f32
  ['matmul_gemv_subgroup.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16w_f32a.wgsl', 'matmul_f32.wgsl'],
  ['matmul_f16w_f32a_tiled.wgsl', 'matmul_f32.wgsl'],
  // f16-activation attention → f32
  ['attention_decode_online_f16.wgsl', 'attention_streaming.wgsl'],
  ['attention_decode_chunked_f16.wgsl', 'attention_streaming.wgsl'],
  ['attention_small_f16.wgsl', 'attention_small.wgsl'],
  ['attention_streaming_f16.wgsl', 'attention_streaming.wgsl'],
  // f16kv attention (f32 Q, f16 KV) → f32
  ['attention_decode_online_f16kv.wgsl', 'attention_streaming.wgsl'],
  ['attention_decode_chunked_f16kv.wgsl', 'attention_streaming.wgsl'],
  ['attention_small_f16kv.wgsl', 'attention_small.wgsl'],
  ['attention_streaming_f16kv.wgsl', 'attention_streaming.wgsl'],
  ['attention_head256_f16kv.wgsl', 'attention_small.wgsl'],
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
    const q4FallbackEntry = deriveQ4DecodeF32ActivationKernelEntry(entry)
      ?? deriveQ4PrefillF32ActivationKernelEntry(entry);
    if (q4FallbackEntry) {
      result.kernels[key] = q4FallbackEntry;
      continue;
    }
    const replacement = shaderMap.get(entry.kernel);
    if (replacement !== undefined) {
      result.kernels[key] = deriveMappedKernelEntry(entry, replacement);
    }
  }

  return result;
}

/**
 * Widen an f16 execution graph onto the stable f32 correctness lane used for
 * alternate-plan recovery after finiteness failure.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function widenToF32CorrectnessFallback(graph, ctx) {
  if (hasKernelFile(graph, 'fused_ffn_f16.wgsl')) {
    return null;
  }

  const hasTargetShader = Object.values(graph.kernels).some(
    (entry) => F16_TO_F32_CORRECTNESS_FALLBACK_MAP.has(entry.kernel)
  );
  if (!hasTargetShader) {
    return null;
  }

  const result = cloneGraph(graph);
  for (const [key, entry] of Object.entries(result.kernels)) {
    const q4FallbackEntry = deriveQ4DecodeF32ActivationKernelEntry(entry)
      ?? deriveQ4PrefillF32ActivationKernelEntry(entry);
    if (q4FallbackEntry) {
      result.kernels[key] = q4FallbackEntry;
      continue;
    }
    const replacement = F16_TO_F32_CORRECTNESS_FALLBACK_MAP.get(entry.kernel);
    if (replacement !== undefined) {
      result.kernels[key] = deriveMappedKernelEntry(entry, replacement);
    }
  }
  return result;
}

/**
 * Narrow f32-activation shaders back onto their f16-activation equivalents.
 *
 * Returns null if the graph has no supported f32-activation kernels to swap or
 * if the runtime did not explicitly request f16 activations on an f16-capable
 * GPU.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function narrowToF16Activations(graph, ctx) {
  if (ctx.activationDtype !== 'f16' || ctx.capabilities?.hasF16 !== true) {
    return null;
  }

  const hasTargetShader = Object.values(graph.kernels).some(
    (entry) => !hasExplicitF32ActivationContract(entry) && F32_TO_F16_ACTIVATION_MAP.has(entry.kernel)
  );
  if (!hasTargetShader) {
    return null;
  }

  const result = cloneGraph(graph);
  for (const [key, entry] of Object.entries(result.kernels)) {
    if (hasExplicitF32ActivationContract(entry)) {
      continue;
    }
    const replacement = F32_TO_F16_ACTIVATION_MAP.get(entry.kernel);
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

/**
 * Promote small-tile prefill attention onto the fixed 256-dim shared-block kernel.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useHead256SmallPrefillAttention(graph, ctx) {
  return swapPrefillAttention(graph, ctx, {
    from: 'attention_small_f16kv.wgsl',
    to: 'attention_head256_f16kv.wgsl',
  });
}

/**
 * Promote prefill attention onto the fixed 256-dim shared-block kernel.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
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
// Transform: remapQ4KDecodeAttentionToFusedQ4KGemv
// =============================================================================

/**
 * Replace fused Q4K ATTENTION-ONLY decode projection kernels with the
 * optimised fused Q4K GEMV variant (main_gemv), which combines shared-A
 * cooperative loading with fast nibble extraction for maximum M=1 throughput
 * while preserving full Q4K dequant precision (no f16 weight materialization).
 *
 * This is the production fix for the f16-precision-loss regression observed
 * when attention projections use the f16-weight GEMV path: softmax amplifies
 * the f16 round-trip error in Q/K/V projections, causing garbage output.
 * By keeping inline Q4K dequant (f32 arithmetic) the attention path stays
 * numerically correct.  FFN projections are unaffected and can safely use
 * the f16-weight GEMV path via remapQ4KDecodeFFNToGemv.
 *
 * Because the derived kernel still references fused_matmul_q4.wgsl,
 * isKernelPathFusedQ4K stays true and the weight loader remains in
 * mixed-materialization mode (Q4K retained for attention, f16 for FFN).
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function remapQ4KDecodeAttentionToFusedQ4KGemv(graph, ctx) {
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
    const newKey = deriveKernelKey(result.kernels, key, '_gemv');
    result.kernels[newKey] = deriveKernelEntry(
      result.kernels[key],
      'fused_matmul_q4.wgsl',
      'main_gemv',
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
// Transform: useQwenF16PrimaryMatmuls
// =============================================================================

/**
 * Promote the Qwen 3.5 0.8B execution graph onto its selective f16 primary
 * lane when the runtime explicitly requests f16 activations.
 *
 * This transform narrows the decode projection and LM-head path while keeping
 * `o_proj` on the stable manifest-owned f32-output kernel. The prefill path is
 * already authored in the manifest on WideTile/head256 kernels and remains
 * manifest-owned. The residual stream feeds directly into the next RMSNorm, and
 * the Qwen 3.5 0.8B promoted f16 lane becomes numerically unstable when
 * `o_proj` writes f16 there (observed first at the first full-attention block's
 * post-attention RMSNorm).
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useQwenF16PrimaryMatmuls(graph, ctx) {
  const modelId = typeof ctx.modelId === 'string' ? ctx.modelId.trim() : '';
  if (modelId !== 'qwen-3-5-0-8b-q4k-ehaf16' || ctx.activationDtype !== 'f16' || ctx.capabilities?.hasF16 !== true) {
    return null;
  }

  const layerTypes = Array.isArray(ctx.layerTypes) ? ctx.layerTypes : null;
  if (!layerTypes || layerTypes.length === 0) {
    return null;
  }

  const result = cloneGraph(graph);
  let changed = false;

  for (const [phaseName, op] of [['decode', 'attention'], ['prefill', 'attention']]) {
    const phaseSteps = result[phaseName] || [];
    const step = phaseSteps.find((entry) => Array.isArray(entry) && entry[0] === op);
    const kernelKey = step?.[1];
    const kernelEntry = kernelKey ? result.kernels[kernelKey] : null;
    const derivedEntry = deriveF16AttentionKernelEntry(kernelEntry);
    if (!step || !kernelKey || !derivedEntry) {
      continue;
    }
    const derivedKey = deriveKernelKey(result.kernels, kernelKey, '_primary_f16');
    result.kernels[derivedKey] = derivedEntry;
    const phaseResult = replacePhaseStepKernelKey(phaseSteps, op, derivedKey);
    if (phaseResult.changed) {
      result[phaseName] = phaseResult.steps;
      changed = true;
    }
  }

  const decodeProjectionStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const decodeProjectionKernel = decodeProjectionStep ? result.kernels[decodeProjectionStep[1]] : null;
  const decodeProjectionEntry = deriveQ4DecodeF16KernelEntry(decodeProjectionKernel);
  if (decodeProjectionStep && decodeProjectionEntry) {
    const decodeProjectionKey = deriveKernelKey(result.kernels, decodeProjectionStep[1], '_primary_f16');
    result.kernels[decodeProjectionKey] = decodeProjectionEntry;
    for (const op of ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']) {
      const phaseResult = replacePhaseStepKernelKey(result.decode, op, decodeProjectionKey);
      if (phaseResult.changed) {
        result.decode = phaseResult.steps;
        changed = true;
      }
    }
  }

  const prefillProjectionStep = (result.prefill || []).find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const prefillProjectionKernel = prefillProjectionStep ? result.kernels[prefillProjectionStep[1]] : null;
  const prefillProjectionEntry = deriveQ4PrefillF16KernelEntry(prefillProjectionKernel);
  if (prefillProjectionStep && prefillProjectionEntry) {
    const prefillProjectionKey = deriveKernelKey(result.kernels, prefillProjectionStep[1], '_primary_f16');
    result.kernels[prefillProjectionKey] = prefillProjectionEntry;
    for (const op of ['q_proj', 'k_proj', 'v_proj']) {
      const phaseResult = replacePhaseStepKernelKey(result.prefill, op, prefillProjectionKey);
      if (phaseResult.changed) {
        result.prefill = phaseResult.steps;
        changed = true;
      }
    }
  }

  for (const [phaseName, op] of [['decode', 'o_proj'], ['prefill', 'o_proj']]) {
    const phaseSteps = result[phaseName] || [];
    const step = phaseSteps.find((entry) => Array.isArray(entry) && entry[0] === op);
    const kernelKey = step?.[1];
    const kernelEntry = kernelKey ? result.kernels[kernelKey] : null;
    if (!step || !kernelKey || !kernelEntry) {
      continue;
    }
    const boundaryKey = deriveKernelKey(result.kernels, kernelKey, '_primary_f32_boundary');
    result.kernels[boundaryKey] = deriveKernelEntryWithPrecision(kernelEntry, {
      inputDtype: 'f32',
      outputDtype: 'f32',
    });
    const phaseResult = replacePhaseStepKernelKey(phaseSteps, op, boundaryKey);
    if (phaseResult.changed) {
      result[phaseName] = phaseResult.steps;
      changed = true;
    }
  }

  const lmHeadStep = (result.postLayer || []).find((entry) => Array.isArray(entry) && entry[0] === 'lm_head');
  const lmHeadKernel = lmHeadStep ? result.kernels[lmHeadStep[1]] : null;
  const lmHeadEntry = deriveLmHeadDecodeF16KernelEntry(lmHeadKernel);
  if (lmHeadStep && lmHeadEntry) {
    const lmHeadKey = deriveKernelKey(result.kernels, lmHeadStep[1], '_primary_f16');
    result.kernels[lmHeadKey] = lmHeadEntry;
    const phaseResult = replacePhaseStepKernelKey(result.postLayer, 'lm_head', lmHeadKey);
    if (phaseResult.changed) {
      result.postLayer = phaseResult.steps;
      changed = true;
    }
  }

  return changed ? result : null;
}

// =============================================================================
// Transform: useQwen36F16Activations
// =============================================================================

/**
 * Promote the Qwen 3.6 27B Q4K graph onto its additive all-f16 sibling lane.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useQwen36F16Activations(graph, ctx) {
  const modelId = typeof ctx.modelId === 'string' ? ctx.modelId.trim() : '';
  if (
    modelId !== 'qwen-3-6-27b-q4k-eaf16'
    || ctx.activationDtype !== 'f16'
    || ctx.mathDtype !== 'f16'
    || ctx.accumDtype !== 'f16'
    || ctx.capabilities?.hasF16 !== true
    || ctx.capabilities?.hasSubgroups !== true
  ) {
    return null;
  }

  const narrowed = narrowToF16Activations(graph, ctx);
  const result = narrowed ?? cloneGraph(graph);
  let changed = narrowed != null;

  const replaceKernelEntry = (key, entry) => {
    if (!key || !entry) {
      return;
    }
    result.kernels[key] = entry;
    changed = true;
  };
  const replaceOps = (phaseName, ops, kernelKey) => {
    for (const op of ops) {
      const phaseResult = replacePhaseStepKernelKey(result[phaseName], op, kernelKey);
      if (phaseResult.changed) {
        result[phaseName] = phaseResult.steps;
        changed = true;
      }
    }
  };

  const embedStep = findPhaseStep(result.preLayer, 'embed');
  const embedKey = embedStep?.[1] ?? null;
  const embedEntry = embedKey ? result.kernels[embedKey] : null;
  if (embedEntry?.kernel === 'gather_f16.wgsl') {
    replaceKernelEntry(
      embedKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(embedEntry, 'gather_f16_vec4_f16_out.wgsl', 'gather_vec4_f16_out'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const decodeProjectionStep = findPhaseStep(result.decode, 'q_proj');
  const decodeProjectionKey = decodeProjectionStep?.[1] ?? null;
  const decodeProjectionEntry = deriveQ4DecodeF16KernelEntry(result.kernels[decodeProjectionKey]);
  if (decodeProjectionEntry) {
    const f16Key = deriveKernelKey(result.kernels, decodeProjectionKey, '_qwen36_f16');
    result.kernels[f16Key] = decodeProjectionEntry;
    replaceOps('decode', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], f16Key);
  }

  const prefillProjectionStep = findPhaseStep(result.prefill, 'q_proj');
  const prefillProjectionKey = prefillProjectionStep?.[1] ?? null;
  const prefillProjectionEntry =
    deriveQ4PrefillF16AccumKernelEntry(result.kernels[prefillProjectionKey])
    ?? deriveQ4WideTilePrefillF16KernelEntry(result.kernels[prefillProjectionKey])
    ?? deriveQ4PrefillF16KernelEntry(result.kernels[prefillProjectionKey]);
  if (prefillProjectionEntry) {
    const f16Key = deriveKernelKey(result.kernels, prefillProjectionKey, '_qwen36_f16');
    result.kernels[f16Key] = prefillProjectionEntry;
    replaceOps('prefill', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], f16Key);
  }

  const lmHeadStep = findPhaseStep(result.postLayer, 'lm_head');
  const lmHeadKey = lmHeadStep?.[1] ?? null;
  replaceKernelEntry(
    lmHeadKey,
    deriveQ4DecodeF16KernelEntry(result.kernels[lmHeadKey])
  );

  const lmHeadPrefillStep = findPhaseStep(result.postLayer, 'lm_head_prefill');
  const lmHeadPrefillKey = lmHeadPrefillStep?.[1] ?? null;
  const lmHeadPrefillEntry = lmHeadPrefillKey ? result.kernels[lmHeadPrefillKey] : null;
  if (
    lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a.wgsl'
    || lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a_tiled.wgsl'
  ) {
    replaceKernelEntry(
      lmHeadPrefillKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(lmHeadPrefillEntry, 'matmul_f16_tiled.wgsl', 'main'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  return changed ? result : null;
}

// =============================================================================
// Transform: useGemma4Int4PleSelectiveF16Decode
// =============================================================================

/**
 * Promote only Gemma 4 E2B INT4 PLE decode Q/K/V and online attention onto
 * explicit f16 kernels. Prefill remains on manifest-owned f16kv fixed-head
 * attention because the repository does not currently have pure-f16 head256 or
 * head512 prefill kernels.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useGemma4Int4PleSelectiveF16Decode(graph, ctx) {
  const modelId = typeof ctx.modelId === 'string' ? ctx.modelId.trim() : '';
  if (
    modelId !== 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple'
    || ctx.activationDtype !== 'f16'
    || ctx.capabilities?.hasF16 !== true
    || ctx.capabilities?.hasSubgroups !== true
  ) {
    return null;
  }

  const result = cloneGraph(graph);
  let changed = false;

  const decodeProjectionStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const decodeProjectionKernel = decodeProjectionStep ? result.kernels[decodeProjectionStep[1]] : null;
  const decodeProjectionEntry = deriveDenseDecodeF16KernelEntry(decodeProjectionKernel);
  if (decodeProjectionStep && decodeProjectionEntry) {
    const decodeProjectionKey = deriveKernelKey(result.kernels, decodeProjectionStep[1], '_gemma4_f16');
    result.kernels[decodeProjectionKey] = decodeProjectionEntry;
    for (const op of ['q_proj', 'k_proj', 'v_proj']) {
      const phaseResult = replacePhaseStepKernelKey(result.decode, op, decodeProjectionKey);
      if (phaseResult.changed) {
        result.decode = phaseResult.steps;
        changed = true;
      }
    }
  }

  for (const op of ['rope_q', 'rope_k']) {
    const step = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === op);
    const kernelKey = step?.[1];
    const kernelEntry = kernelKey ? result.kernels[kernelKey] : null;
    const replacement = kernelEntry ? F32_TO_F16_ACTIVATION_MAP.get(kernelEntry.kernel) : null;
    if (!step || !kernelKey || !replacement) {
      continue;
    }
    const ropeKey = deriveKernelKey(result.kernels, kernelKey, '_gemma4_f16');
    result.kernels[ropeKey] = deriveKernelEntryWithPrecision(
      deriveKernelEntry(kernelEntry, replacement, kernelEntry.entry),
      { inputDtype: 'f16', outputDtype: 'f16' }
    );
    const phaseResult = replacePhaseStepKernelKey(result.decode, op, ropeKey);
    if (phaseResult.changed) {
      result.decode = phaseResult.steps;
      changed = true;
    }
  }

  const attentionStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'attention');
  const attentionKernel = attentionStep ? result.kernels[attentionStep[1]] : null;
  const attentionEntry = deriveF16AttentionKernelEntry(attentionKernel);
  if (attentionStep && attentionEntry) {
    const attentionKey = deriveKernelKey(result.kernels, attentionStep[1], '_gemma4_f16');
    result.kernels[attentionKey] = attentionEntry;
    const phaseResult = replacePhaseStepKernelKey(result.decode, 'attention', attentionKey);
    if (phaseResult.changed) {
      result.decode = phaseResult.steps;
      changed = true;
    }
  }

  const oProjStep = (result.decode || []).find((entry) => Array.isArray(entry) && entry[0] === 'o_proj');
  const oProjKernel = oProjStep ? result.kernels[oProjStep[1]] : null;
  if (oProjStep && oProjKernel) {
    const oProjKey = deriveKernelKey(result.kernels, oProjStep[1], '_gemma4_f32_boundary');
    result.kernels[oProjKey] = deriveKernelEntryWithPrecision(oProjKernel, {
      inputDtype: 'f32',
      outputDtype: 'f32',
    });
    const phaseResult = replacePhaseStepKernelKey(result.decode, 'o_proj', oProjKey);
    if (phaseResult.changed) {
      result.decode = phaseResult.steps;
      changed = true;
    }
  }

  return changed ? result : null;
}

// =============================================================================
// Transform: useGemma431BTextF16Activations
// =============================================================================

/**
 * Promote the Gemma 4 31B text Q4K graph onto an experimental all-f16 request
 * lane. This keeps the same Q4K weight pack but remaps execution-v1 kernels so
 * Q4 projections, attention, residual/norm/activation utilities, lm_head, and
 * sampling consume and produce f16 activations where a matching kernel exists.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {import('./execution-graph-transforms.js').ExecutionGraph | null}
 */
export function useGemma431BTextF16Activations(graph, ctx) {
  const modelId = typeof ctx.modelId === 'string' ? ctx.modelId.trim() : '';
  if (
    ![
      'gemma-4-31b-it-text-q4k-ehf16-af32',
      'gemma-4-31b-it-text-q4k-ehf16-af16',
    ].includes(modelId)
    || ctx.activationDtype !== 'f16'
    || ctx.mathDtype !== 'f16'
    || ctx.accumDtype !== 'f16'
    || ctx.capabilities?.hasF16 !== true
    || ctx.capabilities?.hasSubgroups !== true
  ) {
    return null;
  }

  const narrowed = narrowToF16Activations(graph, ctx);
  const result = narrowed ?? cloneGraph(graph);
  let changed = narrowed != null;

  const replaceKernelEntry = (key, entry) => {
    if (!key || !entry) {
      return;
    }
    result.kernels[key] = entry;
    changed = true;
  };

  const embedStep = findPhaseStep(result.preLayer, 'embed');
  const embedKey = embedStep?.[1] ?? null;
  const embedEntry = embedKey ? result.kernels[embedKey] : null;
  if (embedEntry?.kernel === 'gather_f16.wgsl') {
    replaceKernelEntry(
      embedKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(embedEntry, 'gather_f16_vec4_f16_out.wgsl', 'gather_vec4_f16_out'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const decodeProjectionStep = findPhaseStep(result.decode, 'q_proj');
  const decodeProjectionKey = decodeProjectionStep?.[1] ?? null;
  replaceKernelEntry(
    decodeProjectionKey,
    deriveQ4DecodeF16KernelEntry(result.kernels[decodeProjectionKey])
  );

  const prefillProjectionStep = findPhaseStep(result.prefill, 'q_proj');
  const prefillProjectionKey = prefillProjectionStep?.[1] ?? null;
  replaceKernelEntry(
    prefillProjectionKey,
    deriveQ4PrefillF16AccumKernelEntry(result.kernels[prefillProjectionKey])
      ?? deriveQ4WideTilePrefillF16KernelEntry(result.kernels[prefillProjectionKey])
      ?? deriveQ4PrefillF16KernelEntry(result.kernels[prefillProjectionKey])
  );

  const replacePrefillAttentionEntries = (entries) => {
    for (const entry of entries || []) {
      if (Array.isArray(entry)) {
        if (entry[0] !== 'attention') {
          continue;
        }
        replaceKernelEntry(
          entry[1],
          deriveF16AttentionKernelEntry(result.kernels[entry[1]])
        );
        continue;
      }
      if (entry && typeof entry === 'object' && Array.isArray(entry.steps)) {
        replacePrefillAttentionEntries(entry.steps);
      }
    }
  };
  replacePrefillAttentionEntries(result.prefill);

  const finalNormStep = findPhaseStep(result.postLayer, 'final_norm');
  const finalNormKey = finalNormStep?.[1] ?? null;
  const finalNormEntry = finalNormKey ? result.kernels[finalNormKey] : null;
  if (finalNormEntry?.kernel === 'rmsnorm.wgsl') {
    replaceKernelEntry(
      finalNormKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(finalNormEntry, 'rmsnorm_f16.wgsl', finalNormEntry.entry),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
  }

  const lmHeadStep = findPhaseStep(result.postLayer, 'lm_head');
  const lmHeadKey = lmHeadStep?.[1] ?? null;
  replaceKernelEntry(
    lmHeadKey,
    deriveLmHeadDecodeF16KernelEntry(result.kernels[lmHeadKey])
  );

  const lmHeadPrefillStep = findPhaseStep(result.postLayer, 'lm_head_prefill');
  const lmHeadPrefillKey = lmHeadPrefillStep?.[1] ?? null;
  const lmHeadPrefillEntry = lmHeadPrefillKey ? result.kernels[lmHeadPrefillKey] : null;
  if (
    lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a.wgsl'
    || lmHeadPrefillEntry?.kernel === 'matmul_f16w_f32a_tiled.wgsl'
  ) {
    replaceKernelEntry(
      lmHeadPrefillKey,
      deriveKernelEntryWithPrecision(
        deriveKernelEntry(lmHeadPrefillEntry, 'matmul_f16_tiled.wgsl', 'main'),
        { inputDtype: 'f16', outputDtype: 'f16' }
      )
    );
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

// Session-only capability transform marker. The execution-v1 compiler applies
// the matching runtime session patch; the graph itself is intentionally stable.
export function disableRetainQ4KMaterialization() {
  return null;
}

/**
 * Fail-closed sentinel transform. A capability rule installs this when the
 * matched (modelId, runtime profile) combination is contradictory — for
 * example, an af32 manifest variant paired with a runtime profile that
 * demands f16 activations. The matcher reaches this transform only when the
 * earlier manifest-binding gate has been bypassed; throwing here keeps the
 * lane-confusion door shut at the capability layer too.
 *
 * @param {import('./execution-graph-transforms.js').ExecutionGraph} _graph
 * @param {import('./execution-graph-transforms.js').TransformContext} ctx
 * @returns {never}
 */
export function failClosedLaneMismatch(_graph, ctx) {
  const modelId = ctx?.modelId ?? 'unknown';
  const activationDtype = ctx?.activationDtype ?? 'unknown';
  throw new Error(
    `Capability resolver: lane mismatch for "${modelId}" (activationDtype=${activationDtype}). ` +
    'The manifest variant tag is the lane identity — load the manifest variant ' +
    'whose compute lane matches the runtime profile.'
  );
}

// =============================================================================
// Registry
// =============================================================================

/** @type {Readonly<Record<string, Function>>} */
export const TRANSFORMS = Object.freeze({
  narrowToF16Activations,
  removeSubgroups,
  widenToF32Activations,
  widenToF32CorrectnessFallback,
  swapPrefillAttention,
  useHead256SmallPrefillAttention,
  useHead256PrefillAttention,
  widenProjectionWeightsToF32,
  remapDenseQ4KPrefillToQ4Native,
  remapQ4KPrefillToDense,
  useLinearDecodeProjectionF16,
  remapQ4KDecodeToGemv,
  remapQ4KDecodeAttentionToGemv,
  remapQ4KDecodeAttentionToFusedQ4KGemv,
  remapQ4KDecodeFFNToGemv,
  disableRetainQ4KMaterialization,
  useQwenF16PrimaryMatmuls,
  useQwen36F16Activations,
  useQwenDecodeF16Matmuls,
  useGemma4Int4PleSelectiveF16Decode,
  useGemma431BTextF16Activations,
  failClosedLaneMismatch,
  composeTransforms,
});
