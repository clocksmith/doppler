import { F16_TO_F32_ACTIVATION_MAP, F32_TO_F16_ACTIVATION_MAP, KERNEL_FILE_PRECISION_PATCHES, cloneGraph, deriveF16AttentionKernelEntry, deriveKernelEntry, deriveKernelEntryWithPrecision, deriveKernelKey, deriveLinearDecodeF16KernelEntry, deriveLmHeadDecodeF16KernelEntry, deriveQ4DecodeF16KernelEntry, deriveQ4PrefillF16AccumKernelEntry, deriveQ4PrefillF16KernelEntry, deriveQ4WideTilePrefillF16KernelEntry, findPhaseStep, narrowToF16Activations, replacePhaseStepKernelKey, useQwen36F16Activations, useQwenDecodeF16Matmuls, useQwenF16PrimaryMatmuls } from './dtype.js';
import { LAYER_PROJECTION_OPS, useGemma4Int4PleAf16Activations, useGemma4Int4PleSelectiveF16Decode, useGemma4TextF16ActivationsForLane } from './fusion.js';

export const SUBGROUP_REQUIRING_FILES = new Set([
  'attention_decode_online_f16kv.wgsl',
  'attention_decode_online_f16.wgsl',
]);

export function isSubgroupKernel(kernelEntry) {
  if (typeof kernelEntry.kernel !== 'string') return false;
  return kernelEntry.kernel.includes('subgroup') || SUBGROUP_REQUIRING_FILES.has(kernelEntry.kernel);
}

export function requiresNoSubgroupFallback(kernelEntry) {
  if (typeof kernelEntry?.kernel !== 'string') return false;
  return isSubgroupKernel(kernelEntry) || kernelEntry.kernel.startsWith('fused_matmul_q4');
}

export function findKernelKeysByFile(graph, filename) {
  const keys = [];
  for (const [key, entry] of Object.entries(graph.kernels)) {
    if (entry.kernel === filename) {
      keys.push(key);
    }
  }
  return keys;
}

export function hasKernelFile(graph, filename) {
  return findKernelKeysByFile(graph, filename).length > 0;
}

export function deriveMappedKernelEntry(base, newFile) {
  const newEntry = newFile === 'matmul_f32.wgsl' ? 'main' : base.entry;
  return deriveKernelEntry(base, newFile, newEntry);
}

export function remapStepKeys(steps, keyMap) {
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

export function findPhaseKernelKey(graph, steps, ops, predicate) {
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

export function findKernelKeyByFileAndEntry(graph, filename, entryPoint) {
  for (const [key, entry] of Object.entries(graph.kernels)) {
    if (entry.kernel === filename && entry.entry === entryPoint) {
      return key;
    }
  }
  return null;
}

export function deriveQ4DecodeF32ActivationKernelEntry(base) {
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

export function deriveQ4PrefillF32ActivationKernelEntry(base, options = {}) {
  if (typeof base?.kernel !== 'string') {
    return null;
  }
  const replacement = options.fullF32 === true
    ? {
      kernel: 'fused_matmul_q4_batched_multicol_shared.wgsl',
      entry: 'main',
    }
    : {
      kernel: 'fused_matmul_q4_widetile.wgsl',
      entry: 'main',
    };
  if (base.kernel === 'fused_matmul_q4_widetile.wgsl' && options.fullF32 === true) {
    return deriveKernelEntryWithPrecision(
      deriveKernelEntry(base, replacement.kernel, replacement.entry, null),
      { inputDtype: 'f32', outputDtype: 'f32' }
    );
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
      deriveKernelEntry(base, replacement.kernel, replacement.entry, null),
      { inputDtype: 'f32', outputDtype: 'f32' }
    );
  }
  return deriveKernelEntryWithPrecision(
    deriveKernelEntry(base, 'fused_matmul_q4_batched.wgsl', 'main_batched', null),
    { inputDtype: 'f32', outputDtype: 'f32' }
  );
}

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

export const FULL_F32_SHADER_MAP = new Map([
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

export function widenToF32Activations(graph, ctx) {
  // Bail out if fused f16 FFN is present — no direct f32 equivalent
  if (hasKernelFile(graph, 'fused_ffn_f16.wgsl')) {
    return null;
  }

  // When the GPU cannot compile any f16 WGSL (hasF16=false), use the full f32
  // map that covers f16-weight and f16-KV kernels. An f32 KV request must preserve
  // f16 weights; capability policy uses widenToF32CorrectnessFallback for that case.
  const shaderMap = ctx.capabilities?.hasF16 === false
    ? FULL_F32_SHADER_MAP
    : F16_TO_F32_ACTIVATION_MAP;
  const fullF32 = ctx.capabilities?.hasF16 === false;

  const hasTargetShader = Object.values(graph.kernels).some(
    (entry) => shaderMap.has(entry.kernel)
      || deriveQ4DecodeF32ActivationKernelEntry(entry)
      || deriveQ4PrefillF32ActivationKernelEntry(entry, { fullF32 })
  );
  if (!hasTargetShader) {
    return null;
  }

  const result = cloneGraph(graph);

  for (const [key, entry] of Object.entries(result.kernels)) {
    const q4FallbackEntry = deriveQ4DecodeF32ActivationKernelEntry(entry)
      ?? deriveQ4PrefillF32ActivationKernelEntry(entry, { fullF32 });
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

export const PROJECTION_MATMUL_FILES = new Set([
  'matmul_gemv_subgroup.wgsl',
  'matmul_gemv_subgroup_f16a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16w_f32a.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);

export const DENSE_Q4_PREFILL_FILES = new Set([
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);

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
