
import { F16_TO_F32_ACTIVATION_MAP, F32_TO_F16_ACTIVATION_MAP, KERNEL_FILE_PRECISION_PATCHES, cloneGraph, deriveF16AttentionKernelEntry, deriveKernelEntry, deriveKernelEntryWithPrecision, deriveKernelKey, deriveLinearDecodeF16KernelEntry, deriveLmHeadDecodeF16KernelEntry, deriveQ4DecodeF16KernelEntry, deriveQ4PrefillF16AccumKernelEntry, deriveQ4PrefillF16KernelEntry, deriveQ4WideTilePrefillF16KernelEntry, findPhaseStep, narrowToF16Activations, replacePhaseStepKernelKey, useQwen36F16Activations, useQwenDecodeF16Matmuls, useQwenF16PrimaryMatmuls } from './execution-graph/dtype.js';
import { LAYER_PROJECTION_OPS, useGemma4Int4PleAf16Activations, useGemma4Int4PleSelectiveF16Decode, useGemma4TextF16ActivationsForLane } from './execution-graph/fusion.js';
import { deriveMappedKernelEntry, deriveQ4DecodeF32ActivationKernelEntry, deriveQ4PrefillF32ActivationKernelEntry, findKernelKeyByFileAndEntry, hasKernelFile, remapDenseQ4KPrefillToQ4Native, remapStepKeys, removeSubgroups, widenProjectionWeightsToF32, widenToF32Activations } from './execution-graph/validation.js';
export { remapDenseQ4KPrefillToQ4Native, removeSubgroups, widenProjectionWeightsToF32, widenToF32Activations } from './execution-graph/validation.js';
export { useGemma4Int4PleAf16Activations, useGemma4Int4PleSelectiveF16Decode } from './execution-graph/fusion.js';
export { narrowToF16Activations, useQwen36F16Activations, useQwenDecodeF16Matmuls, useQwenF16PrimaryMatmuls } from './execution-graph/dtype.js';// =============================================================================
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

/*
 * Deep-clone an execution graph.
 */

/*
 * Shader files that require subgroups even though "subgroup" is not in the filename.
 * Online attention kernels use subgroup reductions internally.
 */

export function getKernelFilePrecisionPatch(kernel) {
  return KERNEL_FILE_PRECISION_PATCHES.get(kernel) ?? null;
}

/*
 * Check whether a kernel entry requires subgroup support.
 */

/*
 * Find all kernel keys in the graph whose `kernel` file matches the given filename.
 */

/*
 * Check whether any kernel in the graph uses the given shader file.
 */

/*
 * Create a new kernel entry with the digest cleared (shader changed).
 */

/*
 * Derive a non-colliding kernel key name.
 */

/*
 * Replace kernel key references in step tuples.
 */

/*
 * Check whether a step tuple's kernel key resolves to the given shader file.
 */
function stepUsesFile(graph, step, filename) {
  const kernelKey = step[1];
  const entry = graph.kernels[kernelKey];
  return entry != null && entry.kernel === filename;
}

/*
 * Find the first kernel key used by matching ops in a phase whose shader file
 * satisfies the provided predicate.
 */

/*
 * Find an existing kernel key by shader file and entry point.
 */

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

/*
 * Remove subgroup shader dependencies from decode and postLayer steps.
 * Prefill steps are left untouched (they already use tiled matmul).
 *
 * Returns null if the graph has no subgroup kernels.
 *
 */

// =============================================================================
// Transform: widenToF32Activations
// =============================================================================

/*
 * Activation-only widening: f16-activation shaders → f32-activation variants
 * that still use f16 for weights and KV cache. Requires shader-f16 for weight
 * and KV buffer reads.
 */

export function resolveF16ToF32ActivationKernel(kernel) {
  return F16_TO_F32_ACTIVATION_MAP.get(kernel) ?? null;
}

/*
 * Activation-only narrowing: f32-activation shaders that still consume f16
 * weights/KV are rewritten onto the matching f16-activation lane.
 *
 * This is the inverse of `F16_TO_F32_ACTIVATION_MAP` and is used when a
 * runtime session explicitly requests f16 activations for an execution-v1
 * graph that was authored with conservative f32 activation defaults.
 */

F32_TO_F16_ACTIVATION_MAP.set('gather.wgsl', 'gather_f16.wgsl');
F32_TO_F16_ACTIVATION_MAP.set('attention_head256_f16kv.wgsl', 'attention_small_f16.wgsl');
// head512 pure-f16 prefill is model-scoped below; keep generic Gemma 4 E2B
// f16 requests fail-closed until that path has its own evidence.
F32_TO_F16_ACTIVATION_MAP.delete('attention_head512_f16kv.wgsl');

/*
 * Correctness fallback: preserve f16 weights where possible, but widen both
 * activations and KV-cache interactions onto the stable f32 execution lane.
 * Used for alternate-plan recovery after finiteness failure.
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

/*
 * Full f32 widening: every shader that uses `enable f16;` is replaced with a
 * pure-f32 equivalent. Used when the GPU cannot compile any f16 WGSL at all.
 * Covers f16-activation, f16-weight (f16w), and f16-KV (f16kv) kernels.
 */

/*
 * Widen all f16-activation shaders to f32-activation equivalents.
 *
 * Returns null if the graph contains fused_ffn_f16.wgsl (no direct f32
 * equivalent exists) or if no f16 activation shaders are present.
 *
 * NOTE: The caller is responsible for also updating session.activationDtype
 * to reflect the widened dtype.
 *
 */

/*
 * Widen an f16 execution graph onto the stable f32 correctness lane used for
 * alternate-plan recovery after finiteness failure.
 *
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

/*
 * Narrow f32-activation shaders back onto their f16-activation equivalents.
 *
 * Returns null if the graph has no supported f32-activation kernels to swap or
 * if the runtime did not explicitly request f16 activations on an f16-capable
 * GPU.
 *
 */

// =============================================================================
// Transform: swapPrefillAttention
// =============================================================================

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

/*
 * Known layer projection ops. Only these are widened; lm_head and embed are
 * excluded.
 */

function resolveDensePrefillProjectionKernel(ctx) {
  return ctx.activationDtype === 'f16'
    ? 'matmul_f16.wgsl'
    : 'matmul_f16w_f32a.wgsl';
}

/*
 * Replace projection matmul kernels with f32 weight variants.
 *
 * Applies only to layer projection steps (q/k/v/o/gate/up/down), NOT lm_head
 * or embed.
 *
 * Returns null if no applicable projection kernels are found.
 *
 */

// =============================================================================
// Transform: remapDenseQ4KPrefillToQ4Native
// =============================================================================

/*
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
 */

// =============================================================================
// Transform: remapQ4KPrefillToDense
// =============================================================================

/*
 * Replace fused Q4K prefill projection kernels with dense tiled variants.
 *
 * Decode remains unchanged so the runtime can keep using fused Q4K decode while
 * the loader exposes mixed dense+Q4K materializations for prefill.
 *
 * Returns null when the graph has no fused Q4K prefill projection kernels.
 *
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

/*
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

/*
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

/*
 * Replace fused Q4K ATTENTION-ONLY decode projection kernels with GEMV
 * subgroup variants, leaving FFN projections (gate/up/down_proj) untouched.
 *
 * Diagnostic transform for isolating whether the GEMV correctness regression
 * originates in the attention or FFN projection path.  Because FFN ops keep
 * their fused Q4K kernels, `isKernelPathFusedQ4K` stays true and the weight
 * loader remains in mixed-materialization mode.
 *
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

/*
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

/*
 * Replace fused Q4K FFN-ONLY decode projection kernels with GEMV subgroup
 * variants, leaving attention projections (q/k/v/o_proj) as fused Q4K.
 *
 * Diagnostic complement to `remapQ4KDecodeAttentionToGemv`.  Together these
 * two transforms isolate whether the GEMV decode regression originates in
 * the attention or FFN projection path.  Because attention ops keep their
 * fused Q4K kernels, `isKernelPathFusedQ4K` stays true and the weight loader
 * remains in mixed-materialization mode.
 *
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

/*
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
 */

// =============================================================================
// Transform: useQwenF16PrimaryMatmuls
// =============================================================================

/*
 * Select the Qwen 3.5 small-model execution graph's selective f16 probe lane
 * when the runtime explicitly requests f16 activations.
 *
 * This transform narrows the decode projection and LM-head path while keeping
 * `o_proj` on the stable manifest-owned f32-output kernel. The prefill path is
 * already authored in the manifest on WideTile/head256 kernels and remains
 * manifest-owned. The residual stream feeds directly into the next RMSNorm, and
 * the Qwen 3.5 0.8B promoted f16 lane becomes numerically unstable when
 * `o_proj` writes f16 there (observed first at the first full-attention block's
 * post-attention RMSNorm).
 *
 */

// =============================================================================
// Transform: useQwen36F16Activations
// =============================================================================

/*
 * Promote the Qwen 3.6 27B Q4K graph onto its additive all-f16 sibling lane.
 *
 */

// =============================================================================
// Transform: useGemma4Int4PleSelectiveF16Decode
// =============================================================================

/*
 * Promote only Gemma 4 E2B INT4 PLE decode Q/K/V and online attention onto
 * explicit f16 kernels. Prefill remains on manifest-owned f16kv fixed-head
 * attention because the repository does not currently have pure-f16 head256 or
 * head512 prefill kernels.
 *
 */

// =============================================================================
// Transform: useGemma4TextF16Activations
// =============================================================================

export function useGemma4TextF16Activations(graph, ctx) {
  return useGemma4TextF16ActivationsForLane(graph, ctx, { stableTextBoundary: false });
}

export function useGemma412BTextF16Activations(graph, ctx) {
  return useGemma4TextF16ActivationsForLane(graph, ctx, { stableTextBoundary: true });
}

export function useGemma431BTextF16Activations(graph, ctx) {
  return useGemma4TextF16Activations(graph, ctx);
}

// =============================================================================
// Transform: useGemma4Int4PleAf16Activations
// =============================================================================

/*
 * Promote the Gemma 4 E2B INT4-PLE Q4K graph onto the all-f16 lane via the
 * weights-ref sibling manifest gemma-4-e2b-it-q4k-ehf16-af16-int4ple. Mirrors
 * useGemma4TextF16Activations: same Q4 weight pack, kernels narrowed to f16
 * activations, prefill projections promoted from widetile to widetile_f16a,
 * decode projections to multicol_f16a, lm_head/sample/final_norm to their f16
 * counterparts. Apple Metal stays disabled at the capability layer because the
 * fused-q4k+f16 kernel pool produces NaN at L0.ffn_down on metal-3.
 *
 */

// =============================================================================
// Composition
// =============================================================================

/*
 * Compose multiple transforms into a single transform function.
 *
 * Each transform is applied sequentially. If a transform returns null
 * (not applicable), the graph passes through unchanged.
 *
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

/*
 * Fail-closed sentinel transform. A capability rule installs this when the
 * matched (modelId, runtime profile) combination is contradictory — for
 * example, an af32 manifest variant paired with a runtime profile that
 * demands f16 activations. The matcher reaches this transform only when the
 * earlier manifest-binding gate has been bypassed; throwing here keeps the
 * lane-confusion door shut at the capability layer too.
 *
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
  useGemma4TextF16Activations,
  useGemma412BTextF16Activations,
  useGemma431BTextF16Activations,
  useGemma4Int4PleAf16Activations,
  failClosedLaneMismatch,
  composeTransforms,
});
