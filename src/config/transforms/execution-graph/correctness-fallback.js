import { cloneGraph } from './dtype.js';
import { ATTENTION_F16KV_TO_F32_ENTRIES, deriveMappedKernelEntry, deriveQ4DecodeF32ActivationKernelEntry, deriveQ4PrefillF32ActivationKernelEntry, hasKernelFile } from './validation.js';

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
  ...ATTENTION_F16KV_TO_F32_ENTRIES,
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
/** @type {typeof import('../execution-graph-transforms.js').widenToF32CorrectnessFallback} */
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
