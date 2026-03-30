import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  removeSubgroups,
  widenToF32Activations,
  swapPrefillAttention,
  useHead256PrefillAttention,
  widenProjectionWeightsToF32,
  remapDenseQ4KPrefillToQ4Native,
  remapQ4KPrefillToDense,
  useLinearDecodeProjectionF16,
  remapQ4KDecodeToGemv,
  useQwenDecodeF16Matmuls,
  composeTransforms,
} from '../../src/config/transforms/execution-graph-transforms.js';

import {
  resolveCapabilityTransforms,
  resolveFinitenessFallbackTransform,
} from '../../src/config/transforms/capability-transform-resolver.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Load fixtures
// ---------------------------------------------------------------------------

const conversionConfig = JSON.parse(
  readFileSync(
    path.resolve(__dirname, '../../src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json'),
    'utf-8'
  )
);

const qwenConversionConfig = JSON.parse(
  readFileSync(
    path.resolve(__dirname, '../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json'),
    'utf-8'
  )
);

/** The real execution graph from the gemma3-1b-q4k conversion config. */
const REAL_GRAPH = conversionConfig.execution;
const QWEN_REAL_GRAPH = qwenConversionConfig.execution;

/** Default transform context matching the conversion config. */
const CTX_F32 = { activationDtype: 'f32', kvDtype: 'f16' };
const CTX_F16 = { activationDtype: 'f16', kvDtype: 'f16' };

let assertions = 0;

function ok(value, message) {
  assert.ok(value, message);
  assertions++;
}

function equal(actual, expected, message) {
  assert.equal(actual, expected, message);
  assertions++;
}

function deepEqual(actual, expected, message) {
  assert.deepStrictEqual(actual, expected, message);
  assertions++;
}

function notEqual(actual, expected, message) {
  assert.notEqual(actual, expected, message);
  assertions++;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect all kernel filenames referenced in a graph phase's step tuples. */
function collectKernelFilesForPhase(graph, phase) {
  const steps = graph[phase];
  if (!Array.isArray(steps)) return new Set();
  const files = new Set();
  const appendFiles = (stepList) => {
    for (const step of stepList) {
      const kernelKey = step[1];
      const entry = graph.kernels[kernelKey];
      if (entry) {
        files.add(entry.kernel);
      }
    }
  };
  for (const step of steps) {
    if (Array.isArray(step)) {
      appendFiles([step]);
      continue;
    }
    if (step && typeof step === 'object' && Array.isArray(step.steps)) {
      appendFiles(step.steps);
    }
  }
  return files;
}

/** Build a minimal f16 activation graph with subgroup kernels for testing. */
function buildF16SubgroupGraph() {
  return {
    kernels: {
      embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: 'sha256:aaa' },
      rmsnorm: { kernel: 'rmsnorm_f16.wgsl', entry: 'main', digest: 'sha256:bbb' },
      gemv: { kernel: 'matmul_gemv_subgroup_f16a.wgsl', entry: 'main_vec4', digest: 'sha256:ccc' },
      rope: { kernel: 'rope_f16.wgsl', entry: 'main', digest: 'sha256:ddd' },
      attn_decode: { kernel: 'attention_decode_online_f16.wgsl', entry: 'main', digest: 'sha256:eee' },
      residual: { kernel: 'residual_f16.wgsl', entry: 'main', digest: 'sha256:fff' },
      gelu: { kernel: 'gelu_f16.wgsl', entry: 'main', digest: 'sha256:ggg', constants: { HAS_GATE: true } },
      tiled: { kernel: 'matmul_f16.wgsl', entry: 'main', digest: 'sha256:hhh' },
      attn_stream: { kernel: 'attention_streaming_f16.wgsl', entry: 'main', digest: 'sha256:iii' },
      lm_head_gemv: {
        kernel: 'matmul_gemv_subgroup_f16a.wgsl', entry: 'main_multicol', digest: 'sha256:jjj',
        constants: { MULTICOL_COLS_PER_WG: 64, MULTICOL_THREADS_PER_COL: 4 },
      },
      sample: { kernel: 'sample_f16.wgsl', entry: 'sample_single_pass', digest: 'sha256:kkk' },
    },
    preLayer: [['embed', 'embed', 'embed_tokens']],
    decode: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj'],
      ['k_proj', 'gemv', 'layer.{L}.self_attn.k_proj'],
      ['v_proj', 'gemv', 'layer.{L}.self_attn.v_proj'],
      ['rope_q', 'rope'],
      ['rope_k', 'rope'],
      ['attention', 'attn_decode'],
      ['o_proj', 'gemv', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['post_attn_norm', 'rmsnorm'],
      ['gate_proj', 'gemv', 'layer.{L}.mlp.gate_proj'],
      ['up_proj', 'gemv', 'layer.{L}.mlp.up_proj'],
      ['activation', 'gelu'],
      ['down_proj', 'gemv', 'layer.{L}.mlp.down_proj'],
      ['ffn_residual', 'residual'],
    ],
    prefill: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'tiled', 'layer.{L}.self_attn.q_proj'],
      ['k_proj', 'tiled', 'layer.{L}.self_attn.k_proj'],
      ['v_proj', 'tiled', 'layer.{L}.self_attn.v_proj'],
      ['rope_q', 'rope'],
      ['rope_k', 'rope'],
      ['attention', 'attn_stream'],
      ['o_proj', 'tiled', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['post_attn_norm', 'rmsnorm'],
      ['gate_proj', 'tiled', 'layer.{L}.mlp.gate_proj'],
      ['up_proj', 'tiled', 'layer.{L}.mlp.up_proj'],
      ['activation', 'gelu'],
      ['down_proj', 'tiled', 'layer.{L}.mlp.down_proj'],
      ['ffn_residual', 'residual'],
    ],
    postLayer: [
      ['final_norm', 'rmsnorm'],
      ['lm_head', 'lm_head_gemv', 'lm_head'],
      ['lm_head_prefill', 'tiled', 'lm_head'],
      ['sample', 'sample'],
    ],
    policies: { unsupportedPrecision: 'error', dtypeTransition: 'require_cast_step', unresolvedKernel: 'error' },
  };
}

/** Build a graph that already uses only tiled kernels (no subgroups). */
function buildNoSubgroupGraph() {
  return {
    kernels: {
      embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: 'sha256:aaa' },
      rmsnorm: { kernel: 'rmsnorm.wgsl', entry: 'main', digest: 'sha256:bbb' },
      tiled: { kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main', digest: 'sha256:ccc' },
      rope: { kernel: 'rope.wgsl', entry: 'main', digest: 'sha256:ddd' },
      attn_decode: { kernel: 'attention_decode_chunked_f16kv.wgsl', entry: 'main', digest: 'sha256:eee' },
      residual: { kernel: 'residual.wgsl', entry: 'main', digest: 'sha256:fff' },
      gelu: { kernel: 'gelu.wgsl', entry: 'main', digest: 'sha256:ggg', constants: { HAS_GATE: true } },
      attn_stream: { kernel: 'attention_streaming_f16kv.wgsl', entry: 'main', digest: 'sha256:hhh' },
      lm_head: { kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', digest: 'sha256:iii' },
      sample: { kernel: 'sample.wgsl', entry: 'sample_single_pass', digest: 'sha256:jjj' },
    },
    preLayer: [['embed', 'embed', 'embed_tokens']],
    decode: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'tiled', 'layer.{L}.self_attn.q_proj'],
      ['attention', 'attn_decode'],
      ['o_proj', 'tiled', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
    ],
    prefill: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'tiled', 'layer.{L}.self_attn.q_proj'],
      ['attention', 'attn_stream'],
      ['o_proj', 'tiled', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
    ],
    postLayer: [
      ['final_norm', 'rmsnorm'],
      ['lm_head', 'lm_head', 'lm_head'],
      ['sample', 'sample'],
    ],
    policies: { unsupportedPrecision: 'error' },
  };
}

/** Build a minimal f16 activation graph (no subgroup kernels). */
function buildF16ActivationGraph() {
  return {
    kernels: {
      rmsnorm: { kernel: 'rmsnorm_f16.wgsl', entry: 'main', digest: 'sha256:aaa' },
      matmul: { kernel: 'matmul_f16.wgsl', entry: 'main', digest: 'sha256:bbb' },
      rope: { kernel: 'rope_f16.wgsl', entry: 'main', digest: 'sha256:ccc' },
      attn_decode: { kernel: 'attention_decode_chunked_f16.wgsl', entry: 'main', digest: 'sha256:ddd' },
      residual: { kernel: 'residual_f16.wgsl', entry: 'main', digest: 'sha256:eee' },
      gelu: { kernel: 'gelu_f16.wgsl', entry: 'main', digest: 'sha256:fff' },
      attn_stream: { kernel: 'attention_streaming_f16.wgsl', entry: 'main', digest: 'sha256:ggg' },
      sample: { kernel: 'sample_f16.wgsl', entry: 'sample_single_pass', digest: 'sha256:hhh' },
    },
    decode: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'matmul', 'layer.{L}.self_attn.q_proj'],
      ['rope_q', 'rope'],
      ['attention', 'attn_decode'],
      ['o_proj', 'matmul', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['activation', 'gelu'],
    ],
    prefill: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'matmul', 'layer.{L}.self_attn.q_proj'],
      ['rope_q', 'rope'],
      ['attention', 'attn_stream'],
      ['o_proj', 'matmul', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['activation', 'gelu'],
    ],
    postLayer: [
      ['sample', 'sample'],
    ],
    policies: { unsupportedPrecision: 'error' },
  };
}

/** Build a graph with fused_ffn_f16.wgsl. */
function buildFusedF16Graph() {
  return {
    kernels: {
      rmsnorm: { kernel: 'rmsnorm_f16.wgsl', entry: 'main', digest: 'sha256:aaa' },
      fused_ffn: { kernel: 'fused_ffn_f16.wgsl', entry: 'main', digest: 'sha256:bbb' },
    },
    decode: [
      ['input_norm', 'rmsnorm'],
      ['ffn', 'fused_ffn'],
    ],
    prefill: [],
    postLayer: [],
    policies: { unsupportedPrecision: 'error' },
  };
}

/** Build a graph with f16w matmul kernels in projections and lm_head/embed. */
function buildF16WeightProjectionGraph() {
  return {
    kernels: {
      embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: 'sha256:aaa' },
      rmsnorm: { kernel: 'rmsnorm.wgsl', entry: 'main', digest: 'sha256:bbb' },
      proj_matmul: { kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main', digest: 'sha256:ccc' },
      rope: { kernel: 'rope.wgsl', entry: 'main', digest: 'sha256:ddd' },
      residual: { kernel: 'residual.wgsl', entry: 'main', digest: 'sha256:eee' },
      lm_head_kernel: { kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', digest: 'sha256:fff' },
      sample: { kernel: 'sample.wgsl', entry: 'sample_single_pass', digest: 'sha256:ggg' },
    },
    preLayer: [['embed', 'embed', 'embed_tokens']],
    decode: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'proj_matmul', 'layer.{L}.self_attn.q_proj'],
      ['k_proj', 'proj_matmul', 'layer.{L}.self_attn.k_proj'],
      ['v_proj', 'proj_matmul', 'layer.{L}.self_attn.v_proj'],
      ['rope_q', 'rope'],
      ['o_proj', 'proj_matmul', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['gate_proj', 'proj_matmul', 'layer.{L}.mlp.gate_proj'],
      ['up_proj', 'proj_matmul', 'layer.{L}.mlp.up_proj'],
      ['down_proj', 'proj_matmul', 'layer.{L}.mlp.down_proj'],
    ],
    prefill: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'proj_matmul', 'layer.{L}.self_attn.q_proj'],
      ['k_proj', 'proj_matmul', 'layer.{L}.self_attn.k_proj'],
      ['v_proj', 'proj_matmul', 'layer.{L}.self_attn.v_proj'],
      ['rope_q', 'rope'],
      ['o_proj', 'proj_matmul', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['gate_proj', 'proj_matmul', 'layer.{L}.mlp.gate_proj'],
      ['up_proj', 'proj_matmul', 'layer.{L}.mlp.up_proj'],
      ['down_proj', 'proj_matmul', 'layer.{L}.mlp.down_proj'],
    ],
    postLayer: [
      ['final_norm', 'rmsnorm'],
      ['lm_head', 'lm_head_kernel', 'lm_head'],
      ['sample', 'sample'],
    ],
    policies: { unsupportedPrecision: 'error' },
  };
}

// ===========================================================================
// Test 1: removeSubgroups transform
// ===========================================================================
{
  const input = structuredClone(REAL_GRAPH);
  const frozen = structuredClone(input);
  const result = removeSubgroups(input, CTX_F32);

  // Must return a new graph (not null)
  ok(result !== null, 'removeSubgroups should return a non-null result for a graph with subgroup kernels');
  ok(result !== input, 'removeSubgroups should return a new object');

  // No decode kernel should reference matmul subgroup shaders
  const decodeFiles = collectKernelFilesForPhase(result, 'decode');
  ok(!decodeFiles.has('matmul_gemv_subgroup.wgsl'), 'decode should not have matmul_gemv_subgroup.wgsl');

  // Decode projections should now reference tiled kernels
  ok(decodeFiles.has('matmul_f16w_f32a_tiled.wgsl'), 'decode projections should use matmul_f16w_f32a_tiled.wgsl');

  // Online attention kernel should be swapped to chunked (no subgroups needed)
  ok(!decodeFiles.has('attention_decode_online_f16kv.wgsl'), 'decode should not have attention_decode_online_f16kv.wgsl');
  ok(decodeFiles.has('attention_decode_chunked_f16kv.wgsl'), 'decode should have attention_decode_chunked_f16kv.wgsl');

  const prefillFiles = collectKernelFilesForPhase(result, 'prefill');
  ok(prefillFiles.has('matmul_f16w_f32a_tiled.wgsl'),
    'prefill projections should use matmul_f16w_f32a_tiled.wgsl after removeSubgroups');

  // Original graph must not be mutated (purity check)
  deepEqual(input, frozen, 'removeSubgroups must not mutate the input graph');
}

// ===========================================================================
// Test 2: removeSubgroups is no-op when no subgroup kernels present
// ===========================================================================
{
  const graph = buildNoSubgroupGraph();
  const result = removeSubgroups(graph, CTX_F32);
  equal(result, null, 'removeSubgroups should return null when no subgroup kernels present');
}

// ===========================================================================
// Test 3: widenToF32Activations transform
// ===========================================================================
{
  const graph = buildF16ActivationGraph();
  const frozen = structuredClone(graph);
  const result = widenToF32Activations(graph, CTX_F16);

  ok(result !== null, 'widenToF32Activations should return a non-null result for f16 activation graph');
  ok(result !== graph, 'widenToF32Activations should return a new object');

  // All f16 activation shaders should be replaced
  for (const entry of Object.values(result.kernels)) {
    ok(!entry.kernel.match(/^rmsnorm_f16\.wgsl$/), `should not contain rmsnorm_f16.wgsl, found ${entry.kernel}`);
    ok(!entry.kernel.match(/^rope_f16\.wgsl$/), `should not contain rope_f16.wgsl, found ${entry.kernel}`);
    ok(!entry.kernel.match(/^residual_f16\.wgsl$/), `should not contain residual_f16.wgsl, found ${entry.kernel}`);
    ok(!entry.kernel.match(/^gelu_f16\.wgsl$/), `should not contain gelu_f16.wgsl, found ${entry.kernel}`);
    ok(!entry.kernel.match(/^sample_f16\.wgsl$/), `should not contain sample_f16.wgsl, found ${entry.kernel}`);
  }

  // Verify specific replacements
  equal(result.kernels.rmsnorm.kernel, 'rmsnorm.wgsl', 'rmsnorm_f16 -> rmsnorm');
  equal(result.kernels.rope.kernel, 'rope.wgsl', 'rope_f16 -> rope');
  equal(result.kernels.residual.kernel, 'residual.wgsl', 'residual_f16 -> residual');
  equal(result.kernels.gelu.kernel, 'gelu.wgsl', 'gelu_f16 -> gelu');
  equal(result.kernels.sample.kernel, 'sample.wgsl', 'sample_f16 -> sample');
  equal(result.kernels.matmul.kernel, 'matmul_f16w_f32a.wgsl', 'matmul_f16 -> matmul_f16w_f32a');

  // Attention kernels should gain _f16kv suffix (KV stays f16)
  equal(result.kernels.attn_decode.kernel, 'attention_decode_chunked_f16kv.wgsl',
    'attention_decode_chunked_f16 -> attention_decode_chunked_f16kv');
  equal(result.kernels.attn_stream.kernel, 'attention_streaming_f16kv.wgsl',
    'attention_streaming_f16 -> attention_streaming_f16kv');

  // Original must not be mutated
  deepEqual(graph, frozen, 'widenToF32Activations must not mutate the input graph');
}

// ===========================================================================
// Test 4: widenToF32Activations returns null for fused f16 FFN
// ===========================================================================
{
  const graph = buildFusedF16Graph();
  const result = widenToF32Activations(graph, CTX_F16);
  equal(result, null, 'widenToF32Activations should return null when fused_ffn_f16.wgsl is present');
}

// ===========================================================================
// Test 5: swapPrefillAttention transform
// ===========================================================================
{
  const graph = structuredClone(REAL_GRAPH);
  const prefillAttentionIndex = graph.prefill.findIndex((step) => step[0] === 'attention');
  ok(prefillAttentionIndex !== -1, 'precondition: real graph must contain a prefill attention step');
  graph.kernels.attn_stream = {
    ...graph.kernels[graph.prefill[prefillAttentionIndex][1]],
    kernel: 'attention_streaming_f16kv.wgsl',
    digest: 'sha256:prefillstream',
  };
  graph.prefill[prefillAttentionIndex] = ['attention', 'attn_stream'];
  const frozen = structuredClone(graph);

  // Verify precondition: prefill uses attn_stream which has attention_streaming_f16kv.wgsl
  equal(graph.kernels.attn_stream.kernel, 'attention_streaming_f16kv.wgsl',
    'precondition: prefill attention is streaming f16kv');

  const result = swapPrefillAttention(graph, CTX_F32, {
    from: 'attention_streaming_f16kv.wgsl',
    to: 'attention_small_f16kv.wgsl',
  });

  ok(result !== null, 'swapPrefillAttention should return a non-null result');

  // Prefill attention should now use the small variant
  equal(result.kernels.attn_stream.kernel, 'attention_small_f16kv.wgsl',
    'prefill attention should be swapped to attention_small_f16kv.wgsl');

  // Decode attention must be UNCHANGED
  // The decode uses attn_decode which has attention_decode_online_f16kv.wgsl
  equal(result.kernels.attn_decode.kernel, 'attention_decode_online_f16kv.wgsl',
    'decode attention should remain unchanged');

  // Original must not be mutated
  deepEqual(graph, frozen, 'swapPrefillAttention must not mutate the input graph');
}

// ===========================================================================
// Test 6: useHead256PrefillAttention transform
// ===========================================================================
{
  const graph = structuredClone(REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useHead256PrefillAttention(graph, { ...CTX_F32, modelId: 'gemma-3-1b-it-q4k-ehf16-af32' });

  ok(result !== null, 'useHead256PrefillAttention should return a non-null result');
  equal(result.kernels.attn_small.kernel, 'attention_head256_f16kv.wgsl',
    'prefill attention should be swapped to attention_head256_f16kv.wgsl');
  equal(result.kernels.attn_decode.kernel, 'attention_decode_online_f16kv.wgsl',
    'decode attention should remain unchanged');
  equal(result.kernels.attn_small.digest, null,
    'head256 prefill attention should clear the digest after remap');

  deepEqual(graph, frozen, 'useHead256PrefillAttention must not mutate the input graph');
}

// ===========================================================================
// Test 6b: useHead256PrefillAttention supports legacy streaming prefill graphs
// ===========================================================================
{
  const graph = structuredClone(REAL_GRAPH);
  const prefillAttentionIndex = graph.prefill.findIndex((step) => step[0] === 'attention');
  ok(prefillAttentionIndex !== -1, 'precondition: real graph must contain a prefill attention step');
  graph.kernels.attn_stream = {
    ...graph.kernels[graph.prefill[prefillAttentionIndex][1]],
    kernel: 'attention_streaming_f16kv.wgsl',
    digest: 'sha256:legacy_prefill_stream',
  };
  graph.prefill[prefillAttentionIndex] = ['attention', 'attn_stream'];
  const frozen = structuredClone(graph);

  const result = useHead256PrefillAttention(graph, { ...CTX_F32, modelId: 'gemma-3-1b-it-q4k-ehf16-af32' });

  ok(result !== null, 'useHead256PrefillAttention should remap legacy streaming prefill attention');
  equal(result.kernels.attn_stream.kernel, 'attention_head256_f16kv.wgsl',
    'legacy streaming prefill attention should be swapped to attention_head256_f16kv.wgsl');
  equal(result.kernels.attn_stream.digest, null,
    'legacy streaming prefill attention should clear the digest after remap');
  equal(result.kernels.attn_decode.kernel, 'attention_decode_online_f16kv.wgsl',
    'decode attention should remain unchanged');

  deepEqual(graph, frozen, 'useHead256PrefillAttention must not mutate legacy streaming graphs');
}

// ===========================================================================
// Test 7: widenProjectionWeightsToF32 transform
// ===========================================================================
{
  const graph = buildF16WeightProjectionGraph();
  const frozen = structuredClone(graph);
  const result = widenProjectionWeightsToF32(graph, CTX_F32);

  ok(result !== null, 'widenProjectionWeightsToF32 should return a non-null result');

  // Projection kernels should now use matmul_f32.wgsl
  equal(result.kernels.proj_matmul.kernel, 'matmul_f32.wgsl',
    'projection matmul should be widened to matmul_f32.wgsl');

  // lm_head kernel must be UNCHANGED
  equal(result.kernels.lm_head_kernel.kernel, 'matmul_f16w_f32a.wgsl',
    'lm_head kernel should remain unchanged');

  // embed kernel must be UNCHANGED
  equal(result.kernels.embed.kernel, 'gather_f16.wgsl',
    'embed kernel should remain unchanged');

  // Original must not be mutated
  deepEqual(graph, frozen, 'widenProjectionWeightsToF32 must not mutate the input graph');
}

// ===========================================================================
// Test 8: remapDenseQ4KPrefillToQ4Native
// ===========================================================================
{
  const graph = structuredClone(REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = remapDenseQ4KPrefillToQ4Native(graph, { ...CTX_F32, modelId: 'gemma-3-1b-it-q4k-ehf16-af32' });

  ok(result !== null, 'remapDenseQ4KPrefillToQ4Native should remap Gemma 3 1B to the explicit shared Q4 prefill kernel');
  const prefillProjectionFiles = collectKernelFilesForPhase(result, 'prefill');
  ok(prefillProjectionFiles.has('fused_matmul_q4_batched_multicol_shared.wgsl'),
    'remapDenseQ4KPrefillToQ4Native should swap prefill projections to fused_matmul_q4_batched_multicol_shared.wgsl');

  deepEqual(graph, frozen, 'remapDenseQ4KPrefillToQ4Native must not mutate the input graph');
}

// ===========================================================================
// Test 9: remapQ4KPrefillToDense
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = remapQ4KPrefillToDense(graph, { ...CTX_F32, modelId: 'qwen-3-5-0-8b-q4k-ehaf16' });

  ok(result !== null, 'remapQ4KPrefillToDense should remap Qwen 3.5 0.8B prefill projections to dense kernels');
  const prefillProjectionFiles = collectKernelFilesForPhase(result, 'prefill');
  ok(prefillProjectionFiles.has('matmul_f16w_f32a.wgsl'),
    'remapQ4KPrefillToDense should swap prefill projections to matmul_f16w_f32a.wgsl');
  ok(!prefillProjectionFiles.has('fused_matmul_q4_batched.wgsl'),
    'remapQ4KPrefillToDense should remove fused_matmul_q4_batched.wgsl from prefill projections');

  deepEqual(graph, frozen, 'remapQ4KPrefillToDense must not mutate the input graph');
}

// ===========================================================================
// Test 10: useLinearDecodeProjectionF16
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useLinearDecodeProjectionF16(graph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: qwenConversionConfig.inference.layerPattern.layerTypes,
  });

  ok(result !== null, 'useLinearDecodeProjectionF16 should create a decode override for Qwen linear-attention layers');
  const qProjGroups = result.decode.filter((entry) => !Array.isArray(entry) && entry?.steps?.[0]?.[0] === 'q_proj');
  equal(qProjGroups.length, 2, 'useLinearDecodeProjectionF16 should split q_proj into linear and non-linear layer groups');
  const linearGroup = qProjGroups.find((entry) => entry.layers.includes(0) && entry.layers.includes(1) && !entry.layers.includes(3));
  ok(linearGroup, 'useLinearDecodeProjectionF16 should tag the linear-attention layers as a dedicated q_proj group');
  const linearKernelKey = linearGroup.steps[0][1];
  equal(
    result.kernels[linearKernelKey].precision?.outputDtype,
    'f16',
    'useLinearDecodeProjectionF16 should request f16 outputs for linear decode projections'
  );
  // o_proj must NOT be remapped to f16 — f16 truncation in the residual stream
  // across 18 linear-attention layers corrupts the logit distribution.
  const oProjGroups = result.decode.filter((entry) => !Array.isArray(entry) && entry?.steps?.[0]?.[0] === 'o_proj');
  equal(oProjGroups.length, 0, 'useLinearDecodeProjectionF16 must not split o_proj (f16 residual corruption)');
  const oProjTuples = result.decode.filter((entry) => Array.isArray(entry) && entry[0] === 'o_proj');
  equal(oProjTuples.length, 1, 'useLinearDecodeProjectionF16 must leave the o_proj tuple unchanged');
  equal(
    result.kernels.q4_decode.precision,
    undefined,
    'useLinearDecodeProjectionF16 should leave the original decode kernel entry unchanged for full-attention layers'
  );

  deepEqual(graph, frozen, 'useLinearDecodeProjectionF16 must not mutate the input graph');
}

// ===========================================================================
// Test 10b: useQwenDecodeF16Matmuls
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useQwenDecodeF16Matmuls(graph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  });

  ok(result !== null, 'useQwenDecodeF16Matmuls should rewrite selected Qwen decode matmuls onto explicit f16 kernels');
  equal(
    result.kernels[result.decode.find((entry) => Array.isArray(entry) && entry[0] === 'gate_proj')[1]].kernel,
    'fused_matmul_q4_multicol_f16a.wgsl',
    'useQwenDecodeF16Matmuls should move gate_proj onto the explicit q4 f16a kernel'
  );
  equal(
    result.kernels[result.decode.find((entry) => Array.isArray(entry) && entry[0] === 'up_proj')[1]].precision?.inputDtype,
    'f16',
    'useQwenDecodeF16Matmuls should request f16 FFN up-proj inputs'
  );
  equal(
    result.kernels[result.postLayer.find((entry) => entry[0] === 'lm_head')[1]].kernel,
    'matmul_gemv_subgroup_f16a.wgsl',
    'useQwenDecodeF16Matmuls should move decode lm_head onto the explicit f16a GEMV kernel'
  );

  deepEqual(graph, frozen, 'useQwenDecodeF16Matmuls must not mutate the input graph');
}

// ===========================================================================
// Test 10c: remapQ4KDecodeToGemv
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = remapQ4KDecodeToGemv(graph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  });

  ok(result !== null, 'remapQ4KDecodeToGemv should replace fused Q4K decode kernels with GEMV subgroup');
  const decodeFiles = collectKernelFilesForPhase(result, 'decode');
  ok(!decodeFiles.has('fused_matmul_q4.wgsl'),
    'remapQ4KDecodeToGemv should remove fused_matmul_q4.wgsl from decode');
  ok(decodeFiles.has('matmul_gemv_subgroup.wgsl'),
    'remapQ4KDecodeToGemv should add matmul_gemv_subgroup.wgsl to decode');

  // Verify all projection ops now use the GEMV kernel
  for (const op of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
    const step = result.decode.find((entry) => Array.isArray(entry) && entry[0] === op);
    ok(step, `remapQ4KDecodeToGemv: ${op} step should exist in decode`);
    const entry = result.kernels[step[1]];
    equal(entry.kernel, 'matmul_gemv_subgroup.wgsl',
      `remapQ4KDecodeToGemv: ${op} should use matmul_gemv_subgroup.wgsl, got ${entry.kernel}`);
    equal(entry.entry, 'main_multicol',
      `remapQ4KDecodeToGemv: ${op} entry should be main_multicol`);
  }

  // Non-matmul decode ops (rmsnorm, rope, attention, residual, silu) must be unchanged
  const nonMatmulOps = ['input_norm', 'rope_q', 'rope_k', 'attention', 'attn_residual', 'post_attn_norm', 'activation', 'ffn_residual'];
  for (const op of nonMatmulOps) {
    const step = result.decode.find((entry) => Array.isArray(entry) && entry[0] === op);
    ok(step, `remapQ4KDecodeToGemv: ${op} should still be present`);
    const origStep = graph.decode.find((entry) => Array.isArray(entry) && entry[0] === op);
    equal(step[1], origStep[1],
      `remapQ4KDecodeToGemv: ${op} kernel key should be unchanged`);
  }

  // Prefill must be untouched
  deepEqual(result.prefill, graph.prefill, 'remapQ4KDecodeToGemv should not modify prefill');

  // PostLayer (lm_head) must be untouched
  deepEqual(result.postLayer, graph.postLayer, 'remapQ4KDecodeToGemv should not modify postLayer');

  // No-op for f16 activation context
  const f16Result = remapQ4KDecodeToGemv(graph, { ...CTX_F16, modelId: 'qwen-3-5-0-8b-q4k-ehaf16' });
  equal(f16Result, null, 'remapQ4KDecodeToGemv should return null for f16 activations');

  deepEqual(graph, frozen, 'remapQ4KDecodeToGemv must not mutate the input graph');
}

// ===========================================================================
// Test 11: composeTransforms — only first applies
// ===========================================================================
{
  // For the real graph (f32 activations with subgroup kernels):
  // removeSubgroups should apply, widenToF32Activations should return null (already f32)
  const composed = composeTransforms(removeSubgroups, widenToF32Activations);
  const result = composed(structuredClone(REAL_GRAPH), CTX_F32);

  // removeSubgroups should have applied — subgroup matmul kernels replaced
  const decodeFiles = collectKernelFilesForPhase(result, 'decode');
  ok(!decodeFiles.has('matmul_gemv_subgroup.wgsl'),
    'compose: removeSubgroups should have removed subgroup matmul');

  // Decode projections should now reference tiled kernels
  ok(decodeFiles.has('matmul_f16w_f32a_tiled.wgsl'),
    'compose: decode projections should use matmul_f16w_f32a_tiled.wgsl');

  // widenToF32Activations was a no-op since graph already uses f32 activation shaders
  // Verify non-f16 shaders are still present (not widened further)
  ok(Object.values(result.kernels).some(e => e.kernel === 'rmsnorm.wgsl'),
    'compose: rmsnorm.wgsl should still be present (already f32)');
}

// ===========================================================================
// Test 12: composeTransforms with multiple applicable transforms
// ===========================================================================
{
  const graph = buildF16SubgroupGraph();
  const composed = composeTransforms(removeSubgroups, widenToF32Activations);
  const result = composed(graph, CTX_F16);

  // Collect all kernel files that are actually referenced by steps
  function collectReferencedKernelFiles(g) {
    const files = new Set();
    for (const phase of ['preLayer', 'decode', 'prefill', 'postLayer']) {
      const steps = g[phase];
      if (!Array.isArray(steps)) continue;
      for (const step of steps) {
        const entry = g.kernels[step[1]];
        if (entry) files.add(entry.kernel);
      }
    }
    return files;
  }

  const referencedFiles = collectReferencedKernelFiles(result);

  // No referenced kernel should use a subgroup matmul shader
  ok(!referencedFiles.has('matmul_gemv_subgroup_f16a.wgsl'),
    'compose: no referenced kernel should use matmul_gemv_subgroup_f16a.wgsl');
  ok(!referencedFiles.has('matmul_gemv_subgroup.wgsl'),
    'compose: no referenced kernel should use matmul_gemv_subgroup.wgsl');

  // No referenced kernel should use f16 activation shaders
  const f16ActivationShaders = new Set([
    'rmsnorm_f16.wgsl', 'rope_f16.wgsl', 'residual_f16.wgsl',
    'gelu_f16.wgsl', 'sample_f16.wgsl', 'matmul_f16.wgsl',
    'matmul_f16_tiled.wgsl',
    'attention_decode_chunked_f16.wgsl',
    'attention_streaming_f16.wgsl',
  ]);
  for (const file of referencedFiles) {
    ok(!f16ActivationShaders.has(file),
      `compose: no f16 activation shader should be referenced, found ${file}`);
  }
}

// ===========================================================================
// Test 13: resolveCapabilityTransforms
// ===========================================================================
{
  const platform = { id: 'test', vendor: 'test', architecture: 'test' };
  const graphCtx = {
    activationDtype: 'f32',
    kvDtype: 'f16',
    hasDensePrefillProjectionKernel: false,
    hasQ4DecodeProjectionKernel: false,
    hasQ4PrefillProjectionKernel: false,
    hasAvailableQ4PrefillProjectionKernel: false,
  };

  // { hasSubgroups: true, hasF16: true } => empty transforms
  {
    const r = resolveCapabilityTransforms({ hasSubgroups: true, hasF16: true }, platform, graphCtx);
    deepEqual(r.names, [], 'subgroups+f16: no transforms needed');
    deepEqual(r.transforms, [], 'subgroups+f16: no transform functions');
    ok(r.reason.toLowerCase().includes('all required features') || r.reason.toLowerCase().includes('supports all'),
      `subgroups+f16: reason should mention all required features, got: "${r.reason}"`);
  }

  // { hasSubgroups: false, hasF16: true } => ["removeSubgroups"]
  {
    const r = resolveCapabilityTransforms({ hasSubgroups: false, hasF16: true }, platform, graphCtx);
    deepEqual(r.names, ['removeSubgroups'], 'no subgroups: should resolve removeSubgroups');
    equal(r.transforms.length, 1, 'no subgroups: one transform function');
    equal(r.transforms[0], removeSubgroups, 'no subgroups: transform function is removeSubgroups');
  }

  // { hasSubgroups: true, hasF16: false } => ["widenToF32Activations"]
  {
    const r = resolveCapabilityTransforms({ hasSubgroups: true, hasF16: false }, platform, graphCtx);
    deepEqual(r.names, ['widenToF32Activations'], 'no f16: should resolve widenToF32Activations');
    equal(r.transforms.length, 1, 'no f16: one transform function');
    equal(r.transforms[0], widenToF32Activations, 'no f16: transform function is widenToF32Activations');
  }

  // { hasSubgroups: false, hasF16: false } => ["removeSubgroups", "widenToF32Activations"]
  {
    const r = resolveCapabilityTransforms({ hasSubgroups: false, hasF16: false }, platform, graphCtx);
    deepEqual(r.names, ['removeSubgroups', 'widenToF32Activations'],
      'no subgroups+no f16: should resolve both transforms');
    equal(r.transforms.length, 2, 'no subgroups+no f16: two transform functions');
    equal(r.transforms[0], removeSubgroups, 'no subgroups+no f16: first is removeSubgroups');
    equal(r.transforms[1], widenToF32Activations, 'no subgroups+no f16: second is widenToF32Activations');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: {
          vendor: 'apple',
          architecture: 'metal-3',
        },
      },
      { id: 'apple-m3', vendor: 'apple', architecture: 'm-series' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        modelId: 'gemma-3-1b-it-q4k-ehf16-af32',
        hasDensePrefillProjectionKernel: true,
        hasQ4DecodeProjectionKernel: false,
        hasQ4PrefillProjectionKernel: false,
        hasAvailableQ4PrefillProjectionKernel: true,
      }
    );
    deepEqual(
      r.names,
      ['useHead256PrefillAttention'],
      'apple Gemma 3 1B graph should resolve the fixed head-dim-256 prefill attention transform'
    );
    equal(r.transforms.length, 1, 'apple Gemma 3 1B graph: one transform function');
    equal(r.transforms[0], useHead256PrefillAttention,
      'apple Gemma 3 1B graph: transform function is useHead256PrefillAttention');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: {
          vendor: 'apple',
          architecture: 'metal-3',
        },
      },
      { id: 'apple-m3', vendor: 'apple', architecture: 'm-series' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
        hasDensePrefillProjectionKernel: true,
        hasQ4DecodeProjectionKernel: false,
        hasQ4PrefillProjectionKernel: false,
        hasAvailableQ4PrefillProjectionKernel: true,
      }
    );
    deepEqual(r.names, ['remapDenseQ4KPrefillToQ4Native'], 'apple dense-only Gemma graph with explicit Q4 prefill kernel should resolve remapDenseQ4KPrefillToQ4Native');
    equal(r.transforms.length, 1, 'apple dense-only Gemma graph with explicit Q4 prefill kernel: one transform function');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: {
          vendor: 'apple',
          architecture: 'metal-3',
        },
      },
      { id: 'generic' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        hasDensePrefillProjectionKernel: true,
        hasQ4DecodeProjectionKernel: false,
        hasQ4PrefillProjectionKernel: false,
        hasAvailableQ4PrefillProjectionKernel: true,
      }
    );
    deepEqual(r.names, ['remapDenseQ4KPrefillToQ4Native'], 'generic platform with Apple adapter info should still resolve remapDenseQ4KPrefillToQ4Native');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: {
          vendor: 'apple',
          architecture: 'metal-3',
        },
      },
      { id: 'apple-m3', vendor: 'apple', architecture: 'm-series' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
        hasDensePrefillProjectionKernel: false,
        hasQ4DecodeProjectionKernel: true,
        hasQ4PrefillProjectionKernel: true,
        hasAvailableQ4PrefillProjectionKernel: true,
      }
    );
    deepEqual(
      r.names,
      ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeToGemv'],
      'apple Qwen 3.5 0.8B graph should resolve prefill remap and decode GEMV transforms'
    );
    equal(r.transforms.length, 3, 'apple Qwen 3.5 0.8B graph: three transform functions');
    equal(r.transforms[0], useHead256PrefillAttention,
      'apple Qwen 3.5 0.8B graph: first transform function is useHead256PrefillAttention');
    equal(r.transforms[1], remapQ4KPrefillToDense,
      'apple Qwen 3.5 0.8B graph: second transform function is remapQ4KPrefillToDense');
    equal(r.transforms[2], remapQ4KDecodeToGemv,
      'apple Qwen 3.5 0.8B graph: third transform function is remapQ4KDecodeToGemv');
  }
}

// ===========================================================================
// Test 14: resolveFinitenessFallbackTransform
// ===========================================================================
{
  // activationDtype 'f16' => returns transform
  {
    const r = resolveFinitenessFallbackTransform({ activationDtype: 'f16' });
    ok(r !== null, 'f16 activationDtype: should return a transform');
    equal(r.name, 'widenToF32Activations', 'f16 activationDtype: name is widenToF32Activations');
    equal(r.transform, widenToF32Activations, 'f16 activationDtype: transform is widenToF32Activations');
  }

  // activationDtype 'f32' => returns null
  {
    const r = resolveFinitenessFallbackTransform({ activationDtype: 'f32' });
    equal(r, null, 'f32 activationDtype: should return null');
  }
}

// ===========================================================================
// Test 15: Snapshot test — removeSubgroups produces expected kernel files
// ===========================================================================
{
  const input = structuredClone(REAL_GRAPH);
  const result = removeSubgroups(input, CTX_F32);
  ok(result !== null, 'snapshot: removeSubgroups should produce a result');

  // Decode projections should use tiled matmul (not subgroup GEMV)
  for (const step of result.decode) {
    const op = step[0];
    const kernelKey = step[1];
    const entry = result.kernels[kernelKey];
    ok(entry != null, `snapshot decode: kernel entry for op "${op}" (key "${kernelKey}") must exist`);

    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      equal(entry.kernel, 'matmul_f16w_f32a_tiled.wgsl',
        `snapshot decode: op "${op}" should use matmul_f16w_f32a_tiled.wgsl, got ${entry.kernel}`);
    }
    if (op === 'attention') {
      equal(entry.kernel, 'attention_decode_chunked_f16kv.wgsl',
        `snapshot decode: attention should use chunked kernel, got ${entry.kernel}`);
    }
  }

  // PostLayer lm_head should use matmul_f16w_f32a.wgsl (not subgroup multicol)
  {
    const lmHeadStep = result.postLayer.find(s => s[0] === 'lm_head');
    ok(lmHeadStep != null, 'snapshot postLayer: lm_head step must exist');
    const lmHeadEntry = result.kernels[lmHeadStep[1]];
    ok(lmHeadEntry != null, 'snapshot postLayer: lm_head kernel entry must exist');
    equal(lmHeadEntry.kernel, 'matmul_f16w_f32a.wgsl',
      `snapshot postLayer: lm_head should use matmul_f16w_f32a.wgsl, got ${lmHeadEntry.kernel}`);
  }

  // Prefill projection kernels should fall back to dense tiled matmul
  const prefillProjFiles = new Set();
  for (const step of result.prefill) {
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = result.kernels[step[1]];
      if (entry) prefillProjFiles.add(entry.kernel);
    }
  }
  equal(prefillProjFiles.size, 1, 'snapshot: all prefill projections should use the same kernel file');
  ok(prefillProjFiles.has('matmul_f16w_f32a_tiled.wgsl'),
    'snapshot: prefill projections should use matmul_f16w_f32a_tiled.wgsl');
}

// ===========================================================================
// Test 16: Purity/immutability test
// ===========================================================================
{
  const input = structuredClone(REAL_GRAPH);
  const preClone = structuredClone(input);

  // Apply removeSubgroups
  removeSubgroups(input, CTX_F32);
  deepEqual(input, preClone, 'purity: removeSubgroups must not mutate input');

  // Apply widenToF32Activations on an f16 graph
  const f16Graph = buildF16ActivationGraph();
  const f16Clone = structuredClone(f16Graph);
  widenToF32Activations(f16Graph, CTX_F16);
  deepEqual(f16Graph, f16Clone, 'purity: widenToF32Activations must not mutate input');

  // Apply swapPrefillAttention
  const swapGraph = structuredClone(REAL_GRAPH);
  const swapClone = structuredClone(swapGraph);
  swapPrefillAttention(swapGraph, CTX_F32, {
    from: 'attention_streaming_f16kv.wgsl',
    to: 'attention_small_f16kv.wgsl',
  });
  deepEqual(swapGraph, swapClone, 'purity: swapPrefillAttention must not mutate input');

  const head256Graph = structuredClone(REAL_GRAPH);
  const head256Clone = structuredClone(head256Graph);
  useHead256PrefillAttention(head256Graph, CTX_F32);
  deepEqual(head256Graph, head256Clone, 'purity: useHead256PrefillAttention must not mutate input');

  // Apply widenProjectionWeightsToF32
  const projGraph = buildF16WeightProjectionGraph();
  const projClone = structuredClone(projGraph);
  widenProjectionWeightsToF32(projGraph, CTX_F32);
  deepEqual(projGraph, projClone, 'purity: widenProjectionWeightsToF32 must not mutate input');

  // Apply remapDenseQ4KPrefillToQ4Native
  const q4PrefillGraph = structuredClone(REAL_GRAPH);
  const q4PrefillClone = structuredClone(q4PrefillGraph);
  remapDenseQ4KPrefillToQ4Native(q4PrefillGraph, CTX_F32);
  deepEqual(q4PrefillGraph, q4PrefillClone, 'purity: remapDenseQ4KPrefillToQ4Native must not mutate input');

  const qwenLinearGraph = structuredClone(QWEN_REAL_GRAPH);
  const qwenLinearClone = structuredClone(qwenLinearGraph);
  useLinearDecodeProjectionF16(qwenLinearGraph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: qwenConversionConfig.inference.layerPattern.layerTypes,
  });
  deepEqual(qwenLinearGraph, qwenLinearClone, 'purity: useLinearDecodeProjectionF16 must not mutate input');
}

// ===========================================================================
// Test 17: digest is nulled on modified kernels
// ===========================================================================
{
  const input = structuredClone(REAL_GRAPH);
  const result = removeSubgroups(input, CTX_F32);
  ok(result !== null, 'digest test: removeSubgroups should produce a result');

  // Collect kernel keys that were remapped (new keys ending in _nosg)
  const modifiedKeys = Object.keys(result.kernels).filter(k => k.endsWith('_nosg'));
  ok(modifiedKeys.length > 0, 'digest test: should have at least one _nosg kernel key');

  // Modified kernel entries must have digest: null
  for (const key of modifiedKeys) {
    equal(result.kernels[key].digest, null,
      `digest test: modified kernel "${key}" should have digest: null`);
  }

  // Unmodified kernel entries must retain their original digest
  const originalKeys = Object.keys(input.kernels);
  for (const key of originalKeys) {
    if (result.kernels[key] && !modifiedKeys.includes(key)) {
      equal(result.kernels[key].digest, input.kernels[key].digest,
        `digest test: unmodified kernel "${key}" should retain original digest`);
    }
  }
}

// ===========================================================================

console.log(`execution-graph-transforms.test: ${assertions} assertions passed`);
