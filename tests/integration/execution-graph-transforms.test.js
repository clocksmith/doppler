import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
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
  remapQ4KDecodeAttentionToFusedQ4KGemv,
  remapQ4KDecodeFFNToGemv,
  useQwenF16PrimaryMatmuls,
  useQwenDecodeF16Matmuls,
  useGemma431BTextF16Activations,
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

const gemma4ConversionConfig = JSON.parse(
  readFileSync(
    path.resolve(__dirname, '../../src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json'),
    'utf-8'
  )
);

const gemma431BConversionConfig = JSON.parse(
  readFileSync(
    path.resolve(__dirname, '../../src/config/conversion/gemma4/gemma-4-31b-it-text-q4k-ehf16-af32.json'),
    'utf-8'
  )
);

/** The real execution graph from the gemma3-1b-q4k conversion config. */
const REAL_GRAPH = conversionConfig.execution;
const QWEN_REAL_GRAPH = qwenConversionConfig.execution;
const GEMMA4_REAL_GRAPH = gemma4ConversionConfig.execution;
const GEMMA4_31B_REAL_GRAPH = gemma431BConversionConfig.execution;

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
  graph.kernels.attn_decode.precision = { activationDtype: 'f16', kvDtype: 'f16' };
  graph.kernels.attn_stream.precision = { activationDtype: 'f16', kvDtype: 'f16' };
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
  deepEqual(
    result.kernels.attn_decode.precision,
    { activationDtype: 'f32', kvDtype: 'f16' },
    'attention_decode_chunked_f16 -> attention_decode_chunked_f16kv should rewrite activation precision to f32'
  );
  deepEqual(
    result.kernels.attn_stream.precision,
    { activationDtype: 'f32', kvDtype: 'f16' },
    'attention_streaming_f16 -> attention_streaming_f16kv should rewrite activation precision to f32'
  );

  // Original must not be mutated
  deepEqual(graph, frozen, 'widenToF32Activations must not mutate the input graph');
}

// ===========================================================================
// Test 3a: full f32 widening uses valid matmul_f32 entry point
// ===========================================================================
{
  const graph = {
    kernels: {
      q_proj: { kernel: 'matmul_gemv_subgroup_f16a.wgsl', entry: 'main_vec4', digest: 'sha256:q' },
      lm_head: { kernel: 'matmul_gemv_subgroup_f16a.wgsl', entry: 'main_multicol', digest: 'sha256:lm' },
      prefill: { kernel: 'matmul_f16_tiled.wgsl', entry: 'main', digest: 'sha256:p' },
    },
    decode: [['q_proj', 'q_proj'], ['lm_head', 'lm_head']],
    prefill: [['q_proj', 'prefill']],
    postLayer: [],
    policies: { unsupportedPrecision: 'error' },
  };

  const result = widenToF32Activations(graph, { ...CTX_F16, capabilities: { hasF16: false } });
  ok(result !== null, 'full f32 widening should apply to f16 matmul kernels');
  equal(result.kernels.q_proj.kernel, 'matmul_f32.wgsl',
    'full f32 widening should replace q_proj with matmul_f32.wgsl');
  equal(result.kernels.q_proj.entry, 'main',
    'full f32 widening should not preserve main_vec4 for matmul_f32.wgsl');
  equal(result.kernels.lm_head.kernel, 'matmul_f32.wgsl',
    'full f32 widening should replace lm_head with matmul_f32.wgsl');
  equal(result.kernels.lm_head.entry, 'main',
    'full f32 widening should not preserve main_multicol for matmul_f32.wgsl');
  equal(result.kernels.prefill.entry, 'main',
    'full f32 widening should keep the valid matmul_f32.wgsl entry point');
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
  // The real graph now uses attn_head256 as primary. Build a synthetic graph
  // with attn_small to test the transform function's swap behaviour.
  const graph = structuredClone(REAL_GRAPH);
  const prefillAttnIdx = graph.prefill.findIndex((step) => step[0] === 'attention');
  ok(prefillAttnIdx !== -1, 'precondition: real graph must contain a prefill attention step');
  graph.prefill[prefillAttnIdx] = ['attention', 'attn_small'];
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
// Test 6c: useHead256PrefillAttention returns null when already head256
// ===========================================================================
{
  const graph = structuredClone(REAL_GRAPH);
  const result = useHead256PrefillAttention(graph, { ...CTX_F32, modelId: 'gemma-3-1b-it-q4k-ehf16-af32' });
  equal(result, null, 'useHead256PrefillAttention should return null when prefill already uses head256');
}

// ===========================================================================
// Test 6d: grouped prefill graphs support streaming -> small remap
// ===========================================================================
{
  const graph = structuredClone(GEMMA4_REAL_GRAPH);
  graph.prefill[1].steps[6][1] = 'attn_stream';
  const frozen = structuredClone(graph);

  const result = swapPrefillAttention(graph, CTX_F32, {
    from: 'attention_streaming_f16kv.wgsl',
    to: 'attention_small_f16kv.wgsl',
  });

  ok(result !== null, 'swapPrefillAttention should remap grouped Gemma 4 prefill graphs');
  equal(result.kernels.attn_stream.kernel, 'attention_small_f16kv.wgsl',
    'grouped Gemma 4 prefill streaming attention should be swapped to attention_small_f16kv.wgsl');
  equal(result.kernels.attn_small.kernel, 'attention_small_f16kv.wgsl',
    'existing small attention should remain small');
  deepEqual(graph, frozen, 'swapPrefillAttention must not mutate grouped Gemma 4 graphs');
}

// ===========================================================================
// Test 6e: useHead256SmallPrefillAttention remaps grouped Gemma 4 small prefill attention only
// ===========================================================================
{
  const graph = structuredClone(GEMMA4_REAL_GRAPH);
  delete graph.kernels.attn_head256;
  graph.prefill[0].steps[6][1] = 'attn_small';
  const frozen = structuredClone(graph);

  const result = useHead256SmallPrefillAttention(graph, { ...CTX_F32, modelId: 'gemma-4-e2b-it-q4k-ehf16-af32' });

  ok(result !== null, 'useHead256SmallPrefillAttention should remap grouped Gemma 4 small prefill attention');
  equal(result.kernels.attn_small.kernel, 'attention_head256_f16kv.wgsl',
    'grouped Gemma 4 small prefill attention should be remapped to head256');
  equal(result.kernels.attn_stream.kernel, 'attention_streaming_f16kv.wgsl',
    'grouped Gemma 4 streaming prefill attention should remain on the streaming kernel');
  equal(result.kernels.attn_small.digest, null,
    'grouped Gemma 4 head256 prefill remap should clear the small digest');
  equal(result.kernels.attn_stream.digest, graph.kernels.attn_stream.digest,
    'grouped Gemma 4 streaming prefill attention should preserve its digest');
  deepEqual(graph, frozen, 'useHead256SmallPrefillAttention must not mutate grouped Gemma 4 graphs');
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
// Test 9: remapQ4KPrefillToDense — explicit diagnostic transform on Qwen
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = remapQ4KPrefillToDense(graph, { ...CTX_F32, modelId: 'qwen-3-5-0-8b-q4k-ehaf16' });

  ok(result, 'remapQ4KPrefillToDense should return a dense-prefill diagnostic graph on Qwen');

  const prefillProjectionFiles = collectKernelFilesForPhase(result, 'prefill');
  ok(prefillProjectionFiles.has('matmul_f16w_f32a.wgsl'),
    'remapped Qwen prefill should have matmul_f16w_f32a.wgsl');
  ok(!prefillProjectionFiles.has('fused_matmul_q4_batched_f16a.wgsl'),
    'remapped Qwen prefill should no longer have fused_matmul_q4_batched_f16a.wgsl');

  deepEqual(graph, frozen, 'remapQ4KPrefillToDense must not mutate the input graph');
}

// ===========================================================================
// Test 10: useLinearDecodeProjectionF16 — no-op on the promoted Qwen primary graph
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useLinearDecodeProjectionF16(graph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: qwenConversionConfig.inference.layerPattern.layerTypes,
  });

  equal(result, null,
    'useLinearDecodeProjectionF16 should be a no-op when the promoted Qwen graph is already specialized');

  deepEqual(graph, frozen, 'useLinearDecodeProjectionF16 must not mutate the input graph');
}

// ===========================================================================
// Test 10b: useQwenDecodeF16Matmuls — no-op on the manifest-owned Q4 graph
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useQwenDecodeF16Matmuls(graph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  });

  equal(result, null,
    'useQwenDecodeF16Matmuls should be a no-op when Qwen decode and LM head are manifest-owned Q4 kernels');

  deepEqual(graph, frozen, 'useQwenDecodeF16Matmuls must not mutate the input graph');
}

// ===========================================================================
// Test 10c: useQwenF16PrimaryMatmuls — promoted Qwen runtime-requested f16 lane
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useQwenF16PrimaryMatmuls(graph, {
    ...CTX_F16,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    capabilities: { hasF16: true },
    layerTypes: qwenConversionConfig.inference.layerPattern.layerTypes,
  });

  ok(result, 'useQwenF16PrimaryMatmuls should derive the promoted Qwen f16 primary graph');

  const decodeFiles = collectKernelFilesForPhase(result, 'decode');
  ok(decodeFiles.has('fused_matmul_q4_multicol_f16a.wgsl'),
    'promoted Qwen f16 graph should use fused_matmul_q4_multicol_f16a.wgsl in decode');

  const prefillFiles = collectKernelFilesForPhase(result, 'prefill');
  ok(prefillFiles.has('fused_matmul_q4_widetile.wgsl'),
    'promoted Qwen f16 graph should preserve the manifest WideTile Q4 prefill kernel');
  ok(prefillFiles.has('attention_head256_f16kv.wgsl'),
    'promoted Qwen f16 graph should preserve the manifest head256 prefill attention kernel');

  const decodeOProj = result.decode.find((entry) => Array.isArray(entry) && entry[0] === 'o_proj');
  ok(decodeOProj, 'promoted Qwen f16 graph should keep the decode o_proj step');
  equal(result.kernels[decodeOProj[1]].kernel, 'fused_matmul_q4.wgsl',
    'promoted Qwen f16 graph should keep decode o_proj on the original fused q4 kernel');
  deepEqual(
    result.kernels[decodeOProj[1]].precision,
    { inputDtype: 'f32', outputDtype: 'f32' },
    'promoted Qwen f16 graph should declare the decode o_proj f32 boundary explicitly'
  );

  const prefillOProj = result.prefill.find((entry) => Array.isArray(entry) && entry[0] === 'o_proj');
  ok(prefillOProj, 'promoted Qwen f16 graph should keep the prefill o_proj step');
  equal(result.kernels[prefillOProj[1]].kernel, 'fused_matmul_q4_widetile.wgsl',
    'promoted Qwen f16 graph should keep prefill o_proj on the original WideTile q4 prefill kernel');
  deepEqual(
    result.kernels[prefillOProj[1]].precision,
    { inputDtype: 'f32', outputDtype: 'f32' },
    'promoted Qwen f16 graph should declare the prefill o_proj f32 boundary explicitly'
  );

  equal(
    result.kernels[result.postLayer.find((entry) => Array.isArray(entry) && entry[0] === 'lm_head')[1]].kernel,
    'fused_matmul_q4.wgsl',
    'promoted Qwen f16 graph should preserve the manifest-owned Q4 lm_head kernel'
  );

  deepEqual(graph, frozen, 'useQwenF16PrimaryMatmuls must not mutate the input graph');
}

// ===========================================================================
// Test 10d: useGemma431BTextF16Activations — experimental Gemma 4 31B all-f16 lane
// ===========================================================================
{
  const graph = structuredClone(GEMMA4_31B_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = useGemma431BTextF16Activations(graph, {
    ...CTX_F16,
    mathDtype: 'f16',
    accumDtype: 'f16',
    modelId: 'gemma-4-31b-it-text-q4k-ehf16-af32',
    capabilities: { hasF16: true, hasSubgroups: true },
    layerTypes: gemma431BConversionConfig.inference.layerPattern.layerTypes,
  });

  ok(result, 'Gemma 4 31B f16 lane should derive an execution graph');

  const embed = result.preLayer.find((entry) => Array.isArray(entry) && entry[0] === 'embed');
  ok(embed, 'Gemma 4 31B f16 lane should keep embed');
  equal(result.kernels[embed[1]].kernel, 'gather_f16_vec4_f16_out.wgsl',
    'Gemma 4 31B f16 lane should emit f16 embeddings into f16 activations');
  deepEqual(result.kernels[embed[1]].precision, { inputDtype: 'f16', outputDtype: 'f16' },
    'Gemma 4 31B f16 lane should declare embed f16 precision');

  const decodeQ = result.decode.find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  ok(decodeQ, 'Gemma 4 31B f16 lane should keep decode q_proj');
  equal(result.kernels[decodeQ[1]].kernel, 'fused_matmul_q4_multicol_f16a.wgsl',
    'Gemma 4 31B f16 lane should use f16 activation Q4 decode projection');
  deepEqual(result.kernels[decodeQ[1]].precision, { inputDtype: 'f16', outputDtype: 'f16' },
    'Gemma 4 31B f16 lane should declare decode projection f16 precision');

  const decodeAttention = result.decode.find((entry) => Array.isArray(entry) && entry[0] === 'attention');
  ok(decodeAttention, 'Gemma 4 31B f16 lane should keep decode attention');
  equal(result.kernels[decodeAttention[1]].kernel, 'attention_decode_online_f16.wgsl',
    'Gemma 4 31B f16 lane should use f16 decode attention');

  const slidingPrefill = result.prefill[0].steps;
  const fullPrefill = result.prefill[1].steps;
  const slidingPrefillQ = slidingPrefill.find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const fullPrefillQ = fullPrefill.find((entry) => Array.isArray(entry) && entry[0] === 'q_proj');
  const slidingPrefillAttention = slidingPrefill.find((entry) => Array.isArray(entry) && entry[0] === 'attention');
  const fullPrefillAttention = fullPrefill.find((entry) => Array.isArray(entry) && entry[0] === 'attention');
  ok(slidingPrefillQ && fullPrefillQ && slidingPrefillAttention && fullPrefillAttention,
    'Gemma 4 31B f16 lane should keep grouped prefill projection and attention steps');
  equal(result.kernels[slidingPrefillQ[1]].kernel, 'fused_matmul_q4_batched_f16acc_f16a.wgsl',
    'Gemma 4 31B f16 lane should use f16-accum Q4 prefill projection on sliding layers');
  equal(result.kernels[fullPrefillQ[1]].kernel, 'fused_matmul_q4_batched_f16acc_f16a.wgsl',
    'Gemma 4 31B f16 lane should use f16-accum Q4 prefill projection on full-attention layers');
  equal(result.kernels[slidingPrefillAttention[1]].kernel, 'attention_small_f16.wgsl',
    'Gemma 4 31B f16 lane should use f16 sliding prefill attention');
  equal(result.kernels[fullPrefillAttention[1]].kernel, 'attention_head512_f16.wgsl',
    'Gemma 4 31B f16 lane should use f16 head512 prefill attention');

  const postKernel = (op) => {
    const step = result.postLayer.find((entry) => Array.isArray(entry) && entry[0] === op);
    ok(step, `Gemma 4 31B f16 lane should keep postLayer ${op}`);
    return result.kernels[step[1]];
  };
  equal(postKernel('final_norm').kernel, 'rmsnorm_f16.wgsl',
    'Gemma 4 31B f16 lane should move final norm to f16');
  equal(postKernel('lm_head').kernel, 'matmul_gemv_subgroup_f16a.wgsl',
    'Gemma 4 31B f16 lane should move decode lm_head to f16 activations');
  equal(postKernel('lm_head_prefill').kernel, 'matmul_f16_tiled.wgsl',
    'Gemma 4 31B f16 lane should move prefill lm_head to f16 activations');
  equal(postKernel('sample').kernel, 'sample_f16.wgsl',
    'Gemma 4 31B f16 lane should sample from f16 logits');

  deepEqual(graph, frozen, 'useGemma431BTextF16Activations must not mutate the input graph');
}

// ===========================================================================
// Test 10e: remapQ4KDecodeToGemv — explicit diagnostic transform on Qwen
// ===========================================================================
{
  const graph = structuredClone(QWEN_REAL_GRAPH);
  const frozen = structuredClone(graph);
  const result = remapQ4KDecodeToGemv(graph, {
    ...CTX_F32,
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  });

  ok(result, 'remapQ4KDecodeToGemv should return a GEMV diagnostic graph on Qwen');

  const decodeFiles = collectKernelFilesForPhase(result, 'decode');
  ok(decodeFiles.has('matmul_gemv_subgroup.wgsl'),
    'remapped Qwen decode should have matmul_gemv_subgroup.wgsl');
  ok(!decodeFiles.has('fused_matmul_q4.wgsl'),
    'remapped Qwen decode should no longer have fused_matmul_q4.wgsl');

  // Verify all projection ops now use the GEMV kernel
  for (const op of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
    const step = result.decode.find((entry) => Array.isArray(entry) && entry[0] === op);
    ok(step, `Remapped Qwen graph: ${op} step should exist in decode`);
    const entry = result.kernels[step[1]];
    equal(entry.kernel, 'matmul_gemv_subgroup.wgsl',
      `Remapped Qwen graph: ${op} should use matmul_gemv_subgroup.wgsl, got ${entry.kernel}`);
  }

  // Verify prefill keeps the manifest WideTile/head256 primary path.
  const prefillFiles = collectKernelFilesForPhase(result, 'prefill');
  ok(prefillFiles.has('fused_matmul_q4_widetile.wgsl'),
    'Remapped Qwen graph should keep fused_matmul_q4_widetile.wgsl in prefill');
  ok(prefillFiles.has('attention_head256_f16kv.wgsl'),
    'Remapped Qwen graph should keep attention_head256_f16kv.wgsl in prefill');

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

  // f16 request only resolves the narrowing transform when the graph still
  // contains f32-activation kernels that can be narrowed.
  {
    const r = resolveCapabilityTransforms(
      { hasSubgroups: true, hasF16: true },
      platform,
      { ...graphCtx, activationDtype: 'f16', requiresF16ActivationNarrowing: true }
    );
    deepEqual(r.names, ['narrowToF16Activations'],
      'f16 request with narrowable kernels: should resolve narrowToF16Activations');
    equal(r.transforms.length, 1, 'f16 request with narrowable kernels: one transform function');
    equal(r.transforms[0], narrowToF16Activations,
      'f16 request with narrowable kernels: transform function is narrowToF16Activations');
  }

  {
    const r = resolveCapabilityTransforms(
      { hasSubgroups: true, hasF16: true },
      platform,
      { ...graphCtx, activationDtype: 'f16', requiresF16ActivationNarrowing: false }
    );
    deepEqual(r.names, [],
      'f16 request on an already-f16 graph: no capability transforms needed');
    equal(r.transforms.length, 0, 'f16 request on an already-f16 graph: zero transform functions');
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

  // Gemma 3 1B — head256 prefill attention is now baked into the conversion
  // config/manifest as the primary path. No model-specific capability transforms needed.
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
      [],
      'Gemma 3 1B: no capability transforms needed (head256 is primary path)'
    );
    equal(r.transforms.length, 0, 'Gemma 3 1B: zero transform functions');
  }

  // Qwen models — fused-Q4 decode/prefill remains the primary execution
  // graph path (baked into conversion config/manifest). Capability transforms
  // still resolve to the empty catch-all on capable GPUs.
  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: { vendor: 'apple', architecture: 'metal-3' },
      },
      { id: 'apple-m3', vendor: 'apple', architecture: 'm-series' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
      }
    );
    deepEqual(r.names, [], 'Qwen 3.5 0.8B on capable GPU: no capability transforms needed');
    equal(r.transforms.length, 0, 'Qwen 3.5 0.8B on capable GPU: zero transform functions');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: { vendor: 'apple', architecture: 'metal-3' },
      },
      { id: 'apple-m3', vendor: 'apple', architecture: 'm-series' },
      {
        activationDtype: 'f16',
        kvDtype: 'f16',
        modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
        requiresF16ActivationNarrowing: true,
      }
    );
    deepEqual(
      r.names,
      ['useQwenF16PrimaryMatmuls'],
      'Qwen 3.5 0.8B on capable GPU with runtime f16 request: should resolve promoted f16 transforms'
    );
    equal(r.transforms.length, 1, 'Qwen 3.5 0.8B runtime f16 request: one transform function');
    equal(r.transforms[0], useQwenF16PrimaryMatmuls,
      'Qwen 3.5 0.8B runtime f16 request: transform is useQwenF16PrimaryMatmuls');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: { vendor: 'amd', architecture: 'rdna-3' },
      },
      { id: 'amd-rdna3', vendor: 'amd', architecture: 'rdna-3' },
      {
        activationDtype: 'f16',
        mathDtype: 'f16',
        accumDtype: 'f16',
        kvDtype: 'f16',
        modelId: 'gemma-4-31b-it-text-q4k-ehf16-af32',
        requiresF16ActivationNarrowing: true,
      }
    );
    deepEqual(
      r.names,
      ['useGemma431BTextF16Activations'],
      'Gemma 4 31B runtime all-f16 request: should resolve the experimental f16 transform'
    );
    equal(r.transforms.length, 1, 'Gemma 4 31B runtime all-f16 request: one transform function');
    equal(r.transforms[0], useGemma431BTextF16Activations,
      'Gemma 4 31B runtime all-f16 request: transform is useGemma431BTextF16Activations');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: { vendor: 'amd', architecture: 'rdna-3' },
      },
      { id: 'amd-rdna3', vendor: 'amd', architecture: 'rdna-3' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        headDim: 256,
        modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        hasDensePrefillProjectionKernel: false,
        hasQ4DecodeProjectionKernel: false,
        hasQ4PrefillProjectionKernel: false,
        hasAvailableQ4PrefillProjectionKernel: false,
      }
    );
    deepEqual(
      r.names,
      ['useHead256SmallPrefillAttention'],
      'Gemma 4 E2B on a capable GPU should resolve the small-tile head256 prefill remap'
    );
    equal(r.transforms.length, 1, 'Gemma 4 E2B on a capable GPU: one transform function');
    equal(r.transforms[0], useHead256SmallPrefillAttention,
      'Gemma 4 E2B on a capable GPU: transform function is useHead256SmallPrefillAttention');
  }

  {
    const r = resolveCapabilityTransforms(
      {
        hasSubgroups: true,
        hasF16: true,
        maxWorkgroupStorageSize: 32768,
        adapterInfo: { vendor: 'amd', architecture: 'rdna3' },
      },
      { id: 'amd-rdna3', vendor: 'amd', architecture: 'rdna3' },
      {
        activationDtype: 'f32',
        kvDtype: 'f16',
        modelId: 'qwen-3-5-2b-q4k-ehaf16',
      }
    );
    deepEqual(r.names, [], 'Qwen 3.5 2B on AMD: no capability transforms needed');
    equal(r.transforms.length, 0, 'Qwen 3.5 2B on AMD: zero transform functions');
  }
}

// ===========================================================================
// Test 14: resolveFinitenessFallbackTransform
// ===========================================================================
{
  // Large-head f16 decode should keep KV on the f16 lane when widening.
  {
    const r = resolveFinitenessFallbackTransform({ activationDtype: 'f16', headDim: 256 });
    ok(r !== null, 'f16 activationDtype: should return a transform');
    equal(
      r.name,
      'widenToF32Activations',
      'large-head f16 activationDtype: name is widenToF32Activations'
    );
    equal(
      r.transform,
      widenToF32Activations,
      'large-head f16 activationDtype: transform is widenToF32Activations'
    );
    equal(
      r.fallbackKvDtype,
      'f16',
      'large-head f16 activationDtype: fallback kvDtype stays on f16'
    );
  }

  // Small-head f16 decode can still use the full correctness fallback.
  {
    const r = resolveFinitenessFallbackTransform({ activationDtype: 'f16', headDim: 64 });
    ok(r !== null, 'small-head f16 activationDtype: should return a transform');
    equal(
      r.name,
      'widenToF32CorrectnessFallback',
      'small-head f16 activationDtype: name is widenToF32CorrectnessFallback'
    );
    equal(
      r.transform,
      widenToF32CorrectnessFallback,
      'small-head f16 activationDtype: transform is widenToF32CorrectnessFallback'
    );
    equal(
      r.fallbackKvDtype,
      'f32',
      'small-head f16 activationDtype: fallback kvDtype widens to f32'
    );
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
// Test 17: narrowToF16Activations preserves explicit f32 tail precision
// ===========================================================================
{
  const graph = {
    kernels: {
      q_proj: { kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4', digest: 'sha256:aaa' },
      final_norm: {
        kernel: 'rmsnorm.wgsl',
        entry: 'main',
        digest: 'sha256:bbb',
        precision: { inputDtype: 'f32', outputDtype: 'f32' },
      },
      lm_head: {
        kernel: 'matmul_gemv_subgroup.wgsl',
        entry: 'main_multicol',
        digest: 'sha256:ccc',
        precision: { inputDtype: 'f32', outputDtype: 'f32' },
      },
    },
    decode: [
      ['q_proj', 'q_proj', 'layer.{L}.self_attn.q_proj'],
    ],
    prefill: [
      ['q_proj', 'q_proj', 'layer.{L}.self_attn.q_proj'],
    ],
    postLayer: [
      ['final_norm', 'final_norm'],
      ['lm_head', 'lm_head', 'lm_head'],
    ],
    policies: { unsupportedPrecision: 'error' },
  };

  const result = narrowToF16Activations(graph, { ...CTX_F16, capabilities: { hasF16: true } });
  ok(result !== null, 'explicit stable tail: narrowToF16Activations should still remap eligible kernels');
  equal(result.kernels.q_proj.kernel, 'matmul_gemv_subgroup_f16a.wgsl',
    'explicit stable tail: layer matmul should narrow to f16 activation');
  deepEqual(result.kernels.q_proj.precision, { inputDtype: 'f16', outputDtype: 'f16' },
    'explicit stable tail: narrowed matmul should declare f16 input/output precision');
  equal(result.kernels.final_norm.kernel, 'rmsnorm.wgsl',
    'explicit stable tail: final_norm should preserve explicit f32 kernel');
  equal(result.kernels.lm_head.kernel, 'matmul_gemv_subgroup.wgsl',
    'explicit stable tail: lm_head should preserve explicit f32 kernel');
}

// ===========================================================================
// Test 17b: narrowToF16Activations remaps head256 prefill attention onto the f16 small kernel
// ===========================================================================
{
  const graph = {
    kernels: {
      attn_small: {
        kernel: 'attention_head256_f16kv.wgsl',
        entry: 'main',
        digest: 'sha256:aaa',
        precision: { kvDtype: 'f16' },
      },
    },
    decode: [],
    prefill: [
      ['attention', 'attn_small'],
    ],
    postLayer: [],
    policies: { unsupportedPrecision: 'error' },
  };

  const result = narrowToF16Activations(graph, { ...CTX_F16, capabilities: { hasF16: true } });
  ok(result !== null, 'head256 prefill: narrowToF16Activations should remap the baked head256 kernel');
  equal(result.kernels.attn_small.kernel, 'attention_small_f16.wgsl',
    'head256 prefill: narrowToF16Activations should remap to attention_small_f16.wgsl');
  equal(result.kernels.attn_small.digest, null,
    'head256 prefill: remapped kernel digest should be cleared');
}

// ===========================================================================
// Test 18: digest is nulled on modified kernels
// ===========================================================================
{
  const graph = {
    kernels: {
      q_decode: {
        kernel: 'fused_matmul_q4_multicol_f16a.wgsl',
        entry: 'main_multicol_f16a',
        digest: 'sha256:qdec',
        precision: { inputDtype: 'f16', outputDtype: 'f16' },
      },
      q_prefill: {
        kernel: 'fused_matmul_q4_batched_f16a.wgsl',
        entry: 'main_batched_f16a',
        digest: 'sha256:qpref',
        precision: { inputDtype: 'f16', outputDtype: 'f16' },
      },
      attn_decode: {
        kernel: 'attention_decode_chunked_f16.wgsl',
        entry: 'main',
        digest: 'sha256:attn',
        precision: { activationDtype: 'f16', kvDtype: 'f16' },
      },
    },
    decode: [
      ['q_proj', 'q_decode'],
      ['attention', 'attn_decode'],
    ],
    prefill: [
      ['q_proj', 'q_prefill'],
    ],
    policies: { unsupportedPrecision: 'error' },
  };

  const result = widenToF32Activations(graph, CTX_F16);
  ok(result !== null, 'q4 widening: widenToF32Activations should widen fused q4 f16a kernels');
  equal(result.kernels.q_decode.kernel, 'fused_matmul_q4.wgsl',
    'q4 widening: decode fused q4 f16a should widen back to fused_matmul_q4.wgsl');
  equal(result.kernels.q_decode.entry, 'main_multicol',
    'q4 widening: decode fused q4 f16a should restore main_multicol entry');
  deepEqual(result.kernels.q_decode.precision, { inputDtype: 'f32', outputDtype: 'f32' },
    'q4 widening: decode fused q4 fallback should declare f32 input/output');
  equal(result.kernels.q_prefill.kernel, 'fused_matmul_q4_batched.wgsl',
    'q4 widening: prefill fused q4 f16a should widen back to fused_matmul_q4_batched.wgsl');
  equal(result.kernels.q_prefill.entry, 'main_batched',
    'q4 widening: prefill fused q4 f16a should restore main_batched entry');
  deepEqual(result.kernels.q_prefill.precision, { inputDtype: 'f32', outputDtype: 'f32' },
    'q4 widening: prefill fused q4 fallback should declare f32 input/output');
}

// ===========================================================================
// Test 19: digest is nulled on modified kernels
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
