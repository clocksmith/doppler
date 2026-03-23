import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  removeSubgroups,
  widenToF32Activations,
  swapPrefillAttention,
  widenProjectionWeightsToF32,
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

/** The real execution graph from the gemma3-1b-q4k conversion config. */
const REAL_GRAPH = conversionConfig.execution;

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
  for (const step of steps) {
    const kernelKey = step[1];
    const entry = graph.kernels[kernelKey];
    if (entry) {
      files.add(entry.kernel);
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

  // Prefill steps must be UNCHANGED
  deepEqual(result.prefill, input.prefill, 'prefill steps must be unchanged after removeSubgroups');

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
  // Build a graph with attention_streaming_f16kv.wgsl in prefill
  const graph = structuredClone(REAL_GRAPH);
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
// Test 6: widenProjectionWeightsToF32 transform
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
// Test 7: composeTransforms — only first applies
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
// Test 8: composeTransforms with multiple applicable transforms
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
// Test 9: resolveCapabilityTransforms
// ===========================================================================
{
  const platform = { id: 'test', vendor: 'test', architecture: 'test' };
  const graphCtx = { activationDtype: 'f32', kvDtype: 'f16' };

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
}

// ===========================================================================
// Test 10: resolveFinitenessFallbackTransform
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
// Test 11: Snapshot test — removeSubgroups produces expected kernel files
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

  // Prefill must be unchanged
  deepEqual(result.prefill, input.prefill, 'snapshot: prefill must be unchanged by removeSubgroups');

  // Prefill projection kernels still use the original tiled kernel
  const prefillProjFiles = new Set();
  for (const step of result.prefill) {
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = result.kernels[step[1]];
      if (entry) prefillProjFiles.add(entry.kernel);
    }
  }
  equal(prefillProjFiles.size, 1, 'snapshot: all prefill projections should use the same kernel file');
  ok(prefillProjFiles.has('matmul_f16w_f32a.wgsl'),
    'snapshot: prefill projections should use matmul_f16w_f32a.wgsl');
}

// ===========================================================================
// Test 12: Purity/immutability test
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

  // Apply widenProjectionWeightsToF32
  const projGraph = buildF16WeightProjectionGraph();
  const projClone = structuredClone(projGraph);
  widenProjectionWeightsToF32(projGraph, CTX_F32);
  deepEqual(projGraph, projClone, 'purity: widenProjectionWeightsToF32 must not mutate input');
}

// ===========================================================================
// Test 13: digest is nulled on modified kernels
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
