import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function assertProjectionKernel(graph, phase, expectedKernelKey) {
  const projectionOps = new Set([
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
  ]);

  for (const step of graph[phase]) {
    if (!projectionOps.has(step[0])) {
      continue;
    }
    assert.equal(
      step[1],
      expectedKernelKey,
      `${phase}: op "${step[0]}" must use kernel key "${expectedKernelKey}"`
    );
  }
}

function assertGemma3HybridAttention(inference, label) {
  assert.equal(inference.attention?.slidingWindow, 512, `${label}: Gemma 3 sliding window`);
  assert.equal(inference.rope?.ropeLocalTheta, 10000, `${label}: Gemma 3 local RoPE theta`);
  assert.equal(inference.layerPattern?.type, 'every_n', `${label}: Gemma 3 layer pattern type`);
  assert.equal(inference.layerPattern?.period, 6, `${label}: Gemma 3 layer pattern period`);
  assert.equal(inference.layerPattern?.offset, null, `${label}: Gemma 3 layer pattern offset`);
}

const conversionConfig = readJson('src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json');
const localManifest = readJson('models/local/gemma-3-1b-it-q4k-ehf16-af32/manifest.json');

assertGemma3HybridAttention(conversionConfig.inference, 'conversion');
assertGemma3HybridAttention(localManifest.inference, 'manifest');

for (const [label, graph] of [
  ['conversion', conversionConfig.execution],
  ['manifest', localManifest.inference.execution],
]) {
  assert.equal(graph.kernels.q4_decode, undefined, `${label}: q4_decode should not be pinned`);
  assert.equal(graph.kernels.gemv.kernel, 'matmul_gemv_subgroup.wgsl', `${label}: gemv kernel`);
  assert.equal(graph.kernels.gemv.entry, 'main_vec4', `${label}: gemv entry`);
  assert.equal(graph.kernels.tiled.kernel, 'matmul_f16w_f32a_tiled.wgsl', `${label}: tiled kernel`);
  assert.equal(graph.kernels.tiled.entry, 'main', `${label}: tiled entry`);
  assert.equal(graph.kernels.q4_prefill_shared.kernel, 'fused_matmul_q4_batched_multicol_shared.wgsl', `${label}: q4_prefill_shared kernel`);
  assert.equal(graph.kernels.q4_prefill_shared.entry, 'main', `${label}: q4_prefill_shared entry`);
  assert.equal(graph.kernels.attn_small.kernel, 'attention_small_f16kv.wgsl', `${label}: attn_small kernel`);
  assert.equal(graph.kernels.attn_head256.kernel, 'attention_head256_f16kv.wgsl', `${label}: attn_head256 kernel`);
  assertProjectionKernel(graph, 'decode', 'gemv');
  assertProjectionKernel(graph, 'prefill', 'tiled');
  const prefillAttention = graph.prefill.find((step) => step[0] === 'attention');
  assert.ok(prefillAttention, `${label}: prefill attention step must exist`);
  assert.equal(prefillAttention[1], 'attn_head256', `${label}: prefill attention should use attn_head256`);

  const lmHeadPrefill = graph.postLayer.find((step) => step[0] === 'lm_head_prefill');
  assert.ok(lmHeadPrefill, `${label}: lm_head_prefill step must exist`);
  assert.equal(lmHeadPrefill[1], 'tiled', `${label}: lm_head_prefill should stay on tiled f16 weights`);
}

console.log('gemma3-q4k-fused-path-contract.test: ok');
