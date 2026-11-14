import assert from 'node:assert/strict';
import { createQwen4ExpSourceExecution } from '../../src/converter/qwen4-exp-source-execution.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getKernelPathMatmulVariant } from '../../src/config/kernel-path-loader.js';
import profile from '../../src/config/conversion/qwen3/qwen-3-8-flash-next-text-q4k-ehf16-af32.json' with { type: 'json' };
const tensors = Object.entries({
  'per_layer_token_embd.weight': 'Q4_0', 'token_embd.weight': 'Q4_0', 'output.weight': 'Q6_K',
  'output_hc_down.weight': 'BF16', 'output_hc_up.weight': 'BF16',
  'blk.0.attn_q.weight': 'Q6_K', 'blk.0.attn_k.weight': 'Q8_0',
  'blk.0.hc_attn_down.weight': 'Q4_0', 'blk.0.hc_attn_inject.weight': 'BF16',
  'blk.0.ffn_down_shexp.weight': 'Q8_0', 'blk.0.ple_key.weight': 'Q4_0',
  'blk.1.attn_q.weight': 'Q4_1', 'blk.1.ssm_out.weight': 'Q8_0',
}).map(([name, dtype]) => ({ name, dtype }));
const graph = createQwen4ExpSourceExecution(tensors, 2);
assert.equal(graph.kernels.ple_row_decode.kernel, 'ple_q4_0_to_f32.wgsl');
const compiled = compileExecutionV1({ manifestInference: {
  schema: 'doppler.execution/v1', execution: graph, session: profile.session,
}, modelId: 'source-test', numLayers: 2 });
const path = compiled.runtimeInferencePatch.kernelPath;
for (const [role, layer, decode, prefill] of [
  ['q_proj', 0, 'q6k_gemv', 'q6k_prefill'],
  ['q_proj', 1, 'q4_1_gemv', 'q4_1_prefill'],
  ['k_proj', 0, 'q8_0_gemv', 'q8_0_prefill'],
  ['attn_hc_down', 0, 'q4_0_gemv', 'q4_0_prefill'],
  ['attn_hc_inject', 0, 'f32', 'f32'],
  ['ffn_down', 0, 'q8_0_gemv', 'q8_0_prefill'],
  ['ple_key_proj', 0, 'q4_0_gemv', 'q4_0_prefill'],
  ['linear_out_proj', 1, 'q8_0_gemv', 'q8_0_prefill'],
  ['lm_head', 0, 'q6k_gemv', 'q6k_prefill'],
]) {
  assert.equal(getKernelPathMatmulVariant(role, 'decode', layer, path), decode);
  assert.equal(getKernelPathMatmulVariant(role, 'prefill', layer, path), prefill);
}
assert.throws(() => createQwen4ExpSourceExecution([], 2), /tensor storage facts/);
assert.throws(() => createQwen4ExpSourceExecution(tensors.map(t => t.name === 'output.weight' ? {...t,dtype:'IQ2_XXS'} : t), 2), /Unsupported.*dtype/);
assert.throws(() => createQwen4ExpSourceExecution(tensors.filter(t => t.name !== 'output.weight'), 2), /Missing.*output/);
console.log('qwen4-exp-source-execution.test: ok');
