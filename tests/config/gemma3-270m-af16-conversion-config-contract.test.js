import assert from 'node:assert/strict';
import fs from 'node:fs';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const AF32_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af16';
const AF32_CONFIG_PATH = `src/config/conversion/gemma3/${AF32_MODEL_ID}.json`;
const AF16_CONFIG_PATH = `src/config/conversion/gemma3/${AF16_MODEL_ID}.json`;

const af32Config = readJson(AF32_CONFIG_PATH);
const af16Config = readJson(AF16_CONFIG_PATH);

assert.equal(af16Config.modelType, 'transformer');
assert.equal(af32Config.output?.modelBaseId, AF32_MODEL_ID);
assert.equal(af16Config.output?.modelBaseId, AF16_MODEL_ID);

assert.equal(af32Config.quantization?.weights, 'q4k');
assert.equal(af16Config.quantization?.weights, 'q4k');
assert.equal(af32Config.quantization?.embeddings, 'f16');
assert.equal(af16Config.quantization?.embeddings, 'f16');
assert.equal(af32Config.quantization?.lmHead, 'f16');
assert.equal(af16Config.quantization?.lmHead, 'f16');
assert.equal(af32Config.quantization?.q4kLayout, 'row');
assert.equal(af16Config.quantization?.q4kLayout, 'row');
assert.equal(af32Config.quantization?.computePrecision, 'f32');
assert.equal(af16Config.quantization?.computePrecision, 'f16');

assert.equal(
  af16Config.manifest?.artifactIdentity?.sourceCheckpointId,
  af32Config.manifest?.artifactIdentity?.sourceCheckpointId,
  'af16 repro must use the same source checkpoint as the fair af32 lane'
);
assert.equal(
  af16Config.manifest?.artifactIdentity?.weightPackId,
  af32Config.manifest?.artifactIdentity?.weightPackId,
  'af16 repro must use the same Q4K weight-pack identity as the fair af32 lane'
);
assert.notEqual(
  af16Config.manifest?.artifactIdentity?.manifestVariantId,
  af32Config.manifest?.artifactIdentity?.manifestVariantId,
  'af16 repro must carry a distinct manifest variant identity'
);

assert.deepEqual(af16Config.session?.compute?.defaults, {
  activationDtype: 'f16',
  mathDtype: 'f16',
  accumDtype: 'f16',
  outputDtype: 'f16',
});
assert.equal(af16Config.session?.kvcache?.kvDtype, 'f16');
assert.equal(af16Config.session?.perLayerInputs?.rowCache?.decodedDtype, 'f16');
assert.equal(af16Config.session?.perLayerInputs?.hotCache?.outputDtype, 'f16');

assert.equal(af16Config.execution?.kernels?.final_norm_stable?.kernel, 'rmsnorm.wgsl');
assert.equal(af16Config.execution?.kernels?.final_norm_stable?.precision?.inputDtype, 'f32');
assert.equal(af16Config.execution?.kernels?.final_norm_stable?.precision?.outputDtype, 'f32');
assert.equal(af16Config.execution?.kernels?.lm_head_gemv_stable?.kernel, 'matmul_gemv_subgroup.wgsl');
assert.equal(af16Config.execution?.kernels?.lm_head_gemv_stable?.precision?.inputDtype, 'f32');
assert.equal(af16Config.execution?.kernels?.lm_head_gemv_stable?.precision?.outputDtype, 'f32');
assert.equal(af16Config.execution?.kernels?.lm_head_prefill_stable?.kernel, 'matmul_f16w_f32a_tiled.wgsl');
assert.equal(af16Config.execution?.kernels?.lm_head_prefill_stable?.precision?.inputDtype, 'f32');
assert.equal(af16Config.execution?.kernels?.lm_head_prefill_stable?.precision?.outputDtype, 'f32');
assert.equal(af32Config.execution?.kernels?.attn_head256?.kernel, 'attention_head256_f16kv.wgsl');
assert.equal(af16Config.execution?.kernels?.attn_head256?.kernel, 'attention_head256_f16kv.wgsl');
assert.deepEqual(
  af32Config.execution?.prefill?.find((step) => step[0] === 'attention'),
  ['attention', 'attn_head256'],
  'af32 fair lane must use the fixed headDim=256 prefill attention kernel'
);
assert.deepEqual(
  af16Config.execution?.prefill?.find((step) => step[0] === 'attention'),
  ['attention', 'attn_head256'],
  'af16 repro lane must use the same fixed headDim=256 prefill attention kernel'
);

assert.deepEqual(
  af16Config.execution?.postLayer,
  [
    ['final_norm', 'final_norm_stable'],
    ['lm_head', 'lm_head_gemv_stable', 'lm_head'],
    ['lm_head_prefill', 'lm_head_prefill_stable', 'lm_head'],
    ['sample', 'sample'],
  ],
  'af16 repro must declare the f16-to-f32 logits boundary explicitly'
);

console.log('gemma3-270m-af16-conversion-config-contract.test: ok');
