import assert from 'node:assert/strict';
import fs from 'node:fs';

const path = 'src/config/conversion/qwen3/qwen-3-5-9b-q4k-ehaf16.json';
const config = JSON.parse(fs.readFileSync(path, 'utf8'));
const identity = config.manifest?.artifactIdentity;
const exclusions = config.quantization?.modulesToNotConvert;
const layerTypes = config.inference?.layerPattern?.layerTypes;

assert.equal(config.output?.modelBaseId, 'qwen-3-5-9b-q4k-ehaf16');
assert.equal(config.output?.textOnly, true);
assert.equal(config.quantization?.weights, 'q4k');
assert.equal(config.quantization?.embeddings, 'f16');
assert.equal(config.quantization?.lmHead, 'q4k');
assert.equal(config.quantization?.q4kLayout, 'row');
assert.deepEqual(exclusions, [
  'linear_attn.conv1d.weight',
  'linear_attn.in_proj_a.weight',
  'linear_attn.in_proj_b.weight',
  'linear_attn.in_proj_qkv.weight',
  'linear_attn.in_proj_z.weight',
  'linear_attn.out_proj.weight',
]);

assert.equal(identity?.sourceCheckpointId, 'Qwen/Qwen3.5-9B');
assert.equal(identity?.sourceRepo, 'Qwen/Qwen3.5-9B');
assert.equal(identity?.sourceRevision, 'c202236235762e1c871ad0ccb60c8ee5ba337b9a');
assert.equal(identity?.artifactCompleteness, 'complete');
assert.equal(identity?.shardSetHash, undefined);

assert.equal(config.inference?.attention?.queryPreAttnScalar, 256);
assert.equal(config.inference?.attention?.attentionOutputGate, true);
assert.equal(config.inference?.output?.tieWordEmbeddings, false);
assert.equal(layerTypes?.length, 32);
for (let index = 0; index < layerTypes.length; index += 1) {
  assert.equal(
    layerTypes[index],
    (index + 1) % 4 === 0 ? 'full_attention' : 'linear_attention',
    `layer ${index} must match the source checkpoint hybrid schedule`
  );
}

assert.deepEqual(config.session?.compute?.defaults, {
  activationDtype: 'f32',
  mathDtype: 'f32',
  accumDtype: 'f32',
  outputDtype: 'f32',
});
assert.equal(config.session?.kvcache?.kvDtype, 'f16');
assert.equal(config.session?.decodeLoop?.disableCommandBatching, false);
assert.equal(config.execution?.inlineKernelPath, true);
assert.equal(config.execution?.kernels?.q4_decode?.kernel, 'fused_matmul_q4.wgsl');
assert.equal(config.execution?.kernels?.tiled?.kernel, 'matmul_f16w_f32a.wgsl');

console.log('qwen35-9b-q4k-conversion-config-contract.test: ok');
