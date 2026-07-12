import assert from 'node:assert/strict';
import fs from 'node:fs';

const path = 'src/config/conversion/qwen3/qwen-3-5-9b-f16-af32.json';
const config = JSON.parse(fs.readFileSync(path, 'utf8'));
const identity = config.manifest?.artifactIdentity;
const layerTypes = config.inference?.layerPattern?.layerTypes;
const executionJson = JSON.stringify(config.execution);

assert.equal(config.output?.modelBaseId, 'qwen-3-5-9b-f16-af32');
assert.equal(config.output?.textOnly, true);
assert.equal(config.quantization?.weights, 'f16');
assert.equal(config.quantization?.embeddings, 'f16');
assert.equal(config.quantization?.lmHead, 'f16');
assert.equal(config.quantization?.modulesToNotConvert, undefined);

assert.equal(identity?.sourceCheckpointId, 'Qwen/Qwen3.5-9B');
assert.equal(identity?.sourceRepo, 'Qwen/Qwen3.5-9B');
assert.equal(identity?.sourceRevision, 'c202236235762e1c871ad0ccb60c8ee5ba337b9a');
assert.equal(identity?.artifactCompleteness, 'complete');
assert.equal(identity?.shardSetHash, undefined);

assert.equal(config.inference?.attention?.queryPreAttnScalar, 256);
assert.equal(config.inference?.attention?.attentionOutputGate, true);
assert.equal(config.inference?.attention?.outputGateType, undefined);
assert.equal(config.inference?.output?.tieWordEmbeddings, false);
assert.equal(config.inference?.rope?.partialRotaryFactor, 0.25);
assert.deepEqual(config.inference?.rope?.mropeSection, [11, 11, 10]);

assert.equal(config.inference?.layerPattern?.type, 'custom');
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
assert.equal(config.session?.decodeLoop?.disableCommandBatching, true);
assert.equal(config.execution?.inlineKernelPath, true);
assert.equal(config.execution?.kernels?.gemv?.kernel, 'matmul_gemv_subgroup.wgsl');
assert.equal(config.execution?.kernels?.tiled?.kernel, 'matmul_f16w_f32a.wgsl');
assert.equal(config.execution?.kernels?.lm_head_gemv?.kernel, 'matmul_gemv_subgroup.wgsl');
assert.doesNotMatch(executionJson, /q4_/);

console.log('qwen35-9b-conversion-config-contract.test: ok');
