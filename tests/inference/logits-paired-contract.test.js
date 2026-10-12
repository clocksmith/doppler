import assert from 'node:assert/strict';
import { getLogitsConfig } from '../../src/inference/pipelines/text/generator/logits-config.js';
import { createLogitsInputTensor, resolveLogitsNormWeight, coerceTensorDtype,
  recordGreedyLmHeadArgmaxGPU } from '../../src/inference/pipelines/text/logits/gpu-executor.js';

const state = {
  modelConfig: { hiddenSize: 4, vocabSize: 4, normalizationType: 'rmsnorm' },
  runtimeConfig: { inference: { largeWeights: {} }, shared: { debug: { probes: null } } },
  executionPlanState: { primaryPlan: { activationDtype: 'f32', activationPrecision: 'two-f32',
    activationComponents: 2, activationElementStrideBytes: 8 } },
};
const config = getLogitsConfig(state);
assert.equal(config.activationPrecision, 'two-f32', 'final logits must retain the active plan representation');
assert.equal(config.activationComponents, 2);
assert.equal(config.activationElementStrideBytes, 8);
const buffer = { size: 64 };
const input = createLogitsInputTensor(buffer, 'f32', 2, config);
assert.equal(input.buffer, buffer);
assert.equal(input.precision, 'two-f32');
assert.deepEqual(input.shape, [2, 4]);
for (const change of [{ activationComponents: undefined }, { activationPrecision: undefined },
  { activationElementStrideBytes: 4 }, { normalizationType: 'layernorm' }]) {
  assert.throws(() => createLogitsInputTensor(buffer, 'f32', 2, { ...config, ...change }));
}
assert.throws(() => createLogitsInputTensor({ size: 32 }, 'f32', 2, config), /declared bytes/);
assert.throws(() => createLogitsInputTensor(buffer, 'f16', 2, config), /eight-byte stride/);
await assert.rejects(coerceTensorDtype(input, 'f16'), /must remain paired/);
assert.equal(await coerceTensorDtype(input, 'f32'), input);
await assert.rejects(recordGreedyLmHeadArgmaxGPU(null, buffer, 1, {}, config, {}), /full recorded logits/);
const normBuffer = { size: 16 };
const norm = resolveLogitsNormWeight(new Float32Array(4), normBuffer, input);
assert.equal(norm.dtype, 'f32');
assert.equal(norm.buffer, normBuffer);
assert.equal(norm.precision, undefined, 'normalization weights retain their scalar encoding');
const scalar = createLogitsInputTensor({ size: 16 }, 'f16', 2, { hiddenSize: 4 });
assert.equal(scalar.precision, undefined);
assert.equal(resolveLogitsNormWeight(null, normBuffer, scalar), normBuffer);
console.log('logits-paired-contract: passed');
