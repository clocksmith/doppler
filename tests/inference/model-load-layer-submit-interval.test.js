import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();
const { applyModelBatchingRuntimeDefaults } = await import('../../src/inference/pipelines/text/model-load.js');

function resolve(layerSubmitInterval, runtimeValue) {
  const decodeLoop = { batchSize: 1, stopCheckMode: 'per-token', readbackInterval: 1,
    readbackMode: 'sequential', layerSubmitInterval };
  const manifest = { modelId: 'layer-submit-validation', inference: { session: { decodeLoop } } };
  const runtime = { inference: { session: runtimeValue === undefined ? {} :
    { decodeLoop: { layerSubmitInterval: runtimeValue } } } };
  return applyModelBatchingRuntimeDefaults(runtime, manifest, null).inference.session.decodeLoop;
}

for (const value of [0, -1, 2.5, '4', Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1]) {
  assert.throws(() => resolve(value), /layerSubmitInterval must be a positive integer or null/);
  assert.throws(() => resolve(null, value), /layerSubmitInterval must be a positive integer or null/);
}
for (const value of [null, 1, 4, Number.MAX_SAFE_INTEGER]) {
  assert.equal(resolve(value).layerSubmitInterval, value);
}
assert.equal(resolve(4, 1).layerSubmitInterval, 1);
assert.equal(resolve(4, null).layerSubmitInterval, null);
assert.equal(resolve(undefined).layerSubmitInterval, undefined);
console.log('model-load-layer-submit-interval: passed');
