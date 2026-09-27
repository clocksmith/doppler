import assert from 'node:assert/strict';
import * as core from 'doppler-gpu';
import * as alias from 'doppler-gpu/runtime';
import * as host from 'doppler-gpu/host';
import * as generation from 'doppler-gpu/generation';
import * as partitions from 'doppler-gpu/partitions';
for (const api of [core, alias, host, generation]) {
  for (const name of ['createLayerPartitionPlan', 'validateActivationTensorShape', 'serializeActivationFrame',
    'deserializeActivationFrame', 'createPartitionContinuation', 'comparePartitionExecution']) {
    assert.equal(name in api, false, 'Internal partition helpers are not public runtime operations.');
  }
}
assert.equal(core.openCapsule, alias.openCapsule);
assert.equal(typeof host.openCapsule, 'function');
const plan = partitions.createLayerPartitionPlan({ modelId: 'installed-contract',
  numLayers: 4, hiddenSize: 2, vocabSize: 8, splitLayer: 2, activationDtype: 'f32' });
assert.deepEqual(plan.partitions.map(group => group.layerRange), [[0, 1], [2, 3]]);
const frame = partitions.serializeActivationFrame({ shape: [1, 1, 2], dtype: 'f32',
  data: new Float32Array([1, 2]), seqOffset: 0, step: 0 });
assert.deepEqual([...partitions.deserializeActivationFrame(frame).tensorData], [1, 2]);
assert.throws(() => core.openCapsule({}, {}), /explicit ports/);
await assert.rejects(import('doppler-gpu/src/inference/pipelines/text/layer-partition-contract.js'),
  { code: 'ERR_PACKAGE_PATH_NOT_EXPORTED' });
console.log('installed public imports preserve minimal runtime authority');
