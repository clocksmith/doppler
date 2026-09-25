import assert from 'node:assert/strict';
import * as core from 'doppler-gpu';
import * as alias from 'doppler-gpu/runtime';
import * as host from 'doppler-gpu/host';
import * as generation from 'doppler-gpu/generation';
for (const api of [core, alias, host, generation]) {
  for (const name of ['createLayerPartitionPlan', 'validateActivationTensorShape', 'serializeActivationFrame',
    'deserializeActivationFrame', 'createPartitionContinuation', 'comparePartitionExecution']) {
    assert.equal(name in api, false, 'Internal partition helpers are not public runtime operations.');
  }
}
assert.equal(core.openCapsule, alias.openCapsule);
assert.equal(typeof host.openCapsule, 'function');
assert.throws(() => core.openCapsule({}, {}), /explicit ports/);
await assert.rejects(import('doppler-gpu/src/inference/pipelines/text/layer-partition-contract.js'),
  { code: 'ERR_PACKAGE_PATH_NOT_EXPORTED' });
console.log('installed public imports preserve minimal runtime authority');
