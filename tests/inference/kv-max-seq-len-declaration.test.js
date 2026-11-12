import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { requireDeclaredKVMaxSeqLen } from '../../src/config/schema/kvcache.schema.js';
import { applyModelBatchingRuntimeDefaults } from '../../src/inference/pipelines/text/model-load.js';

// The KV-cache length has no code default: schema defaults must not mask the
// manifest declaration, and an undeclared length fails fast.
const runtimeConfig = createDopplerConfig().runtime;
assert.equal(runtimeConfig.inference.session.kvcache.maxSeqLen, undefined);
assert.equal(runtimeConfig.inference.kvcache.maxSeqLen, undefined);

const decodeLoop = { batchSize: 1, stopCheckMode: 'batch', readbackInterval: 1, readbackMode: 'sequential' };
const manifest = (kvcache) => ({
  modelId: 'kv-declaration-fixture',
  inference: { session: { decodeLoop, kvcache: { layout: 'contiguous', kvDtype: 'f16', ...kvcache } } },
});

const fromManifest = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest({ maxSeqLen: 8192 }), null);
assert.equal(fromManifest.inference.session.kvcache.maxSeqLen, 8192);

const overridden = applyModelBatchingRuntimeDefaults(
  { ...runtimeConfig, inference: { ...runtimeConfig.inference, session: {
    ...runtimeConfig.inference.session, kvcache: { ...runtimeConfig.inference.session.kvcache, maxSeqLen: 2048 },
  } } },
  manifest({ maxSeqLen: 8192 }),
  null
);
assert.equal(overridden.inference.session.kvcache.maxSeqLen, 2048);

const undeclared = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest({}), null);
assert.throws(
  () => requireDeclaredKVMaxSeqLen(undeclared.inference.session.kvcache.maxSeqLen),
  /KV cache maxSeqLen is not declared.*conversion config.*runtime profile/
);
for (const invalid of [0, -1, 1.5, '4096', null]) {
  assert.throws(() => requireDeclaredKVMaxSeqLen(invalid), /not declared/);
}
assert.equal(requireDeclaredKVMaxSeqLen(4096), 4096);

console.log('kv-max-seq-len-declaration.test: ok');
