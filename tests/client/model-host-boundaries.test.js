import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import * as legacyHost from '../../src/client/runtime/index.js';
import * as host from '../../src/client/model-host/index.js';
import * as legacySession from '../../src/client/runtime/model-session.js';
import * as session from '../../src/client/model-host/model-session.js';

for (const [legacy, current] of [[legacyHost, host], [legacySession, session]]) {
  assert.deepEqual(Object.keys(legacy), Object.keys(current));
  for (const key of Object.keys(current)) assert.equal(legacy[key], current[key]);
}
assert.throws(() => host.createDopplerRuntimeService({}), /ensureWebGPUAvailable/);
const service = host.createDopplerRuntimeService({ ensureWebGPUAvailable: async () => {} });
await assert.rejects(service.openPack('missing-resolver'), /no Pack source resolver/);
assert.throws(() => session.assertSupportedGenerationOptions({ stopTokens: [1] }), /stopSequences/);

let resume;
const setup = new Promise(resolve => { resume = resolve; });
let observedOptions;
let setupCalls = 0;
const applicationPolicy = { acceptedTargetPlanDigests: [`sha256:${'a'.repeat(64)}`],
  preferredTargetPlanDigests: [`sha256:${'a'.repeat(64)}`], requiredOperations: ['rerank'] };
const expectedPolicy = structuredClone(applicationPolicy);
const delayedHost = host.createDopplerRuntimeService({
  ensureWebGPUAvailable: async () => { setupCalls += 1; await setup; },
  resolvePackInput: async (source, options) => { observedOptions = options; throw new Error('stop before GPU creation'); },
});
const opening = delayedHost.openPack('delayed-source', applicationPolicy);
applicationPolicy.acceptedTargetPlanDigests.length = 0;
applicationPolicy.preferredTargetPlanDigests[0] = `sha256:${'b'.repeat(64)}`;
applicationPolicy.requiredOperations[0] = 'generate';
resume();
await assert.rejects(opening, /stop before GPU creation/);
assert.deepEqual(observedOptions, expectedPolicy, 'host preparation cannot change application selection authority');
assert.equal(Object.isFrozen(observedOptions.acceptedTargetPlanDigests), true);
await assert.rejects(delayedHost.openPack('invalid-policy', { requiredOperations: null }), /requiredOperations/);
assert.equal(setupCalls, 1, 'invalid selection policy must fail before host preparation');

const policy = JSON.parse(await fs.readFile('tools/policies/source-architecture-policy.json', 'utf8'));
const runtime = policy.constitutionalImportGraphs.find(rule => rule.domain === 'runtime');
for (const forbidden of ['client/model-host/', 'client/runtime/index.js', 'client/runtime/model-session.js']) {
  assert.ok(runtime.forbiddenPathPrefixes.includes(forbidden), `Pack core must exclude ${forbidden}`);
}
const evidence = policy.constitutionalImportGraphs.find(rule => rule.domain === 'host-evidence');
assert.deepEqual(evidence.entryPoints, ['client/model-host/model-evidence.js']);
for (const forbidden of ['gpu/', 'loader/', 'inference/', 'converter/', 'tooling/']) {
  assert.ok(evidence.forbiddenPathPrefixes.includes(forbidden));
}
assert.equal(policy.softLimitReviews['client/runtime/model-session.js'], undefined);
console.log('✔ model-host-boundaries.test.js passed');
