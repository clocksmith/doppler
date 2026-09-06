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
