// Installed public contracts with injected selected tokens. No physical GPU claim.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import * as runtime from 'doppler-gpu';
import { computeCanonicalSha256 } from './consumer-evidence.js';

const fixture = JSON.parse(await fs.readFile(new URL('./gpu-generation-fixture.json', import.meta.url)));
const artifacts = new Map(fixture.artifacts.map(([id, bytes]) => [id, Uint8Array.from(bytes)]));
let phases = 0, releases = 0, next = 0;
const session = await runtime.openCapsule(fixture.capsule, {
  trustedSigners: fixture.trustedSigners,
  artifactStore: { readArtifact: async artifact => artifacts.get(artifact.artifactId) },
  device: { getDevice: () => ({ createBuffer: () => ({ destroy() {} }), createCommandEncoder() {}, queue: { writeBuffer() {} } }),
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  programFactory: async () => ({ executionGraphHash: fixture.capsule.program.executionGraphHash,
    getInitialExecutionIdentity: () => fixture.capsule.targetPlans[0].initialExecutionIdentity,
    getActiveAdapterIdentity: () => null, tokenize: () => [0], decodeTokens: ids => ids.join(''),
    getTokenContract: () => ({}), reset() { next = 0; }, close() {},
    createIncrementalDecoder: () => ({ push: String, pendingText: () => '', finish: () => '' }),
    executePhase() { phases++; return { tokenId: next++, vocabSize: 3, get logits() { throw Error('Score readback is forbidden by this fixture'); } }; },
    releaseStepResult(result) { if (result) releases++; },
  }),
});
const request = { schema: 'doppler.capsule-operation-request/v2', operation: { name: 'generate', version: 1 },
  input: { promptTokens: [0] }, options: { maxTokens: 3, maxSeqLen: 16, temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false, seed: 0 }, assignment: null,
  limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 60000 } };
try {
  if (process.argv.includes('--reploid')) {
    const { createDopplerProvider } = await import('reploid/doppler');
    const { resolveConfig } = await import('reploid/config');
    const contract = Object.fromEntries(['modelId', 'capsuleId', 'semanticRoot', 'selectedTargetPlanDigest'].map(key => [key, session[key]]));
    const config = resolveConfig({ overrides: { models: { providerId: 'doppler', contract } } });
    const provider = createDopplerProvider({ config, session, runtime, ownership: 'borrowed', toOperationRequest: () => request });
    try {
      const additions = [];
      const result = await provider.generate([{ role: 'user', content: 'test' }], async text => { additions.push(text); await Promise.resolve(); });
      assert.equal(additions.join(''), '012');
      assert.equal(result.content, '012');
      assert.equal(result.evidence.receipt.outputHash, computeCanonicalSha256(result.evidence.output));
      const controller = new AbortController();
      await assert.rejects(provider.generate([], () => controller.abort(new Error('display cancelled')), { signal: controller.signal }), /display cancelled/);
      assert.equal((await provider.generate([])).content, '012');
    } finally { await provider.close(); }
  } else {
    const accumulator = runtime.createCapsuleStreamAccumulator(request);
    for await (const event of session.executeOperation(request)) accumulator.accept(event);
    const completed = accumulator.finish();
    assert.deepEqual(completed.output.tokenIds, [0, 1, 2]);
    assert.equal(completed.output.text, '012');
    assert.equal(completed.output.completion.stopReason, 'max-tokens');
    assert.equal(completed.receipt.outputHash, computeCanonicalSha256(completed.output));
    const legacy = [];
    for await (const event of session.executeOperation({ ...request, schema: 'doppler.capsule-operation-request/v1' })) legacy.push(event);
    assert.deepEqual(legacy.at(-1).output, completed.output, 'transport version does not change token selection');
  }
  assert.equal(releases, phases);
} finally { await session.close(); }
console.log(JSON.stringify({ passed: true, consumer: process.argv.includes('--reploid') ? 'installed-reploid' : 'standalone',
  runtimeVersion: runtime.DOPPLER_VERSION, phases, releases, physicalExecution: false }));
