// Installed API acceptance with injected logits; not physical model qualification.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { openCapsule, DOPPLER_VERSION } from 'doppler-gpu';
import { computeCanonicalSha256 } from './consumer-evidence.js';

const fixture = JSON.parse(await fs.readFile(new URL('./generation-fixture.json', import.meta.url)));
const artifacts = new Map(fixture.artifacts.map(([id, bytes]) => [id, Uint8Array.from(bytes)]));
let executions = 0, releases = 0, closes = 0;
const session = await openCapsule(fixture.capsule, {
  trustedSigners: fixture.trustedSigners,
  artifactStore: { readArtifact: async artifact => artifacts.get(artifact.artifactId) },
  device: { getDevice: () => ({ createBuffer: () => ({ destroy() {} }), createCommandEncoder() {}, queue: { writeBuffer() {} } }),
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  programFactory: async () => ({ executionGraphHash: fixture.capsule.program.executionGraphHash,
    tokenize: () => [0, 2], decodeTokens: ids => ids.join(','), getTokenContract: () => ({}),
    reset() {}, getActiveAdapterIdentity: () => null,
    async executePhase() { executions++; return { logits: new Float32Array([3, 2, 1]) }; },
    releaseStepResult(result) { if (result) releases++; }, close() { closes++; },
  }),
});
const request = { schema: 'doppler.capsule-operation-request/v1', operation: { name: 'generate', version: 1 },
  input: { promptTokens: [0, 2] }, options: { maxTokens: 3, maxSeqLen: 16, temperature: 0, topP: 1, topK: 0,
    repetitionPenalty: 1, repetitionPenaltyWindow: 1, presencePenalty: 2, useChatTemplate: false, seed: 0 },
  assignment: null, limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 60000 } };
try {
  const events = [];
  for await (const event of session.executeOperation(request)) events.push(event);
  assert.deepEqual(events.map(event => event.status), ['partial', 'partial', 'partial', 'completed']);
  const { output, receipt } = events.at(-1);
  assert.deepEqual(output.tokenIds, [0, 1, 0]);
  assert.equal(output.text, '0,1,0');
  assert.equal(output.completion.stopReason, 'max-tokens');
  assert.equal(receipt.outputHash, computeCanonicalSha256(output));
  assert.equal(receipt.runtimeVersion, DOPPLER_VERSION);
  const controller = new AbortController();
  await assert.rejects(async () => {
    for await (const event of session.executeOperation(request, { signal: controller.signal })) {
      assert.equal(event.status, 'partial');
      controller.abort(new Error('consumer cancelled'));
    }
  }, /consumer cancelled/);
  assert.equal(executions, 4);
  assert.equal(releases, executions);
} finally { await session.close(); }
assert.equal(closes, 1);
console.log('Installed Capsule generation passed (injected logits; no physical qualification).');
