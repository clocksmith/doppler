import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createDopplerRun, createCapsuleStreamAccumulator, resolveEmbeddingBatchSemantics } from 'doppler-gpu';
import { computeCanonicalSha256 } from './consumer-evidence.js';

// Installed public API contract, with synthetic device and execution components.
const fixture = JSON.parse(await fs.readFile(new URL('./embedding-fixture.json', import.meta.url)));
const bytes = new Map(fixture.artifacts.map(([id, values]) => [id, new Uint8Array(values)]));
const manifest = JSON.parse(new TextDecoder().decode(bytes.get('manifest')));
const semantics = resolveEmbeddingBatchSemantics(manifest);
assert.equal(semantics.state, 'reset-per-text');
assert.equal(semantics.operation.name, 'embed');
assert(Object.isFrozen(semantics.operation));
assert.throws(() => resolveEmbeddingBatchSemantics({ ...manifest, inference: {} }), /embeddingPostprocessor/);
let executions = 0;
let closes = 0;
const manifestHash = fixture.capsule.artifacts.find(artifact => artifact.artifactId === 'manifest').hash;
const runtime = createDopplerRun({
  device: { getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }),
    getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024, hasF16: false, hasSubgroups: false }) },
  artifactStore: { readArtifact: async artifact => bytes.get(artifact.artifactId) },
  trustedSigners: fixture.trustedSigners,
  async programFactory() {
    return { async embed(text) {
      executions += 1;
      const output = { embedding: [0, 0, 0, 1], tokens: [1, 2], seqLen: 2, embeddingMode: 'last' };
      const backendIdentity = { backend: 'webgpu', adapter: { vendor: 'synthetic-test' } };
      const executionIdentity = { schema: 'doppler.resolved-execution-identity/v1', backendIdentity };
      return { schema: 'doppler_embedding_evidence/v1', ...output,
        inputHash: computeCanonicalSha256({ text }), outputHash: computeCanonicalSha256(output),
        backendIdentity, backendIdentityHash: computeCanonicalSha256(backendIdentity), executionIdentity,
        resolution: { schema: 'doppler.resolution-identity/v1', resolvedArtifactVariantId: manifestHash,
          resolvedExecutionId: computeCanonicalSha256(executionIdentity) } };
    }, async close() { closes += 1; } };
  },
});
await assert.rejects(runtime.openCapsule(fixture.capsule, { maxVerifiedBackingBytes: 0 }), /maxVerifiedBackingBytes/);
await assert.rejects(runtime.openCapsule(fixture.capsule, { artifactHashBackend: 'unchecked' }), /artifactHashBackend/);
const backingBudget = fixture.capsule.artifacts.reduce((sum, artifact) => sum + artifact.sizeBytes, 0);
const session = await runtime.openCapsule(fixture.capsule, { maxVerifiedBackingBytes: backingBudget, artifactHashBackend: 'node-crypto' });
try {
  const request = { application: fixture.capsule.release.application, text: 'Installed text embedding.' };
  const result = await session.embed(request);
  assert.equal(result.receipt.operation, 'embed');
  assert.equal(result.receipt.capsule.capsuleId, fixture.capsule.capsuleId);
  assert.equal(result.receipt.inputHash, computeCanonicalSha256(request));
  assert.equal(Object.isFrozen(result.embedding), true);
  await assert.rejects(session.embed({ ...request, options: { embeddingMode: 'mean' } }), /only signal/);
  await assert.rejects(session.encodeSequence('MKT'), /not qualified.*encodeSequence/);
  const controller = new AbortController();
  controller.abort(new Error('installed cancellation'));
  await assert.rejects(session.embed({ ...request, options: { signal: controller.signal } }), /installed cancellation/);
  assert.equal(executions, 1);
  const job = { schema: 'doppler.capsule-operation-request/v1', operation: { name: 'embed', version: 1 },
    input: { texts: ['Installed batch item.'], application: request.application }, options: {},
    assignment: { jobId: 'installed-embedding', attempt: 1 },
    limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 60000 } };
  const events = [];
  for await (const event of session.executeOperation(job)) events.push(event);
  assert.deepEqual(events.map(event => event.status), ['partial', 'completed']);
  assert.equal(events[1].receipt.assignmentHash, computeCanonicalSha256(job.assignment));
  assert.equal(events[1].output.embeddings[0].receipt.operation, 'embed');
  assert.equal(executions, 2);
  const denied = structuredClone(job);
  denied.input.application.applicationId = 'not-authorized';
  await assert.rejects(async () => {
    for await (const event of session.executeOperation(denied)) assert.fail(`Unauthorized event ${event.status}`);
  }, /application identity/);
  assert.equal(executions, 2);
  const localJob = { ...job, assignment: null };
  const direct = [];
  for await (const event of session.executeOperation(localJob)) direct.push(event);
  assert.equal(direct.at(-1).status, 'completed');
  assert.equal(executions, 3);
  const incrementalJob = { ...localJob, schema: 'doppler.capsule-operation-request/v2',
    input: { ...localJob.input, texts: ['First incremental item.', 'Second incremental item.'] } };
  const accumulator = createCapsuleStreamAccumulator(incrementalJob);
  const deltas = [];
  for await (const event of session.executeOperation(incrementalJob)) deltas.push(event);
  deltas.forEach(event => accumulator.accept(event));
  assert.equal(accumulator.finish().output.embeddings.length, 2);
  assert.deepEqual(deltas.slice(0, -1).map(event => event.delta.itemIndex), [0, 1]);
  assert(deltas.slice(0, -1).every(event => !Object.hasOwn(event, 'output')));
  assert.equal(executions, 5);
  assert.equal(session.closed, false);
  console.log('Installed Capsule direct and incremental operations passed (synthetic).');
} finally { await session.close(); }
assert.equal(closes, 1);
console.log('Installed Capsule embedding smoke passed (synthetic; no physical qualification).');
