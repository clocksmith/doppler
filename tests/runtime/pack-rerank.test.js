import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import {
  TEST_PACK_AUTHORITY,
  TEST_PACK_PUBLIC_KEY,
  createSignedPackFixture,
} from '../helpers/pack-v2-fixture.js';

const fixture = await createSignedPackFixture();
const digest = (character) => `sha256:${character.repeat(64)}`;
const rerankCalls = [];
const evidence = {
  schema: 'doppler_rerank_evidence/v1',
  query: 'What runs locally?',
  documents: ['Doppler runs WebGPU locally.', 'A remote API runs elsewhere.'],
  scores: [],
  ranking: [],
  inputHash: digest('1'),
  outputHash: digest('2'),
  resolution: {},
  executionIdentity: {},
  backendIdentity: {},
  backendIdentityHash: digest('3'),
  stats: null,
};
const program = {
  executionGraphHash: fixture.pack.program.executionGraphHash,
  tokenize() { return []; },
  decodeTokens() { return ''; },
  getTokenContract() { return {}; },
  reset() {},
  async rerank(request) {
    rerankCalls.push(request);
    return evidence;
  },
  async executePhase() { throw new Error('not reached'); },
  releaseStepResult() {},
  async close() {},
};
const events = [];
const runtime = createDopplerRuntime({
  device: {
    getDevice() {
      return {
        limits: { maxBufferSize: 1024 },
        createBuffer() { return { destroy() {} }; },
        createCommandEncoder() {},
        queue: { writeBuffer() {} },
      };
    },
    getProfile() {
      return { surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 };
    },
  },
  artifactStore: fixture.artifactStore,
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  observer: { observe(event) { events.push(event); } },
  async programFactory() { return program; },
});

const session = await runtime.openPack(fixture.pack);
const application = structuredClone(fixture.pack.release.application);
const request = {
  application,
  query: evidence.query,
  documents: evidence.documents,
};
const receipt = await session.rerank(request);
assert.equal(receipt.schema, 'doppler.pack-rerank-receipt/v1');
assert.equal(receipt.pack.packId, fixture.pack.packId);
assert.equal(receipt.pack.semanticRoot, fixture.pack.semanticRoot);
assert.equal(receipt.target.targetId, session.selectedTargetId);
assert.deepEqual(receipt.application, fixture.pack.release.application);
assert.equal(receipt.evidence, evidence);
assert.match(receipt.receiptDigest, /^sha256:[0-9a-f]{64}$/);
assert.deepEqual(rerankCalls, [{
  query: evidence.query,
  documents: evidence.documents,
  options: undefined,
}]);
assert.equal(events.at(-1).type, 'pack-rerank-complete');
assert.equal(events.at(-1).receiptDigest, receipt.receiptDigest);

const mismatched = structuredClone(request);
mismatched.application.workload.digest = digest('f');
await assert.rejects(
  session.rerank(mismatched),
  /workload.digest does not match the signed Pack release contract/,
);
assert.equal(rerankCalls.length, 1, 'identity mismatch must fail before program execution');

await assert.rejects(
  session.rerank({ ...request, documents: [] }),
  /documents must be a non-empty array/,
);
assert.equal(rerankCalls.length, 1, 'invalid workload input must fail before program execution');

await session.close();
await assert.rejects(session.rerank(request), /session is closed/);

console.log('✔ pack-rerank.test.js passed');
