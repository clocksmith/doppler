import assert from 'node:assert/strict';
import { createDopplerRuntime } from 'doppler-gpu';
import { createElectronRendererRuntime } from 'doppler-gpu/electron';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import {
  TEST_CAPSULE_AUTHORITY,
  TEST_CAPSULE_PUBLIC_KEY,
  createSignedCapsuleFixture,
} from '../helpers/capsule-v2-fixture.js';

const fixture = await createSignedCapsuleFixture({ operation: 'rerank' });
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
  executionGraphHash: fixture.capsule.program.executionGraphHash,
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
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  observer: { observe(event) { events.push(event); } },
  async programFactory() { return program; },
});

const session = await runtime.openCapsule(fixture.capsule);
const application = structuredClone(fixture.capsule.release.application);
const request = {
  application,
  query: evidence.query,
  documents: evidence.documents,
};
const receipt = await session.rerank(request);
assert.equal(receipt.schema, 'doppler.capsule-rerank-receipt/v1');
assert.equal(receipt.capsule.capsuleId, fixture.capsule.capsuleId);
assert.equal(receipt.capsule.semanticRoot, fixture.capsule.semanticRoot);
assert.equal(receipt.target.targetId, session.selectedTargetId);
assert.deepEqual(receipt.application, fixture.capsule.release.application);
assert.equal(receipt.evidence, evidence);
assert.match(receipt.receiptDigest, /^sha256:[0-9a-f]{64}$/);
assert.deepEqual(rerankCalls.map(({ options, ...input }) => input), [{
  query: evidence.query,
  documents: evidence.documents,
}]);
assert.equal(rerankCalls[0].options.signal.aborted, false);
assert.equal(events.at(-1).type, 'capsule-rerank-complete');
assert.equal(events.at(-1).receiptDigest, receipt.receiptDigest);

await assert.rejects(session.generate({}).next(), /not qualified.*generate/);
await assert.rejects(session.encodeSequence('MKT'), /not qualified.*encodeSequence/);
const generationOnly = await createSignedCapsuleFixture();
const unqualified = await runtime.openCapsule(generationOnly.capsule);
await assert.rejects(unqualified.rerank({ ...request, application: generationOnly.capsule.release.application }), /not qualified.*rerank/);
assert.equal(rerankCalls.length, 1, 'another operation must not borrow generation qualification');
await unqualified.close();

// Real public runtime + signed multi-plan Capsule + real Electron adapter;
// the program and device remain synthetic, not physical inference evidence.
const rerankAlternative = { ...fixture.targetPlan, targetId: 'rerank-alternative' };
const multiple = await createSignedCapsuleFixture({ targetPlans: [generationOnly.targetPlan, rerankAlternative] });
const selection = { acceptedTargetPlanDigests: [hashTargetPlan(rerankAlternative)], requiredOperations: ['rerank'] };
const approvedSession = await runtime.openCapsule(multiple.capsule, selection);
assert.equal(approvedSession.selectedTargetId, rerankAlternative.targetId);
const selectedDigest = approvedSession.selectedTargetPlanDigest;
await approvedSession.rerank(request);
await assert.rejects(approvedSession.generate({}).next(), /not qualified.*generate/);
assert.equal(approvedSession.selectedTargetPlanDigest, selectedDigest, 'execution never reselects another plan');
await approvedSession.close();

const renderer = createElectronRendererRuntime({
  releaseState: { async resolveCurrent() { return { ...multiple.capsule, path: 'multi-plan.json' }; } },
  async openCapsule(path, options) {
    assert.equal(path, 'multi-plan.json');
    return runtime.openCapsule(multiple.capsule, options);
  },
});
const electronReceipt = await renderer.rerank(request);
assert.equal(electronReceipt.target.targetId, rerankAlternative.targetId,
  'the reranking adapter requests rerank qualification before loading');
await assert.rejects(renderer.rerank(request, {
  acceptedTargetPlanDigests: [hashTargetPlan(generationOnly.targetPlan)],
}), /required operations/);
await assert.rejects(renderer.rerank(request, { requiredOperations: ['generate'] }), /required operations/);

const mutablePolicy = structuredClone(selection);
let factories = 0;
const snapshotRuntime = createDopplerRuntime({
  ...runtime.ports,
  artifactStore: {
    async readArtifact(artifact) {
      mutablePolicy.acceptedTargetPlanDigests.splice(0);
      mutablePolicy.requiredOperations.push('generate');
      return multiple.artifactStore.readArtifact(artifact);
    },
  },
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  async programFactory() { factories += 1; return program; },
});
const snapshotSession = await snapshotRuntime.openCapsule(multiple.capsule, mutablePolicy);
assert.equal(snapshotSession.selectedTargetId, rerankAlternative.targetId, 'opening snapshots authority before asynchronous artifact reads');
await snapshotSession.close();
await assert.rejects(snapshotRuntime.openCapsule(multiple.capsule, { acceptedTargetPlanDigests: [] }), /not accepted/);
await assert.rejects(snapshotRuntime.openCapsule(multiple.capsule, { requiredOperations: ['embed'] }), /required operations/);
assert.equal(factories, 1, 'no eligible plan must reject before program creation');

const mismatched = structuredClone(request);
mismatched.application.workload.digest = digest('f');
const callsBeforeInvalidInput = rerankCalls.length;
await assert.rejects(
  session.rerank(mismatched),
  /workload.digest does not match the signed Capsule release contract/,
);
assert.equal(rerankCalls.length, callsBeforeInvalidInput, 'identity mismatch must fail before program execution');

await assert.rejects(
  session.rerank({ ...request, documents: [] }),
  /documents must be a non-empty array/,
);
assert.equal(rerankCalls.length, callsBeforeInvalidInput, 'invalid workload input must fail before program execution');

await session.close();
await assert.rejects(session.rerank(request), /session is closed/);

console.log('✔ capsule-rerank.test.js passed');
