import assert from 'node:assert/strict';
import { createPackNodeQualificationReceipt, isSoftwareAdapter } from '../../tools/qualify-pack-v0-node.js';

const digest = (character) => `sha256:${character.repeat(64)}`;
const identity = {
  digest: digest('a'),
  executionGraphHash: digest('b'),
  kernelClosureHash: digest('c'),
  runtimeEngineDigest: digest('d'),
};
const pack = {
  signature: {
    authority: 'test-authority',
    algorithm: 'Ed25519',
    publicKeyDigest: digest('e'),
  },
  artifacts: [{ artifactId: 'one' }, { artifactId: 'two' }],
  wgslModules: [{ id: 'kernel' }],
  modelIR: { supportScope: { qualifiedEntryPoints: ['text.generate'] } },
};
const session = {
  packId: 'pack.test',
  semanticRoot: digest('f'),
  selectedTargetId: 'node-webgpu',
  selectedTargetPlanDigest: digest('1'),
  selectedPlan: { initialExecutionIdentity: identity },
  observedInitialExecutionIdentity: structuredClone(identity),
  verification: { artifactReceipts: [{}, {}] },
  deviceProfile: {
    surface: 'node-webgpu',
    adapter: { vendor: 'test', device: 'hardware' },
  },
};

const receipt = createPackNodeQualificationReceipt({
  pack,
  session,
  generatedTokenIds: [1, 2, 3],
  expectedTokenIds: [1, 2, 3],
  targetPlanDigestBefore: digest('1'),
  capturedAtUtc: '2026-08-23T00:00:00.000Z',
});
assert.equal(receipt.schema, 'doppler.pack-node-qualification/v2');
assert.equal(receipt.passed, true);
assert.equal(receipt.targetPlanImmutable, true);
assert.equal(receipt.initialExecutionIdentity.boundBeforePrefill, true);
assert.equal(receipt.initialExecutionIdentity.declaredDigest, identity.digest);
assert.equal(receipt.initialExecutionIdentity.observedDigest, identity.digest);
assert.equal(receipt.closure.artifacts, 2);
assert.equal(receipt.closure.verifiedArtifacts, 2);
assert.deepEqual(receipt.closure.qualifiedEntryPoints, ['text.generate']);

await assert.rejects(async () => createPackNodeQualificationReceipt({
  pack,
  session: {
    ...session,
    observedInitialExecutionIdentity: { ...identity, digest: digest('9') },
  },
  generatedTokenIds: [1],
  expectedTokenIds: [1],
  targetPlanDigestBefore: digest('1'),
  capturedAtUtc: '2026-08-23T00:00:00.000Z',
}), /initial execution identity mismatch/);

assert.equal(isSoftwareAdapter({ adapter: { description: 'SwiftShader Device' } }), true);
assert.equal(isSoftwareAdapter(session.deviceProfile), false);
console.log('✔ qualify-pack-v0-node.test.js passed');
