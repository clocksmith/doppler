import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  hashCapsuleV2,
  loadCapsuleV2,
  validateCapsuleV2,
  verifyCapsuleV2,
  writeCapsuleV2,
} from '../../src/tooling/capsule-v2.js';
import {
  TEST_CAPSULE_AUTHORITY,
  TEST_CAPSULE_PUBLIC_KEY,
  createSignedCapsuleFixture,
} from '../helpers/capsule-v2-fixture.js';

const fixture = await createSignedCapsuleFixture();
assert.deepEqual(validateCapsuleV2(fixture.capsule), { ok: true, errors: [] });
assert.equal(hashCapsuleV2(fixture.capsule), fixture.capsule.semanticRoot);
await verifyCapsuleV2(fixture.capsule, {
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  artifactStore: fixture.artifactStore,
});

const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-capsule-v2-test-'));
const outPath = path.join(tmpRoot, 'test.capsule.json');
await writeCapsuleV2(outPath, fixture.capsule);
const loaded = await loadCapsuleV2(outPath);
assert.equal(loaded.capsuleId, fixture.capsule.capsuleId);
assert.equal(Object.isFrozen(loaded), true);

const mismatchedModel = structuredClone(fixture.capsule);
mismatchedModel.modelId = 'wrong-model';
assert.equal(validateCapsuleV2(mismatchedModel).ok, false);

const missingKernel = structuredClone(fixture.capsule);
missingKernel.targetPlans[0].kernelClosure[0].moduleId = 'missing';
assert.equal(validateCapsuleV2(missingKernel).ok, false);

const changedWgslDescriptor = structuredClone(fixture.capsule);
changedWgslDescriptor.wgslModules[0].entry = 'changed';
assert.equal(validateCapsuleV2(changedWgslDescriptor).ok, false, 'semantic root must reject WGSL metadata mutation');

const changedReleaseSource = structuredClone(fixture.capsule);
changedReleaseSource.release.source.revision = 'changed-revision';
assert.equal(validateCapsuleV2(changedReleaseSource).ok, false, 'semantic root must bind source revision');

const unsignedTopLevelExtension = structuredClone(fixture.capsule);
unsignedTopLevelExtension.runtimeFallback = 'invented';
assert.ok(validateCapsuleV2(unsignedTopLevelExtension).errors.includes(
  'capsule.runtimeFallback is not allowed.'
));

const unsignedArtifactExtension = structuredClone(fixture.capsule);
unsignedArtifactExtension.artifacts[0].mutableSource = true;
assert.ok(validateCapsuleV2(unsignedArtifactExtension).errors.includes(
  'artifacts[0].mutableSource is not allowed.'
));

const unsignedSignatureExtension = structuredClone(fixture.capsule);
unsignedSignatureExtension.signature.mutableSignerState = true;
assert.ok(validateCapsuleV2(unsignedSignatureExtension).errors.includes(
  'signature.mutableSignerState is not allowed.'
));

const malformedSignature = structuredClone(fixture.capsule);
malformedSignature.signature.signatureHex = '00';
assert.ok(validateCapsuleV2(malformedSignature).errors.includes(
  'signature.signatureHex must be a 64-byte hexadecimal Ed25519 signature.'
));

const missingLicenseDigest = structuredClone(fixture.capsule);
missingLicenseDigest.release.source.license.textDigest = null;
assert.ok(validateCapsuleV2(missingLicenseDigest).errors.includes(
  'release.source.license.textDigest must be a SHA-256 digest.'
));

const changedWorkloadIdentity = structuredClone(fixture.capsule);
changedWorkloadIdentity.release.application.workload.digest = `sha256:${'0'.repeat(64)}`;
assert.equal(validateCapsuleV2(changedWorkloadIdentity).ok, false, 'semantic root must bind workload identity');

const untypedExclusion = structuredClone(fixture.capsule);
untypedExclusion.release.exclusions.known[0].code = 'unknown-rejection';
assert.ok(validateCapsuleV2(untypedExclusion).errors.includes(
  'release.exclusions.known[0].code is unsupported.'
));

const discardedPreviousCapsule = structuredClone(fixture.capsule);
discardedPreviousCapsule.release.lifecycle.failedUpgrade.preservePrevious = false;
assert.ok(validateCapsuleV2(discardedPreviousCapsule).errors.includes(
  'release.lifecycle.failedUpgrade.preservePrevious must be true.'
));

const changedRevocationPolicy = structuredClone(fixture.capsule);
changedRevocationPolicy.release.revocation.failClosedAfterExpiry = false;
assert.ok(validateCapsuleV2(changedRevocationPolicy).errors.includes(
  'release.revocation.failClosedAfterExpiry must be true.'
));

const unboundSnapshotTarget = structuredClone(fixture.capsule);
unboundSnapshotTarget.release.stateSnapshot.portableAcrossTargetIds = ['missing-target'];
assert.ok(validateCapsuleV2(unboundSnapshotTarget).errors.includes(
  'release.stateSnapshot target "missing-target" is not carried by the Capsule.'
));

const changedBytesStore = {
  ...fixture.artifactStore,
  async hashArtifact(artifact) {
    const receipt = await fixture.artifactStore.hashArtifact(artifact);
    return artifact.artifactId === 'weights'
      ? { ...receipt, hash: `sha256:${'f'.repeat(64)}` }
      : receipt;
  },
};
await assert.rejects(
  verifyCapsuleV2(fixture.capsule, {
    trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    artifactStore: changedBytesStore,
  }),
  /artifact hash mismatch/
);

const changedWgslBytesStore = {
  ...fixture.artifactStore,
  async hashArtifact(artifact) {
    const receipt = await fixture.artifactStore.hashArtifact(artifact);
    return artifact.role === 'wgsl-source'
      ? { ...receipt, hash: `sha256:${'e'.repeat(64)}` }
      : receipt;
  },
};
await assert.rejects(
  verifyCapsuleV2(fixture.capsule, {
    trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    artifactStore: changedWgslBytesStore,
  }),
  /artifact hash mismatch/
);

await assert.rejects(
  verifyCapsuleV2(fixture.capsule, { trustedSigners: {}, artifactStore: fixture.artifactStore }),
  /Untrusted Doppler Capsule signing authority/
);

console.log('✔ capsule-v2.test.js passed');
