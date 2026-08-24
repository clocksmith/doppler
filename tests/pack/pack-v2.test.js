import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  hashPackV2,
  loadPackV2,
  validatePackV2,
  verifyPackV2,
  writePackV2,
} from '../../src/tooling/pack-v2.js';
import {
  TEST_PACK_AUTHORITY,
  TEST_PACK_PUBLIC_KEY,
  createSignedPackFixture,
} from '../helpers/pack-v2-fixture.js';

const fixture = await createSignedPackFixture();
assert.deepEqual(validatePackV2(fixture.pack), { ok: true, errors: [] });
assert.equal(hashPackV2(fixture.pack), fixture.pack.semanticRoot);
await verifyPackV2(fixture.pack, {
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  artifactStore: fixture.artifactStore,
});

const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-pack-v2-test-'));
const outPath = path.join(tmpRoot, 'test.pack.json');
await writePackV2(outPath, fixture.pack);
const loaded = await loadPackV2(outPath);
assert.equal(loaded.packId, fixture.pack.packId);
assert.equal(Object.isFrozen(loaded), true);

const mismatchedModel = structuredClone(fixture.pack);
mismatchedModel.modelId = 'wrong-model';
assert.equal(validatePackV2(mismatchedModel).ok, false);

const missingKernel = structuredClone(fixture.pack);
missingKernel.targetPlans[0].kernelClosure[0].moduleId = 'missing';
assert.equal(validatePackV2(missingKernel).ok, false);

const changedWgslDescriptor = structuredClone(fixture.pack);
changedWgslDescriptor.wgslModules[0].entry = 'changed';
assert.equal(validatePackV2(changedWgslDescriptor).ok, false, 'semantic root must reject WGSL metadata mutation');

const changedReleaseSource = structuredClone(fixture.pack);
changedReleaseSource.release.source.revision = 'changed-revision';
assert.equal(validatePackV2(changedReleaseSource).ok, false, 'semantic root must bind source revision');

const unsignedTopLevelExtension = structuredClone(fixture.pack);
unsignedTopLevelExtension.runtimeFallback = 'invented';
assert.ok(validatePackV2(unsignedTopLevelExtension).errors.includes(
  'pack.runtimeFallback is not allowed.'
));

const unsignedArtifactExtension = structuredClone(fixture.pack);
unsignedArtifactExtension.artifacts[0].mutableSource = true;
assert.ok(validatePackV2(unsignedArtifactExtension).errors.includes(
  'artifacts[0].mutableSource is not allowed.'
));

const unsignedSignatureExtension = structuredClone(fixture.pack);
unsignedSignatureExtension.signature.mutableSignerState = true;
assert.ok(validatePackV2(unsignedSignatureExtension).errors.includes(
  'signature.mutableSignerState is not allowed.'
));

const malformedSignature = structuredClone(fixture.pack);
malformedSignature.signature.signatureHex = '00';
assert.ok(validatePackV2(malformedSignature).errors.includes(
  'signature.signatureHex must be a 64-byte hexadecimal Ed25519 signature.'
));

const missingLicenseDigest = structuredClone(fixture.pack);
missingLicenseDigest.release.source.license.textDigest = null;
assert.ok(validatePackV2(missingLicenseDigest).errors.includes(
  'release.source.license.textDigest must be a SHA-256 digest.'
));

const changedWorkloadIdentity = structuredClone(fixture.pack);
changedWorkloadIdentity.release.application.workload.digest = `sha256:${'0'.repeat(64)}`;
assert.equal(validatePackV2(changedWorkloadIdentity).ok, false, 'semantic root must bind workload identity');

const untypedExclusion = structuredClone(fixture.pack);
untypedExclusion.release.exclusions.known[0].code = 'unknown-rejection';
assert.ok(validatePackV2(untypedExclusion).errors.includes(
  'release.exclusions.known[0].code is unsupported.'
));

const discardedPreviousPack = structuredClone(fixture.pack);
discardedPreviousPack.release.lifecycle.failedUpgrade.preservePrevious = false;
assert.ok(validatePackV2(discardedPreviousPack).errors.includes(
  'release.lifecycle.failedUpgrade.preservePrevious must be true.'
));

const changedRevocationPolicy = structuredClone(fixture.pack);
changedRevocationPolicy.release.revocation.failClosedAfterExpiry = false;
assert.ok(validatePackV2(changedRevocationPolicy).errors.includes(
  'release.revocation.failClosedAfterExpiry must be true.'
));

const unboundSnapshotTarget = structuredClone(fixture.pack);
unboundSnapshotTarget.release.stateSnapshot.portableAcrossTargetIds = ['missing-target'];
assert.ok(validatePackV2(unboundSnapshotTarget).errors.includes(
  'release.stateSnapshot target "missing-target" is not carried by the Pack.'
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
  verifyPackV2(fixture.pack, {
    trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
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
  verifyPackV2(fixture.pack, {
    trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
    artifactStore: changedWgslBytesStore,
  }),
  /artifact hash mismatch/
);

await assert.rejects(
  verifyPackV2(fixture.pack, { trustedSigners: {}, artifactStore: fixture.artifactStore }),
  /Untrusted Doppler Pack signing authority/
);

console.log('✔ pack-v2.test.js passed');
