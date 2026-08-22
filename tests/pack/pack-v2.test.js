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
