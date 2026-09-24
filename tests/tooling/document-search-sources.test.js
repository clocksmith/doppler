import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { checkDocumentSearchSources } from '../../tools/check-document-search-sources.js';
import { getCapsuleIdentity, buildCapsuleV2, signCapsuleV2 } from '../../src/capsule.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY, TEST_CAPSULE_PRIVATE_KEY } from '../helpers/capsule-v2-fixture.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-search-sources-'));
const originalFetch = globalThis.fetch;
try {
  const app = path.join(root, 'app');
  await fs.mkdir(app);
  const fixture = await createSignedCapsuleFixture();
  const candidate = structuredClone(fixture.capsule);
  candidate.artifacts.find(artifact => artifact.artifactId === 'weights').path = 'artifacts/model/shard_00000.bin';
  const capsule = await signCapsuleV2(buildCapsuleV2(candidate), { authority: TEST_CAPSULE_AUTHORITY,
    publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY, privateKeyJwk: TEST_CAPSULE_PRIVATE_KEY });
  const models = [], sources = { schema: 'doppler.document-search-sources/v1', artifacts: {} };
  for (const role of ['embedding', 'reranker']) {
    const modelRoot = path.join(app, 'capsules', role);
    await fs.mkdir(modelRoot, { recursive: true });
    await fs.writeFile(path.join(modelRoot, 'capsule-v3.json'), JSON.stringify(capsule));
    await fs.writeFile(path.join(modelRoot, 'MODEL_LICENSE.txt'), 'Synthetic test license');
    for (const artifact of capsule.artifacts) {
      const filename = path.join(modelRoot, artifact.path);
      await fs.mkdir(path.dirname(filename), { recursive: true });
      await fs.writeFile(filename, fixture.artifactBytes.get(artifact.artifactId));
      if (artifact.artifactId === 'weights') sources.artifacts['capsules/' + role + '/' + artifact.path] = {
        hash: artifact.hash, sizeBytes: artifact.sizeBytes, url: null,
      };
    }
    models.push({ role, capsuleUrl: './capsules/' + role + '/capsule-v3.json', identity: getCapsuleIdentity(capsule),
      options: { trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY } } });
  }
  await fs.writeFile(path.join(app, 'models.json'), JSON.stringify({ models }));
  const sourceFile = path.join(app, 'shard-sources.json');
  await fs.writeFile(sourceFile, JSON.stringify(sources));
  const config = { applicationDir: app, localRoot: app, receiptPath: path.join(root, 'local.json'), stagingDir: path.join(root, 'upload') };
  const local = await checkDocumentSearchSources(config);
  assert.equal(local.passed, true);
  assert.equal(local.publicAcquisitionComplete, false);
  assert.equal(local.missingSources.length, 2);
  assert.deepEqual(JSON.parse(await fs.readFile(sourceFile)), sources);
  const weight = capsule.artifacts.find(artifact => artifact.artifactId === 'weights');
  assert.deepEqual(await fs.readFile(path.join(config.stagingDir, 'document-search/artifacts/sha256', weight.hash.slice(7) + '.bin')),
    Buffer.from(fixture.artifactBytes.get('weights')));
  const corrupt = structuredClone(sources);
  Object.values(corrupt.artifacts)[0].sizeBytes++;
  await fs.writeFile(sourceFile, JSON.stringify(corrupt));
  await assert.rejects(checkDocumentSearchSources(config), /size declaration/);
  await fs.writeFile(sourceFile, JSON.stringify(sources));
  const remoteConfig = { applicationDir: app, revision: 'a'.repeat(40), receiptPath: path.join(root, 'remote.json') };
  globalThis.fetch = async () => new Response(new Uint8Array([0, 0, 0, 0]));
  await assert.rejects(checkDocumentSearchSources(remoteConfig), /downloaded hash/);
  assert.deepEqual(JSON.parse(await fs.readFile(sourceFile)), sources, 'Failed download must not publish URLs');
  globalThis.fetch = async url => {
    assert.ok(url.includes('/resolve/' + remoteConfig.revision + '/document-search/artifacts/sha256/'));
    return new Response(fixture.artifactBytes.get('weights'));
  };
  assert.equal((await checkDocumentSearchSources(remoteConfig)).publicAcquisitionComplete, true);
  assert.ok(Object.values(JSON.parse(await fs.readFile(sourceFile)).artifacts).every(artifact => artifact.url));
} finally {
  globalThis.fetch = originalFetch;
  await fs.rm(root, { recursive: true, force: true });
}
console.log('document-search-sources: synthetic signature, exact-byte, corruption, staging and immutable URL checks passed');
