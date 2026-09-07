import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { generateKeyPairSync } from 'node:crypto';
import { buildDocumentSearchApplication } from '../../tools/build-document-search-app.js';
import { signCapsuleV2, signCapsuleReleaseEvent } from '../../src/capsule.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';
import { createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';

// Synthetic files test signing continuity, not installed runtime execution.
const root = await fs.mkdtemp(path.join(tmpdir(), 'doppler-search-build-'));
const write = (filename, value) => fs.writeFile(filename, JSON.stringify(value));
const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
try {
  const keys = generateKeyPairSync('ed25519');
  const signer = { authority: 'search-builder-test', publicKeyJwk: keys.publicKey.export({ format: 'jwk' }),
    privateKeyJwk: keys.privateKey.export({ format: 'jwk' }) };
  const capsuleRoot = path.join(root, 'capsule');
  await fs.mkdir(path.join(capsuleRoot, 'distribution'), { recursive: true });
  await fs.mkdir(path.join(capsuleRoot, 'custody'));
  const fixture = await createSignedCapsuleFixture();
  const capsule = await signCapsuleV2(fixture.capsule, signer);
  await write(path.join(capsuleRoot, 'distribution/capsule.json'), capsule);
  await write(path.join(capsuleRoot, 'custody/public-key.json'), signer.publicKeyJwk);
  await write(path.join(capsuleRoot, 'custody/private-key.json'), signer.privateKeyJwk);
  await write(path.join(capsuleRoot, 'open-options.json'), { trustedSigners: { [signer.authority]: signer.publicKeyJwk } });
  const packageBundlePath = path.join(root, 'package');
  await fs.mkdir(path.join(packageBundlePath, 'consumer/node_modules/doppler-gpu/src'), { recursive: true });
  const archive = Buffer.from('synthetic archive');
  await fs.writeFile(path.join(packageBundlePath, 'fixture.tgz'), archive);
  await write(path.join(packageBundlePath, 'receipt.json'), { passed: true,
    package: { filename: 'fixture.tgz', sha256: hashBytesSha256(archive).slice(7) } });
  const config = { packageBundlePath, outputDir: path.join(root, 'first'), previousApplicationDir: null,
    models: ['embedding', 'reranker'].map(role => ({ role, capsuleRoot })), search: {}, storage: {} };
  await buildDocumentSearchApplication(config);
  const first = await read(path.join(config.outputDir, 'models.json'));
  const secondDir = path.join(root, 'second');
  await buildDocumentSearchApplication({ ...config, previousApplicationDir: config.outputDir, outputDir: secondDir });
  const second = await read(path.join(secondDir, 'models.json'));
  for (let index = 0; index < 2; index++) {
    assert.deepEqual(second.models[index].options.releaseEvents, first.models[index].options.releaseEvents);
    assert.deepEqual(second.models[index].options.releasePolicy.checkpoint, first.models[index].options.releasePolicy.checkpoint);
  }
  const prior = structuredClone(first);
  const model = prior.models[0];
  const eligible = model.options.releaseEvents[0];
  const changed = await signCapsuleReleaseEvent({ ...eligible, sequence: 2, previousEventDigest: eligible.digest,
    release: { ...eligible.release, application: { ...eligible.release.application, applicationRevision: 'other-application-revision' } } }, signer);
  model.options.releaseEvents.push(changed);
  model.options.releasePolicy.checkpoint = { sequence: 2, digest: changed.digest };
  model.options.releasePolicy.minimumSequence = 2;
  const previousDir = path.join(root, 'previous'); await fs.mkdir(previousDir);
  await write(path.join(previousDir, 'models.json'), prior);
  const updatedDir = path.join(root, 'updated');
  await buildDocumentSearchApplication({ ...config, previousApplicationDir: previousDir, outputDir: updatedDir });
  const updated = await read(path.join(updatedDir, 'models.json'));
  assert.deepEqual(updated.models[0].options.releaseEvents.slice(0, 2), model.options.releaseEvents);
  assert.equal(updated.models[0].options.releasePolicy.checkpoint.sequence, 3);
  const revoked = await signCapsuleReleaseEvent({ ...eligible, sequence: 2, previousEventDigest: eligible.digest, action: 'revoked' }, signer);
  model.options.releaseEvents = [eligible, revoked];
  model.options.releasePolicy.checkpoint = { sequence: 2, digest: revoked.digest };
  await write(path.join(previousDir, 'models.json'), prior);
  await assert.rejects(buildDocumentSearchApplication({ ...config, previousApplicationDir: previousDir,
    outputDir: path.join(root, 'denied') }), /blocked/);
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('document-search-build.test: passed (synthetic release continuity and denial)');
