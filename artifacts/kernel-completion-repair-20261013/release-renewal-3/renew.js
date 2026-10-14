// Renew the test release of Capsule 67f240af (candidate 2df37d88) before sequence 2 expires.
// Appends sequence 3 to the authority's existing history: same Capsule envelope, release contract
// and migration source as sequence 2, chained to its digest, with a new 24-hour expiry. Nothing is
// rebuilt or re-signed except the new event. The private key is read from its file and never printed.
// Usage: node renew.js /var/tmp/qwen35-capsule-completion-20261013-renewal-3
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const output = process.argv[2];
assert(output === '/var/tmp/qwen35-capsule-completion-20261013-renewal-3', 'Usage: node renew.js /var/tmp/qwen35-capsule-completion-20261013-renewal-3');
const capsuleDir = '/var/tmp/qwen35-capsule-completion-20261013-c';
const runtime = '/var/tmp/reploid-completion-candidate-20261013/vendor/doppler/0.6.28';
const keys = '/var/tmp/doppler-0.6.28-source-20261012/capsule-input';
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const load = name => import(pathToFileURL(`${runtime}/src/config/${name}`).href);
const { signCapsuleReleaseEvent } = await load('capsule-release-events.js');
const { getCapsuleIdentity, verifyCapsule } = await load('capsule.js');

const capsule = await read(`${capsuleDir}/capsule-v3.json`);
const policy = await read(`${capsuleDir}/release-policy.json`);
const history = await read(`${capsuleDir}/release-events.json`);
assert.equal(history.length, 2);
const prior = history[1];
assert.equal(prior.sequence, 2);
assert.equal(prior.digest, 'sha256:923ac7e602b3075a17f9e540eb0abfd436eaf1de41bebf8b6f15bdaee604b45f');
const identity = getCapsuleIdentity(capsule);
assert.deepEqual(prior.capsule, { schema: identity.schema, semanticRoot: identity.semanticRoot, envelopeDigest: identity.envelopeDigest });
const signer = { authority: policy.authority, privateKeyJwk: await read(`${keys}/signing-private.jwk.json`),
  publicKeyJwk: await read(`${keys}/signing-public.jwk.json`) };
assert.equal(prior.signature.authority, signer.authority);
const trustedSigners = await read(`${capsuleDir}/trusted-signers.json`);
assert.deepEqual(trustedSigners, { [signer.authority]: signer.publicKeyJwk });

const now = new Date().toISOString();
assert(Date.parse(now) < Date.parse(prior.expiresAtUtc), 'Sequence 2 already expired; renew through a new decision');
const expires = new Date(Date.parse(now) + policy.offlineExpirySeconds * 1000).toISOString();
const event = await signCapsuleReleaseEvent({ capsule: prior.capsule, sequence: 3, previousEventDigest: prior.digest,
  issuedAtUtc: now, expiresAtUtc: expires, action: 'eligible', release: prior.release, migratedFrom: prior.migratedFrom,
  nextSigner: null }, signer);
const renewed = [...history, event];

// Hosts holding checkpoint 2 (final-5 installs), 1, or none must all admit the renewed history.
const open = checkpoint => verifyCapsule(capsule, { trustedSigners, releaseTrustedSigners: trustedSigners, releaseEvents: renewed,
  releasePolicy: { now, minimumSequence: 3, checkpoint },
  artifactStore: { readArtifact: artifact => fs.readFile(`${capsuleDir}/${artifact.path}`) } });
const checkpoints = [{ sequence: 2, digest: prior.digest }, { sequence: 1, digest: history[0].digest }, { sequence: 0, digest: null }];
const results = [];
for (const checkpoint of checkpoints) {
  const verification = await open(checkpoint);
  assert.deepEqual(verification.lifecycle.checkpoint, { sequence: 3, digest: event.digest });
  results.push({ from: checkpoint, to: verification.lifecycle.checkpoint, artifacts: verification.artifactReceipts.length });
}

await fs.mkdir(output);
const write = (name, value) => fs.writeFile(`${output}/${name}`, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
await write('release-events.json', renewed);
await write('renewal-verification.json', { schema: 'doppler.capsule-release-renewal/v1', capsule: identity,
  runtimeArchive: '2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8', authority: signer.authority,
  scope: policy.scope, previous: { sequence: 2, digest: prior.digest, expiresAtUtc: prior.expiresAtUtc },
  renewed: { sequence: 3, digest: event.digest, issuedAtUtc: now, expiresAtUtc: expires },
  verifiedFromCheckpoints: results, verifiedAt: now, publicPromotion: false });
console.log(JSON.stringify({ sequence: 3, digest: event.digest, expiresAtUtc: expires, verified: results.length }));
