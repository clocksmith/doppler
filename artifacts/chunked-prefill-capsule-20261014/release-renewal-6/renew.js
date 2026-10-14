// Renew the test release of the chunked-prefill Capsule (semantic root 171132d4..., candidate
// 2df37d88) before sequence 5 expires. Appends sequence 6 to the authority's existing history
// (renewal 5's output): same Capsule envelope, release contract and migration source as
// sequence 5, chained to its
// digest, with a new expiry from the release policy. Nothing is rebuilt or re-signed except the
// new event. The private key is read from its file and never printed.
// Usage: node renew.js /var/tmp/qwen35-capsule-chunk16-20261014-renewal-6
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const output = process.argv[2];
assert(output === '/var/tmp/qwen35-capsule-chunk16-20261014-renewal-6', 'Usage: node renew.js /var/tmp/qwen35-capsule-chunk16-20261014-renewal-6');
const capsuleDir = '/var/tmp/qwen35-capsule-chunk16-20261014-b';
const runtime = '/var/tmp/reploid-completion-candidate-20261013/vendor/doppler/0.6.28';
const keys = '/var/tmp/doppler-0.6.28-source-20261012/capsule-input';
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const load = name => import(pathToFileURL(`${runtime}/src/config/${name}`).href);
const { signCapsuleReleaseEvent } = await load('capsule-release-events.js');
const { getCapsuleIdentity, verifyCapsule } = await load('capsule.js');

const capsule = await read(`${capsuleDir}/capsule-v3.json`);
const policy = await read(`${capsuleDir}/release-policy.json`);
const history = await read('/var/tmp/qwen35-capsule-chunk16-20261014-renewal-5/release-events.json');
assert.equal(history.length, 5);
const prior = history[4];
assert.equal(prior.sequence, 5);
assert.equal(prior.digest, 'sha256:70f24c979f6b08c25827de341f5b08ab441b1c30fdca6bcec019a596c3ed71d2');
const identity = getCapsuleIdentity(capsule);
assert.deepEqual(prior.capsule, { schema: identity.schema, semanticRoot: identity.semanticRoot, envelopeDigest: identity.envelopeDigest });
const signer = { authority: policy.authority, privateKeyJwk: await read(`${keys}/signing-private.jwk.json`),
  publicKeyJwk: await read(`${keys}/signing-public.jwk.json`) };
assert.equal(prior.signature.authority, signer.authority);
const trustedSigners = await read(`${capsuleDir}/trusted-signers.json`);
assert.deepEqual(trustedSigners, { [signer.authority]: signer.publicKeyJwk });

const now = new Date().toISOString();
assert(Date.parse(now) < Date.parse(prior.expiresAtUtc), 'Sequence 5 already expired; renew through a new decision');
const expires = new Date(Date.parse(now) + policy.offlineExpirySeconds * 1000).toISOString();
const event = await signCapsuleReleaseEvent({ capsule: prior.capsule, sequence: 6, previousEventDigest: prior.digest,
  issuedAtUtc: now, expiresAtUtc: expires, action: 'eligible', release: prior.release, migratedFrom: prior.migratedFrom,
  nextSigner: null }, signer);
const renewed = [...history, event];

// Hosts holding checkpoint 5 (final-9), 4 (final-6 to final-8), or none must all admit the
// renewed history.
const open = checkpoint => verifyCapsule(capsule, { trustedSigners, releaseTrustedSigners: trustedSigners, releaseEvents: renewed,
  releasePolicy: { now, minimumSequence: 6, checkpoint },
  artifactStore: { readArtifact: artifact => fs.readFile(`${capsuleDir}/${artifact.path}`) } });
const checkpoints = [{ sequence: 5, digest: prior.digest }, { sequence: 4, digest: history[3].digest }, { sequence: 0, digest: null }];
const results = [];
for (const checkpoint of checkpoints) {
  const verification = await open(checkpoint);
  assert.deepEqual(verification.lifecycle.checkpoint, { sequence: 6, digest: event.digest });
  results.push({ from: checkpoint, to: verification.lifecycle.checkpoint, artifacts: verification.artifactReceipts.length });
}

await fs.mkdir(output);
const write = (name, value) => fs.writeFile(`${output}/${name}`, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
await write('release-events.json', renewed);
await write('renewal-verification.json', { schema: 'doppler.capsule-release-renewal/v1', capsule: identity,
  runtimeArchive: policy.runtimeArchiveSha256, authority: signer.authority,
  scope: policy.scope, previous: { sequence: 5, digest: prior.digest, expiresAtUtc: prior.expiresAtUtc },
  renewed: { sequence: 6, digest: event.digest, issuedAtUtc: now, expiresAtUtc: expires },
  verifiedFromCheckpoints: results, verifiedAt: now, publicPromotion: false });
console.log(JSON.stringify({ sequence: 6, digest: event.digest, expiresAtUtc: expires, verified: results.length }));
