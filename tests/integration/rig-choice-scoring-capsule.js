// Existing Rig, with explicitly local test-release metadata and fresh test keys.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { generateKeyPairSync, createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { rigModelCapsule } from '../../src/tooling/model-capsule-rig.js';

const [modelArg, nodePath, browserPath, outputArg] = process.argv.slice(2);
assert(outputArg, 'Expected model, Node qualification, browser qualification and new Capsule output directory');
const modelDir = resolve(modelArg), output = resolve(outputArg);
const node = JSON.parse(await readFile(nodePath, 'utf8')), browser = JSON.parse(await readFile(browserPath, 'utf8'));
assert(node.passed && browser.passed);
assert.equal(node.referenceDigest, browser.referenceDigest);
const hash = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
assert.equal(hash(await readFile(modelDir + '/manifest.json')), node.model.manifestHash);
const keys = generateKeyPairSync('ed25519');
await mkdir(output); // Never replace a previous signed artifact set.
await writeFile(output + '/signing-private.json', JSON.stringify(keys.privateKey.export({ format: 'jwk' }), null, 2) + '\n', { mode: 0o600 });
await writeFile(output + '/signing-public.json', JSON.stringify(keys.publicKey.export({ format: 'jwk' }), null, 2) + '\n');
const note = 'Local integration test; upstream source licensing and public release not qualified.';
const release = {
  schema: 'doppler.capsule-release/v1',
  source: { repository: 'local-verified-rdrr:' + node.model.modelId, revision: node.model.manifestHash,
    revisionDigest: node.model.manifestHash, provenanceDigest: node.model.manifestHash,
    license: { spdxId: 'LicenseRef-TestMetadata', name: note, sourceUrl: 'https://huggingface.co/Qwen/Qwen3-Reranker-0.6B', textDigest: hash(note) } },
  application: { applicationId: 'choice-acceptance', applicationRevision: 'test-fixture',
    applicationRevisionDigest: hash(await readFile(new URL(import.meta.url))),
    workload: { id: 'relevance-choices', digest: node.reference.contractHash },
    oracle: { id: 'independent-transformers-cpu', digest: node.referenceDigest } },
  exclusions: { rejectionTypes: ['acceptance-failed', 'application-gate-failed', 'artifact-invalid', 'evidence-expired', 'migration-required', 'revoked', 'unsupported-device'],
    known: [{ code: 'acceptance-failed', scope: 'public-release', reason: note, evidenceDigest: node.referenceDigest }] },
  lifecycle: { releaseVersion: '0.0.1', supersedes: null, migration: null,
    failedUpgrade: { preservePrevious: true, previousCapsuleId: null, previousSemanticRoot: null } },
  revocation: { authorityId: 'local-choice-acceptance', policyDigest: computeCanonicalSha256({ scope: 'local-test-only' }),
    offlineExpirySeconds: 86400, failClosedAfterExpiry: true },
  stateSnapshot: { schema: 'doppler.capsule-state-snapshot/v1', format: 'canonical-json',
    identityDigest: computeCanonicalSha256({ persistentModelState: false }), portableAcrossTargetIds: ['webgpu-f32-f16-subgroups'] },
};
await writeFile(output + '/release.json', JSON.stringify(release, null, 2) + '\n');
const options = { repoRoot: resolve(import.meta.dirname, '../..'), modelDir, manifestPath: modelDir + '/manifest.json',
  referenceReportPath: resolve(nodePath), qualificationReportPaths: [resolve(browserPath)],
  releaseManifestPath: output + '/release.json', outputPath: output + '/capsule.json',
  signingPrivateKeyPath: output + '/signing-private.json', signingPublicKeyPath: output + '/signing-public.json',
  signingAuthority: 'local-choice-acceptance', createdAtUtc: new Date().toISOString() };
await writeFile(output + '/build-config.json', JSON.stringify(options, null, 2) + '\n');
const result = await rigModelCapsule(options);
await writeFile(output + '/build.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ output, capsuleId: result.capsuleId }));
