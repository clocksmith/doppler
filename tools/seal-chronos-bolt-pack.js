// Internal Forge release tool. Qualification does not imply forecast skill or
// external adoption. The application explicitly pins the emitted trust profile.
import fs from 'node:fs/promises';
import path from 'node:path';
import { generateKeyPairSync } from 'node:crypto';
import { buildPackV3, signPackV3, getPackIdentity, signPackReleaseEvent } from '../src/pack.js';
import { hashTargetPlan } from '../src/config/target-plan.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, i, values) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), values[i + 1]]);
  return pairs;
}, []));
if (args['init-signer']) {
  const keys = generateKeyPairSync('ed25519');
  const signer = { authority: 'd4da-local-qualification',
    publicKeyJwk: keys.publicKey.export({ format: 'jwk' }), privateKeyJwk: keys.privateKey.export({ format: 'jwk' }) };
  await fs.writeFile(path.resolve(args['init-signer']), JSON.stringify(signer), { flag: 'wx', mode: 0o600 });
  console.log('Created local qualification signer; keep its private file outside public artifacts.');
} else {
  for (const name of ['model', 'node-report', 'browser-report', 'application', 'signer', 'expires']) {
    if (!args[name]) throw new Error('Required --' + name);
  }
  const root = path.resolve(args.model);
  const signerPath = path.resolve(args.signer);
  if (signerPath.startsWith(root + path.sep)) throw new Error('Private signer cannot be inside the distributable Pack directory.');
  const json = async file => JSON.parse(await fs.readFile(file, 'utf8'));
  const candidate = await json(path.join(root, 'candidate.json'));
  const reference = await json(path.join(root, 'source/reference.json'));
  const applicationContract = await json(path.resolve(args.application));
  const signer = await json(signerPath);
  const artifacts = structuredClone(candidate.artifacts);
  // Every byte in the observed candidate remains under custody at sealing.
  for (const entry of artifacts) {
    const file = path.resolve(root, entry.path);
    if (!file.startsWith(root + path.sep)) throw new Error('Artifact path escapes Pack');
    const bytes = await fs.readFile(file);
    if (bytes.byteLength !== entry.sizeBytes || hashBytesSha256(bytes) !== entry.hash) throw new Error('Artifact changed after qualification: ' + entry.path);
  }
  async function retain(id, role, file, bytes) {
    await fs.mkdir(path.dirname(path.join(root, file)), { recursive: true });
    await fs.writeFile(path.join(root, file), bytes);
    const entry = { artifactId: id, role, path: file, hash: hashBytesSha256(bytes), sizeBytes: bytes.byteLength };
    artifacts.push(entry);
    return entry;
  }
  const qualification = [];
  for (const [name, surface] of [['node-report', 'node-webgpu'], ['browser-report', 'browser-webgpu']]) {
    const bytes = await fs.readFile(path.resolve(args[name]));
    const report = JSON.parse(bytes);
    if (report.schema !== 'doppler.forecast-qualification/v1' || report.status !== 'passed' || report.failure !== null
      || report.surface !== surface || report.candidateHash !== computeCanonicalSha256(candidate)
      || report.artifactClosureHash !== computeCanonicalSha256(candidate.artifacts)
      || report.executionGraphHash !== candidate.program.executionGraphHash
      || report.referenceHash !== computeCanonicalSha256(reference)
      || computeCanonicalSha256(report.tolerance) !== computeCanonicalSha256(reference.tolerance)
      || report.rows.length !== reference.cases.length || !report.adapter
      || /swiftshader|llvmpipe|software/i.test(Object.values(report.adapter).join(' '))) {
      throw new Error('Qualification is missing, rejected, or belongs to another candidate: ' + surface);
    }
    for (const [index, expected] of reference.cases.entries()) {
      const observed = report.rows[index];
      if (observed.id !== expected.id || observed.output.horizon !== expected.horizon
        || observed.output.layout !== 'time-quantile'
        || computeCanonicalSha256(observed.output.quantileLevels) !== computeCanonicalSha256(reference.quantiles)
        || observed.output.values.length !== expected.values.length) throw new Error('Qualification output shape mismatch.');
      for (const [i, actual] of observed.output.values.entries()) {
        if (!Number.isFinite(actual) || Math.abs(actual - expected.values[i]) >
          reference.tolerance.absolute + reference.tolerance.relative * Math.abs(expected.values[i])) {
          throw new Error('Retained numerical output fails the pinned oracle.');
        }
      }
    }
    const entry = await retain(surface, 'qualification-evidence', 'qualification/' + surface + '.json', bytes);
    qualification.push({ surface, status: 'passed', operation: 'forecast', forecastCases: report.rows.length,
      evidenceArtifactId: entry.artifactId, evidenceHash: entry.hash, transcriptHash: computeCanonicalSha256(report.rows) });
  }
  await retain('application-contract', 'host-source', 'application/contract.json', new TextEncoder().encode(JSON.stringify(applicationContract)));
  const plan = { ...candidate.targetPlan, qualification };
  const { targetPlan, ...executable } = candidate;
  const pack = await signPackV3(buildPackV3({ ...executable, artifacts, targetPlans: [plan] }), signer);
  const identity = getPackIdentity(pack);
  const source = pack.modelIR.sourceIdentity;
  const app = { applicationId: applicationContract.applicationId, applicationRevision: applicationContract.applicationRevision,
    applicationRevisionDigest: computeCanonicalSha256(applicationContract),
    workload: { id: applicationContract.workload.id, digest: computeCanonicalSha256(applicationContract.workload) },
    oracle: { id: 'chronos-bolt-source-reference', digest: computeCanonicalSha256(reference) } };
  const issuedAtUtc = new Date().toISOString();
  const expiresAtUtc = args.expires;
  const offlineSeconds = applicationContract.revocation.offlineExpirySeconds;
  if (new Date(expiresAtUtc).toISOString() !== expiresAtUtc || Date.parse(expiresAtUtc) <= Date.parse(issuedAtUtc)
    || Date.parse(expiresAtUtc) - Date.parse(issuedAtUtc) > offlineSeconds * 1000) {
    throw new Error('Release expiry must follow issuance and fit the explicit offline lease.');
  }
  const release = { schema: 'doppler.pack-release/v1', source: {
    repository: source.repository, revision: source.revision,
    revisionDigest: computeCanonicalSha256({ repository: source.repository, revision: source.revision }),
    provenanceDigest: computeCanonicalSha256(pack.modelIR.provenance),
    license: { spdxId: 'Apache-2.0', name: 'Apache License 2.0',
      sourceUrl: 'https://huggingface.co/' + source.repository + '/tree/' + source.revision,
      textDigest: artifacts.find(a => a.artifactId === 'LICENSE').hash } }, application: app,
    exclusions: { rejectionTypes: ['acceptance-failed', 'application-gate-failed', 'artifact-invalid', 'evidence-expired', 'revoked', 'unsupported-device'], known: [] },
    lifecycle: { releaseVersion: '0.1.0', supersedes: null, migration: null,
      failedUpgrade: { preservePrevious: true, previousPackId: null, previousSemanticRoot: null } },
    revocation: { authorityId: signer.authority, policyDigest: computeCanonicalSha256(applicationContract.revocation),
      offlineExpirySeconds: offlineSeconds, failClosedAfterExpiry: true },
    stateSnapshot: { schema: 'doppler.pack-state-snapshot/v1', format: applicationContract.state.format,
      identityDigest: computeCanonicalSha256(applicationContract.state), portableAcrossTargetIds: [plan.targetId] } };
  const { schema, semanticRoot, envelopeDigest } = identity;
  const event = await signPackReleaseEvent({ pack: { schema, semanticRoot, envelopeDigest }, sequence: 1,
    previousEventDigest: null, issuedAtUtc, expiresAtUtc, action: 'eligible', release, migratedFrom: null, nextSigner: null }, signer);
  const profile = { schema: 'd4da.doppler-trust-profile/v1', authorityKind: 'local-qualification', identity,
    trustedSigners: { [signer.authority]: signer.publicKeyJwk }, application: app,
    acceptedTargetPlanDigests: [hashTargetPlan(plan)], minimumSequence: 1,
    qualificationScope: 'five CPU-reference cases on the retained Node and browser adapters; no forecasting-skill claim' };
  await fs.writeFile(path.join(root, 'pack.json'), JSON.stringify(pack));
  await fs.writeFile(path.join(root, 'release-events.json'), JSON.stringify([event]));
  await fs.writeFile(path.join(root, 'trust-profile.json'), JSON.stringify(profile, null, 2) + '\n');
  console.log(JSON.stringify({ pack: identity, artifactBytes: artifacts.reduce((sum, a) => sum + a.sizeBytes, 0), expiresAtUtc }));
}
