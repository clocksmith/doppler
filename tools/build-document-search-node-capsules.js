#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { generateKeyPairSync } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { rigModelCapsule } from '../src/tooling/model-capsule-rig.js';
import { hashBytesSha256, computeCanonicalSha256 } from '../src/formats/canonical-hash.js';
import { hashTargetPlan } from '../src/config/target-plan.js';

function child(root, relative) {
  const filename = path.resolve(root, relative);
  const rel = path.relative(root, filename);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error('Artifact path escapes its root.');
  return filename;
}

// Reconstruct Forge inputs from signed artifact descriptors. In particular,
// never regenerate a retained Program Bundle from checkout shader sources.
export async function buildDocumentSearchNodeCapsules(config) {
  for (const key of ['outputDir', 'sourceCapsulesDir', 'qualificationDir', 'applicationPath']) {
    if (!path.isAbsolute(config[key] ?? '')) throw new Error(`Absolute ${key} required.`);
  }
  if (!config.authorityId?.trim()) throw new Error('Explicit signing authority required.');
  await fs.mkdir(config.outputDir);
  const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
  const write = async (filename, value, mode = 0o644) => {
    await fs.mkdir(path.dirname(filename), { recursive: true });
    await fs.writeFile(filename, JSON.stringify(value, null, 2) + '\n', { flag: 'wx', mode });
  };
  const receipts = [];
  for (const role of ['embedding', 'reranker']) {
    const sourceRoot = path.join(config.sourceCapsulesDir, role);
    const old = await read(path.join(sourceRoot, 'capsule.json'));
    const bundleArtifact = old.artifacts.find(artifact => artifact.role === 'program-bundle');
    if (!bundleArtifact) throw new Error('Retained Program Bundle required.');
    const bundleBytes = await fs.readFile(child(sourceRoot, bundleArtifact.path));
    if (hashBytesSha256(bundleBytes) !== bundleArtifact.hash || bundleBytes.length !== bundleArtifact.sizeBytes) {
      throw new Error('Retained Program Bundle identity mismatch.');
    }
    const bundle = JSON.parse(bundleBytes);
    const root = path.join(config.outputDir, role);
    const input = path.join(root, 'inputs');
    const bundlePath = path.join(input, 'program-bundle.json');
    await fs.mkdir(input, { recursive: true });
    await fs.writeFile(bundlePath, bundleBytes, { flag: 'wx' });
    for (const descriptor of bundle.artifacts) {
      const signed = old.artifacts.find(artifact => artifact.role === descriptor.role
        && artifact.hash === descriptor.hash && artifact.sizeBytes === descriptor.sizeBytes);
      if (!signed) throw new Error(`No retained artifact matches ${descriptor.path}.`);
      const bytes = await fs.readFile(child(sourceRoot, signed.path));
      if (hashBytesSha256(bytes) !== signed.hash || bytes.length !== signed.sizeBytes) throw new Error(`Corrupt artifact: ${signed.path}.`);
      const destination = child(input, descriptor.path);
      await fs.mkdir(path.dirname(destination), { recursive: true });
      await fs.writeFile(destination, bytes, { flag: 'wx' });
    }
    const reportPath = path.join(config.qualificationDir, `${role}-model-qualification.json`);
    const report = await read(reportPath);
    if (!report.passed || report.runtime?.surface !== 'node-webgpu') throw new Error('Passing Node source qualification required.');
    const signing = generateKeyPairSync('ed25519');
    const publicKey = signing.publicKey.export({ format: 'jwk' });
    await fs.mkdir(path.join(root, 'custody'), { mode: 0o700 });
    await write(path.join(root, 'custody/private-key.json'), signing.privateKey.export({ format: 'jwk' }), 0o600);
    await write(path.join(root, 'custody/public-key.json'), publicKey);
    const release = structuredClone(old.release);
    const applicationDigest = hashBytesSha256(await fs.readFile(config.applicationPath));
    release.application = { ...release.application, applicationId: 'doppler-offline-document-search',
      applicationRevision: applicationDigest, applicationRevisionDigest: applicationDigest };
    release.revocation.authorityId = config.authorityId;
    const { policyDigest, ...revocation } = release.revocation;
    release.revocation.policyDigest = computeCanonicalSha256(revocation);
    release.exclusions.known = [{ code: 'unsupported-device', scope: 'outside-the-observed-node-webgpu-tuple',
      reason: 'Node source qualification; installed application acceptance is separate.',
      evidenceDigest: hashBytesSha256(await fs.readFile(reportPath)) }];
    await write(path.join(root, 'release.json'), release);
    const modelIR = old.artifacts.find(artifact => artifact.role === 'source-truth-evidence'
      && path.basename(artifact.path).startsWith('model-ir-'));
    const options = { repoRoot: input, programBundlePath: bundlePath,
      manifestPath: child(input, bundle.sources.manifest.path),
      ...(modelIR ? { modelIRReceiptPath: child(sourceRoot, modelIR.path) } : {}),
      qualificationReportPaths: [reportPath], initialExecutionIdentityPath: reportPath,
      releaseManifestPath: path.join(root, 'release.json'), outputPath: path.join(root, 'distribution/capsule.json'),
      signingAuthority: config.authorityId, signingPrivateKeyPath: path.join(root, 'custody/private-key.json'),
      signingPublicKeyPath: path.join(root, 'custody/public-key.json'), allowDevelopmentSigner: false };
    await write(path.join(root, 'forge-config.json'), options);
    const result = await rigModelCapsule(options);
    const capsule = await read(options.outputPath);
    await write(path.join(root, 'open-options.json'), { trustedSigners: { [config.authorityId]: publicKey },
      acceptedTargetPlanDigests: capsule.targetPlans.map(hashTargetPlan) });
    const license = await fs.readFile(path.join(sourceRoot, 'MODEL_LICENSE.txt'));
    if (hashBytesSha256(license) !== capsule.release.source.license.textDigest) throw new Error('License identity mismatch.');
    await fs.writeFile(path.join(root, 'distribution/MODEL_LICENSE.txt'), license, { flag: 'wx' });
    const receipt = { role, result, sourceBundleHash: bundleArtifact.hash,
      qualificationHash: hashBytesSha256(await fs.readFile(reportPath)),
      physicalCapsuleExecution: false, installedApplicationAcceptance: false, externalAdoption: false };
    await write(path.join(root, 'build-receipt.json'), receipt);
    receipts.push(receipt);
  }
  return receipts;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await buildDocumentSearchNodeCapsules(JSON.parse(await fs.readFile(process.argv[2], 'utf8')))));
}
