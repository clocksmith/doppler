#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { generateKeyPairSync } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { writeProgramBundle } from '../src/tooling/program-bundle.js';
import { forgeModelPack } from '../src/tooling/model-pack-forge.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { hashTargetPlan } from '../src/config/target-plan.js';
import { getPackIdentity } from '../src/config/pack.js';
import { evaluateRerankReference } from '../src/config/rerank-reference.js';
import { evaluateEmbeddingReference } from '../src/config/embedding-reference.js';
import kernelRegistry from '../src/config/kernels/registry.json' with { type: 'json' };

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function assertRerankerObservedShaderClosure(report, modules) {
  if (!Array.isArray(report.requests)) throw new Error('Evaluation Pack requires observed shader requests.');
  const probes = new Set(Object.values(kernelRegistry.operations.runtime_probe.variants).map(variant => variant.wgsl));
  const requested = [...new Set(report.requests.map(request => path.posix.basename(decodeURIComponent(new URL(request).pathname)))
    .filter(file => file.endsWith('.wgsl')))].sort();
  const modelShaders = requested.filter(file => !probes.has(file));
  if (!modelShaders.length) throw new Error('Evaluation Pack requires observed model shader requests.');
  const declared = new Set(modules.map(module => module.file));
  const missing = modelShaders.filter(file => !declared.has(file));
  if (missing.length) throw new Error(`Evaluation Pack closure omits observed model shaders: ${missing.join(', ')}.`);
  return { modelShaders, deviceProbes: requested.filter(file => probes.has(file)),
    scope: 'Observed qualification path only; not all shapes or devices.' };
}

export async function buildRerankerEvaluationPack(config) {
  return buildOperationEvaluationPack(config, 'rerank');
}

export async function buildEmbeddingEvaluationPack(config) {
  if (typeof config?.applicationId !== 'string' || !config.applicationId.trim()) throw new Error('Embedding Pack requires an explicit applicationId.');
  return buildOperationEvaluationPack(config, 'embed');
}

async function buildOperationEvaluationPack(config, operation) {
  for (const field of ['qualificationPath', 'conversionConfigPath', 'licensePath', 'applicationPath', 'outputDir', 'authorityId']) {
    if (typeof config?.[field] !== 'string' || !config[field].trim()) throw new Error(`Evaluation Pack requires ${field}.`);
  }
  if (!Number.isSafeInteger(config.revocation?.offlineExpirySeconds)
    || config.revocation.offlineExpirySeconds <= 0 || config.revocation.failClosedAfterExpiry !== true) {
    throw new Error('Evaluation Pack requires an explicit fail-closed revocation policy.');
  }
  const report = JSON.parse(await fs.readFile(config.qualificationPath, 'utf8'));
  const expectedSchema = operation === 'embed' ? 'doppler.embeddingModelQualification.v1' : 'doppler.rerankModelQualification.v1';
  const evaluate = operation === 'embed' ? evaluateEmbeddingReference : evaluateRerankReference;
  if (report.schema !== expectedSchema || !report.passed
    || !evaluate(report.reference, report.observation).passed) {
    throw new Error(`Evaluation Pack requires passing ${operation} source qualification.`);
  }
  if (!['browser-webgpu', 'node-webgpu'].includes(report.runtime?.surface)) {
    throw new Error('Evaluation Pack requires an explicitly observed WebGPU surface.');
  }
  const outputDir = path.resolve(config.outputDir);
  await fs.mkdir(path.dirname(outputDir), { recursive: true });
  await fs.mkdir(outputDir);
  await fs.mkdir(path.join(outputDir, 'custody'), { mode: 0o700 });
  const write = (name, value, mode = 0o644) => fs.writeFile(path.join(outputDir, name), `${JSON.stringify(value, null, 2)}\n`, { mode });
  const signing = generateKeyPairSync('ed25519');
  const publicKey = signing.publicKey.export({ format: 'jwk' });
  await write('custody/private-key.json', signing.privateKey.export({ format: 'jwk' }), 0o600);
  await write('custody/public-key.json', publicKey);
  const manifestPath = path.resolve(report.config.modelDir, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const bundle = await writeProgramBundle({ repoRoot: ROOT, manifestPath, modelDir: path.dirname(manifestPath),
    conversionConfigPath: path.resolve(config.conversionConfigPath), referenceReportPath: path.resolve(config.qualificationPath),
    createdAtUtc: report.generatedAt, outputPath: path.join(outputDir, 'build/program-bundle.json') });
  const observedShaderClosure = assertRerankerObservedShaderClosure(report, bundle.bundle.wgslModules);
  const targetId = `webgpu-${manifest.inference.session.compute.defaults.activationDtype}-${manifest.inference.session.kvcache.kvDtype}-${bundle.bundle.wgslModules.some((module) => module.metadata.requiresSubgroups) ? 'subgroups' : 'portable'}`;
  const applicationDigest = hashBytesSha256(await fs.readFile(config.applicationPath));
  const application = {
    applicationId: operation === 'embed' ? config.applicationId : 'doppler-electron-reranker-evaluation', applicationRevision: applicationDigest,
    applicationRevisionDigest: applicationDigest,
    workload: { id: operation === 'embed' ? 'frozen-document-embedding' : 'frozen-document-reranking', digest: computeCanonicalSha256(report.reference.input) },
    oracle: { id: 'pinned-hf-source-comparison', digest: report.referenceDigest },
  };
  const revocation = { authorityId: config.authorityId,
    offlineExpirySeconds: config.revocation.offlineExpirySeconds,
    failClosedAfterExpiry: config.revocation.failClosedAfterExpiry };
  const release = {
    schema: 'doppler.pack-release/v1',
    source: { repository: report.reference.source.repository, revision: report.reference.source.revision,
      revisionDigest: computeCanonicalSha256(report.reference.source.files), provenanceDigest: computeCanonicalSha256(report.reference.source),
      license: { spdxId: 'Apache-2.0', name: 'Apache License 2.0', sourceUrl: 'https://www.apache.org/licenses/LICENSE-2.0.txt',
        textDigest: hashBytesSha256(await fs.readFile(config.licensePath)) } },
    application,
    exclusions: { rejectionTypes: ['acceptance-failed', 'application-gate-failed', 'artifact-invalid', 'evidence-expired', 'migration-required', 'revoked', 'unsupported-device'],
      known: [{ code: 'unsupported-device', scope: `outside-the-observed-${report.runtime.surface}-tuple`,
        reason: 'Internal physical evaluation only; other hosts, fleet support and adoption are unestablished.',
        evidenceDigest: hashBytesSha256(await fs.readFile(config.qualificationPath)) }] },
    lifecycle: { releaseVersion: '1.0.0', supersedes: null, migration: null,
      failedUpgrade: { preservePrevious: true, previousPackId: null, previousSemanticRoot: null } },
    revocation: { ...revocation, policyDigest: computeCanonicalSha256(revocation) },
    stateSnapshot: { schema: 'doppler.pack-state-snapshot/v1', format: 'canonical-json',
      identityDigest: computeCanonicalSha256({ application, state: operation === 'embed' ? 'stateless-embedding' : 'stateless-reranking' }), portableAcrossTargetIds: [targetId] },
  };
  await write('release.json', release);
  const forge = { repoRoot: ROOT, manifestPath, modelDir: path.dirname(manifestPath),
    ...(config.modelIRReceiptPath ? { modelIRReceiptPath: path.resolve(config.modelIRReceiptPath) } : {}),
    programBundlePath: bundle.outputPath, referenceReportPath: path.resolve(config.qualificationPath),
    releaseManifestPath: path.join(outputDir, 'release.json'),
    initialExecutionIdentityPath: path.resolve(config.qualificationPath), outputPath: path.join(outputDir, 'distribution/pack.json'),
    signingPrivateKeyPath: path.join(outputDir, 'custody/private-key.json'), signingPublicKeyPath: path.join(outputDir, 'custody/public-key.json'),
    signingAuthority: config.authorityId, allowDevelopmentSigner: false };
  await write('forge-config.json', forge);
  const result = await forgeModelPack(forge);
  const pack = JSON.parse(await fs.readFile(forge.outputPath, 'utf8'));
  const licensePath = path.join(outputDir, 'distribution/MODEL_LICENSE.txt');
  await fs.copyFile(config.licensePath, licensePath);
  if (hashBytesSha256(await fs.readFile(licensePath)) !== pack.release.source.license.textDigest) {
    throw new Error('Retained model license differs from its signed release digest.');
  }
  await write('open-options.json', { trustedSigners: { [config.authorityId]: publicKey },
    acceptedTargetPlanDigests: pack.targetPlans.map(hashTargetPlan) });
  await write('build-receipt.json', { result, packIdentity: getPackIdentity(pack), application, observedShaderClosure,
    modelLicense: { path: licensePath, digest: pack.release.source.license.textDigest,
      binding: 'release.source.license.textDigest', artifactInventoryMember: false },
    evidenceClass: 'internal-source-qualified-evaluation-pack', physicalPackExecution: false, externalAdoption: false });
  return result;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
  try {
    if (config.operation !== undefined && !['embed', 'rerank'].includes(config.operation)) throw new Error('Unsupported evaluation Pack operation.');
    console.log(JSON.stringify(await (config.operation === 'embed' ? buildEmbeddingEvaluationPack(config) : buildRerankerEvaluationPack(config))));
  } catch (error) {
    if (error.code !== 'EEXIST') {
      await fs.writeFile(path.join(config.outputDir, 'build-failure.json'), JSON.stringify({
        passed: false, config, error: { name: error.name, message: error.message, stack: error.stack },
      }, null, 2), { flag: 'wx' }).catch((custodyError) => console.error('Failure custody:', custodyError.message));
    }
    console.error(error);
    process.exitCode = 1;
  }
}
