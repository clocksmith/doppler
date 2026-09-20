#!/usr/bin/env node
// Local evaluation custody only. Does not publish or promote a model release.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { generateKeyPairSync } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { writeProgramBundle } from '../src/tooling/program-bundle.js';
import { forgeModelCapsule } from '../src/tooling/model-capsule-forge.js';
import { hashTargetPlan } from '../src/config/target-plan.js';
import { getCapsuleIdentity } from '../src/config/capsule.js';
import { resolveCapsuleAdapterSet } from '../src/config/capsule-adapters.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';

const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const config = await read(process.argv[2]);
assert.equal(new Date(config.createdAtUtc).toISOString(), config.createdAtUtc, 'createdAtUtc must be a canonical ISO instant');
const probe = await read(config.probePath);
const previous = await read(config.previousQualificationPath);
const reference = await read(config.sourceTranscriptPath);
assert(probe.passed && probe.sourceParity?.passed && !probe.hardware.isFallbackAdapter);
if (config.adapterExecution) assert(probe.adapterPreparation?.passed, 'Physical adapter preparation evidence required');
assert.deepEqual(probe.sourceParity.promptTokenIds, reference.promptTokenIds);
assert.deepEqual(probe.sourceParity.tokenIds, reference.generatedTokenIds);
assert.deepEqual(probe.sourceParity.tokenIds, previous.metrics.referenceTranscript.tokens.ids);
const sourcePath = path.join(probe.config.sourceRoot, probe.config.sourceFile);
assert.equal(hashBytesSha256(await fs.readFile(sourcePath)), `sha256:${probe.sourceSha256}`);
const source = await read(sourcePath);
const output = path.resolve(config.outputDir);
await fs.mkdir(output);
await fs.mkdir(path.join(output, 'custody'), { mode: 0o700 });
await fs.mkdir(path.join(output, 'source-kernels'));
const write = (file, data, mode = 0o644) => fs.writeFile(path.join(output, file), JSON.stringify(data, null, 2) + '\n', { mode });
const signing = generateKeyPairSync('ed25519');
const publicKey = signing.publicKey.export({ format: 'jwk' });
await write('custody/private-key.json', signing.privateKey.export({ format: 'jwk' }), 0o600);
await write('custody/public-key.json', publicKey);
for (const module of source.wgslModules) {
  const artifact = source.artifacts.find(row => row.artifactId === module.sourceArtifactId);
  const bytes = await fs.readFile(path.join(probe.config.sourceRoot, artifact.path));
  assert.equal(hashBytesSha256(bytes), module.sourceHash);
  await fs.writeFile(path.join(output, 'source-kernels', module.file), bytes);
}
const manifestArtifact = source.artifacts.find(row => row.artifactId === source.program.manifestArtifactId);
const manifestPath = path.join(probe.config.sourceRoot, manifestArtifact.path);
assert.equal(hashBytesSha256(await fs.readFile(manifestPath)), manifestArtifact.hash);
const qualification = {
  schema: 'doppler.gpu-token-selection-qualification/v1', passed: true,
  modelId: source.modelId, env: { runtime: 'browser-webgpu' }, deviceInfo: probe.hardware,
  initialExecutionIdentity: probe.initialExecutionIdentity,
  // Same verified tokenizer and identical IDs: retain the reference rendering;
  // actual generated IDs and prompt IDs are separately retained in the probe.
  output: previous.output,
  metrics: { prompt: previous.metrics.prompt, tokensGenerated: probe.sourceParity.tokenIds.length,
    referenceTranscript: { generationConfig: previous.metrics.referenceTranscript.generationConfig,
      prompt: previous.metrics.referenceTranscript.prompt, tokens: { ids: probe.sourceParity.tokenIds },
      output: { tokensGenerated: probe.sourceParity.tokenIds.length, stopReason: 'stop-token',
        stopTokenId: probe.sourceParity.tokenIds.at(-1) } },
    sourceParity: { ...previous.metrics.sourceParity, expectedTranscriptPath: config.sourceTranscriptPath,
      expectedTranscriptHash: hashBytesSha256(await fs.readFile(config.sourceTranscriptPath)) } },
  evidence: { probePath: config.probePath, probeHash: hashBytesSha256(await fs.readFile(config.probePath)),
    runtimeArchive: probe.package, referenceRendering: 'Identical token IDs and tokenizer; text retained from the reference.',
    scope: config.adapterExecution
      ? 'Internal AMD Chrome zero-delta PEFT fixture and base greedy parity; no trained-adapter quality, nonzero adapter parity, publication or fleet qualification.'
      : 'Internal AMD Chrome greedy source parity and frozen workload equivalence; no publication, fleet qualification or external adoption.',
    ...(config.adapterExecution ? { adapterPreparation: probe.adapterPreparation } : {}) },
};
await write('qualification.json', qualification);
const release = await read(config.previousReleasePath);
const workloadHash = computeCanonicalSha256({ prompt: qualification.metrics.prompt, tokenIds: reference.generatedTokenIds });
release.application = { applicationId: config.adapterExecution ? 'doppler-adapter-fixture-evaluation' : 'doppler-gpu-token-selection-evaluation', applicationRevision: `sha256:${probe.probeSha256}`,
  applicationRevisionDigest: `sha256:${probe.probeSha256}`, workload: { id: 'frozen-source-generation', digest: workloadHash },
  oracle: { id: 'pinned-source-token-reference', digest: qualification.metrics.sourceParity.expectedTranscriptHash } };
release.exclusions.known = [{ code: 'unsupported-device', scope: 'outside-observed-chrome-amd',
  reason: qualification.evidence.scope, evidenceDigest: hashBytesSha256(await fs.readFile(path.join(output, 'qualification.json'))) }];
release.revocation.authorityId = config.authority;
const { policyDigest: ignored, ...revocation } = release.revocation;
release.revocation.policyDigest = computeCanonicalSha256(revocation);
release.stateSnapshot.identityDigest = computeCanonicalSha256({ application: release.application, state: 'generation-evaluation' });
await write('release.json', release);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const bundle = await writeProgramBundle({ repoRoot, manifestPath, modelDir: path.dirname(manifestPath),
  kernelSourceRoot: path.join(output, 'source-kernels'), referenceReportPath: path.join(output, 'qualification.json'),
  createdAtUtc: config.createdAtUtc, outputPath: path.join(output, 'build/program-bundle.json') });
const forge = { repoRoot, manifestPath, modelDir: path.dirname(manifestPath), programBundlePath: bundle.outputPath,
  initialExecutionIdentityPath: config.probePath, referenceReportPath: path.join(output, 'qualification.json'),
  releaseManifestPath: path.join(output, 'release.json'), outputPath: path.join(output, 'distribution/capsule.json'),
  tokenSelection: config.tokenSelection, ...(config.adapterExecution ? { adapterExecution: config.adapterExecution } : {}),
  signingPrivateKeyPath: path.join(output, 'custody/private-key.json'),
  signingPublicKeyPath: path.join(output, 'custody/public-key.json'), signingAuthority: config.authority, allowDevelopmentSigner: false };
await write('forge-config.json', forge);
const result = await forgeModelCapsule(forge);
const capsule = await read(forge.outputPath);
if (config.adapterExecution) {
  const manifest = await read(probe.config.adapterManifestPath);
  const bytes = await fs.readFile(path.join(probe.config.sourceRoot, manifest.weightsPath));
  const digest = hashBytesSha256(bytes);
  assert.equal(digest, probe.adapterPreparation.sourceDigest);
  assert.equal(digest, manifest.checksum);
  assert.equal(bytes.length, manifest.weightsSize);
  manifest.weightsPath = 'adapters/zero-delta.safetensors';
  const adapter = { schema: 'doppler.capsule-adapter/v1',
    identity: computeCanonicalSha256({ manifest, purpose: 'controlled-zero-delta-fixture' }),
    baseModel: { ...getCapsuleIdentity(capsule), modelId: capsule.modelId },
    format: 'peft_safetensors', manifest,
    artifact: { artifactId: manifest.id, role: 'lora-weights', path: manifest.weightsPath,
      hash: digest, sizeBytes: bytes.length } };
  resolveCapsuleAdapterSet([adapter], { capsule: { ...capsule, ...getCapsuleIdentity(capsule) },
    targetPlan: capsule.targetPlans[0], operation: 'generate' });
  await fs.mkdir(path.join(output, 'distribution/adapters'));
  await fs.writeFile(path.join(output, 'distribution', manifest.weightsPath), bytes, { flag: 'wx' });
  await write('adapter-descriptor.json', adapter);
}
await write('open-options.json', { trustedSigners: { [config.authority]: publicKey }, acceptedTargetPlanDigests: capsule.targetPlans.map(hashTargetPlan) });
await write('build-receipt.json', { result, physicalCapsuleExecution: false, publication: false, sourceProbe: qualification.evidence });
console.log(JSON.stringify(result));
