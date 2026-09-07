import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { hashOnboardingFile } from '../src/tooling/model-onboarding-custody.js';
import { computeCanonicalSha256 } from '../src/formats/canonical-hash.js';
import { assertEmbeddingReference, assertEmbeddingSourceIdentity, evaluateEmbeddingReference } from '../src/config/embedding-reference.js';
import { getCapsuleIdentity, verifyCapsuleMetadata } from '../src/config/capsule.js';
import { qualifyEmbeddingBrowser } from './qualify-embedding-browser.js';
import { buildEmbeddingEvaluationCapsule } from './build-reranker-evaluation-capsule.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const json = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const equal = (actual, expected, label) => assert.equal(computeCanonicalSha256(actual), computeCanonicalSha256(expected), label);
const write = (filename, value) => fs.writeFile(filename, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });

async function command(context, executable, args) {
  await write(path.join(context.outputDir, 'command.json'), { executable, args, cwd: context.sourceRoot });
  const log = await fs.open(path.join(context.outputDir, 'command.log'), 'wx');
  try {
    await new Promise((resolve, reject) => {
      const child = spawn(executable, args, { cwd: context.sourceRoot, stdio: ['ignore', log.fd, log.fd] });
      child.once('error', reject);
      child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`Onboarding ${context.stage} exited ${code ?? signal}.`)));
    });
  } finally { await log.close(); }
}

async function policies(context) {
  const { inputs, sourceIdentity } = context;
  assert.equal(inputs.driver, fileURLToPath(import.meta.url), 'Execution must pin this concrete driver.');
  const source = await json(inputs.sourcePolicy);
  const reference = assertEmbeddingReference(await json(inputs.frozenReference));
  assert.equal(source.repository, sourceIdentity.repository);
  assert.equal(source.revision, sourceIdentity.revision);
  assert.equal(reference.source.repository, source.repository);
  assert.equal(reference.source.revision, source.revision);
  for (const field of ['input', 'embeddingContract', 'tolerances']) equal(source[field], reference[field], `Frozen ${field} differs.`);
  const qualification = await json(inputs.qualificationConfig);
  const installed = await json(inputs.installedPackageReceipt);
  const bundle = path.dirname(inputs.installedPackageReceipt);
  assert.equal(inputs.installedPackageArchive, path.join(bundle, installed.package.filename));
  assert(installed.passed);
  assert.equal(await hashOnboardingFile(inputs.installedPackageArchive), 'sha256:' + installed.package.sha256);
  return { source, reference, qualification, installed, bundle };
}

export async function runEmbeddingOnboardingStage(context) {
  console.log(JSON.stringify({ onboardingStage: context.stage, attemptDirectory: context.outputDir }));
  const policy = await policies(context);
  const { outputDir, inputs, completed, stage } = context;
  const receipt = { schema: 'doppler.onboarding-stage-result/v1', stage, passed: true,
    source: { repository: policy.source.repository, revision: policy.source.revision }, physicalExecution: false, data: {} };
  const modelDir = stage === 'conversion' ? path.join(outputDir, 'model') : completed.conversion.data.modelDir;
  if (stage === 'conversion') {
    await command(context, process.execPath, [path.join(policy.bundle, 'consumer/node_modules/doppler-gpu/tools/convert-safetensors-node.js'),
      path.resolve(context.sourceRoot, policy.source.sourceDirectory), '--config', context.conversionPath, '--output-dir', modelDir]);
    receipt.data = { modelDir, manifestPath: path.join(modelDir, 'manifest.json') };
  } else if (stage === 'source-reference') {
    const referencePath = path.join(outputDir, 'reference.json');
    await command(context, 'python3', [path.join(ROOT, 'tools/capture-embedding-source-reference.py'), '--policy', inputs.sourcePolicy, '--out', referencePath]);
    receipt.data = { referencePath };
  } else if (stage === 'capsule-construction') {
    const config = { ...await json(inputs.capsuleConfig), qualificationPath: completed['model-qualification'].data.qualificationPath,
      conversionConfigPath: context.conversionPath, modelIRReceiptPath: path.join(path.dirname(context.conversionPath), 'model-ir-receipt.json'),
      licensePath: inputs.license, applicationPath: inputs.application, outputDir: path.join(outputDir, 'capsule') };
    await write(path.join(outputDir, 'config.json'), config);
    await buildEmbeddingEvaluationCapsule(config);
    receipt.data = { capsulePath: path.join(config.outputDir, 'distribution/capsule.json'),
      buildReceiptPath: path.join(config.outputDir, 'build-receipt.json'), openOptionsPath: path.join(config.outputDir, 'open-options.json') };
  } else {
    const mode = stage === 'model-qualification' ? 'model' : 'capsule';
    const config = { ...policy.qualification, mode, packageBundlePath: policy.bundle, modelDir,
      referencePath: inputs.frozenReference, outputDir: path.join(outputDir, 'browser') };
    if (mode === 'capsule') {
      const built = completed['capsule-construction'].data;
      config.capsulePath = built.capsulePath;
      config.application = (await json(built.buildReceiptPath)).application;
      config.openOptions = await json(built.openOptionsPath);
    }
    await write(path.join(outputDir, 'config.json'), config);
    await qualifyEmbeddingBrowser(config);
    receipt.physicalExecution = true;
    receipt.data = { qualificationPath: path.join(config.outputDir, 'qualification.json') };
  }
  const receiptPath = path.join(outputDir, 'stage-result.json');
  await write(receiptPath, receipt);
  return receiptPath;
}

export async function verifyEmbeddingOnboardingStage(context, receipt) {
  const policy = await policies(context);
  if (context.stage === 'conversion') {
    const manifest = await json(receipt.data.manifestPath);
    assertEmbeddingSourceIdentity(manifest.artifactIdentity, policy.reference);
    assert.equal(manifest.hashAlgorithm, 'sha256');
    assert(manifest.shards.length > 0);
    for (const shard of manifest.shards) {
      assert.equal(await hashOnboardingFile(path.join(receipt.data.modelDir, shard.filename)), 'sha256:' + shard.hash);
    }
  } else if (context.stage === 'source-reference') {
    const reference = assertEmbeddingReference(await json(receipt.data.referencePath));
    equal(reference.source, policy.reference.source, 'Independent source stack or checkpoint differs.');
    equal(reference.tolerances, policy.reference.tolerances);
    assert(evaluateEmbeddingReference(policy.reference, reference).passed);
  } else if (context.stage === 'capsule-construction') {
    const capsule = await json(receipt.data.capsulePath);
    const built = await json(receipt.data.buildReceiptPath);
    await verifyCapsuleMetadata(capsule, await json(receipt.data.openOptionsPath));
    equal(getCapsuleIdentity(capsule), built.capsuleIdentity);
    assert.equal(capsule.release.source.repository, policy.source.repository);
    assert.equal(capsule.release.source.revision, policy.source.revision);
    assert.equal(capsule.release.application.applicationRevisionDigest, await hashOnboardingFile(context.inputs.application));
    assert.equal(capsule.release.source.license.textDigest, await hashOnboardingFile(context.inputs.license));
    assert(built.result.ok);
  } else {
    const report = await json(receipt.data.qualificationPath);
    assert(report.passed && report.cleanup.passed);
    equal(report.reference, policy.reference);
    equal(report.installedPackage, policy.installed.package);
    assert(evaluateEmbeddingReference(policy.reference, report.observation).passed);
    assert.equal(report.runtime.adapterInfo.isFallbackAdapter, false);
    assert.equal(report.runtime.adapterInfo.vendor.toLowerCase(), policy.qualification.requiredVendor);
    const manifestPath = context.completed.conversion.data.manifestPath;
    assert.equal(report.model.manifestHash, await hashOnboardingFile(manifestPath));
    if (context.stage === 'capsule-qualification') {
      const built = context.completed['capsule-construction'].data;
      assert(report.boundary.signedCapsuleExecution);
      equal(report.raw.capsuleIdentity, (await json(built.buildReceiptPath)).capsuleIdentity);
      assert((await json(built.openOptionsPath)).acceptedTargetPlanDigests.includes(report.raw.selectedTargetPlanDigest));
    }
  }
  return true;
}
