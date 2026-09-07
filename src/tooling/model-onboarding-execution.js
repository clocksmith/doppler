import fs from 'node:fs/promises';
import path from 'node:path';
import schema from '../config/forge/model-onboarding-execution.schema.json' with { type: 'json' };
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { isPlainObject } from '../formats/plain-object.js';
import { hashOnboardingFile, inventoryOnboardingFiles, retainOnboardingJson } from './model-onboarding-custody.js';

function keys(value, required, label) {
  if (!isPlainObject(value) || required.some(key => !Object.hasOwn(value, key))
    || Object.keys(value).some(key => !required.includes(key))) throw new Error(`${label} requires exactly: ${required.join(', ')}.`);
}

export function validateOnboardingExecution(config) {
  keys(config, schema.required, 'Onboarding execution');
  for (const field of ['schema', 'operation', 'stages']) {
    if (computeCanonicalSha256(config[field]) !== computeCanonicalSha256(schema.properties[field].const)) {
      throw new Error(`Unsupported onboarding execution ${field}.`);
    }
  }
  keys(config.inputs, schema.properties.inputs.required, 'Execution inputs');
  if (!Array.isArray(config.sourceFiles) || !config.sourceFiles.length) throw new Error('Pinned source files required.');
  for (const input of Object.values(config.inputs)) keys(input, schema.$defs.input.required, 'Execution input');
  for (const input of config.sourceFiles) {
    keys(input, schema.properties.sourceFiles.items.required, 'Execution source file');
    if (!Number.isSafeInteger(input.sizeBytes) || input.sizeBytes < 0) throw new Error('Source file size required.');
  }
  for (const input of [...Object.values(config.inputs), ...config.sourceFiles]) {
    if (typeof input.path !== 'string' || !input.path.trim() || !/^sha256:[a-f0-9]{64}$/.test(input.digest)) {
      throw new Error('Execution inputs require paths and pinned byte SHA-256 digests.');
    }
  }
  if (new Set(config.sourceFiles.map(file => file.path)).size !== config.sourceFiles.length) throw new Error('Duplicate source file.');
}

async function verifyInputs(config, sourceRoot) {
  for (const input of [...Object.values(config.inputs), ...config.sourceFiles]) {
    const filename = path.resolve(sourceRoot, input.path);
    if ((input.sizeBytes !== undefined && (await fs.stat(filename)).size !== input.sizeBytes)
      || await hashOnboardingFile(filename) !== input.digest) throw new Error(`Onboarding execution input changed: ${input.path}`);
  }
}

function ownedPath(root, filename) {
  if (typeof filename !== 'string') throw new Error('Stage receipt path required.');
  const absolute = path.resolve(root, filename);
  const relative = path.relative(root, absolute);
  if (!relative || relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
    throw new Error('Stage receipt must be inside its owned attempt directory.');
  }
  return absolute;
}

export async function runOnboardingExecution(config, { sourceRoot, outputDir, sourceIdentity, conversionPath, conversionDigest, runStage, verifyStage }) {
  validateOnboardingExecution(config);
  config = structuredClone(config); sourceIdentity = structuredClone(sourceIdentity);
  if (typeof runStage !== 'function' || typeof verifyStage !== 'function') throw new Error('Explicit stage execution and evidence verification ports required.');
  async function verifyPinnedInputs() {
    await verifyInputs(config, sourceRoot);
    if (await hashOnboardingFile(conversionPath) !== conversionDigest) throw new Error('Materialized conversion configuration changed.');
  }
  await verifyPinnedInputs();
  await fs.mkdir(outputDir, { recursive: true });
  const inputDigest = computeCanonicalSha256({ config, sourceIdentity, conversionDigest });
  await retainOnboardingJson(outputDir, 'execution-input.json', { config, sourceIdentity, inputDigest });
  const completed = {};
  const stages = [];
  const inputs = Object.fromEntries(Object.entries(config.inputs).map(([key, input]) => [key, path.resolve(sourceRoot, input.path)]));
  for (const stage of config.stages) {
    const stageRoot = path.join(outputDir, stage);
    await fs.mkdir(stageRoot, { recursive: true });
    const dependencyDigest = computeCanonicalSha256(stages);
    let checkpoint;
    try { checkpoint = JSON.parse(await fs.readFile(path.join(stageRoot, 'complete.json'), 'utf8')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    let attemptDir;
    if (checkpoint) {
      if (checkpoint.inputDigest !== inputDigest || checkpoint.dependencyDigest !== dependencyDigest) throw new Error(`Retained stage inputs differ: ${stage}`);
      attemptDir = ownedPath(stageRoot, checkpoint.attempt);
      if (computeCanonicalSha256(await inventoryOnboardingFiles(attemptDir)) !== computeCanonicalSha256(checkpoint.files)) {
        throw new Error(`Retained stage output changed: ${stage}`);
      }
    } else {
      for (let attempt = 0;; attempt++) {
        attemptDir = path.join(stageRoot, `attempt-${attempt}`);
        try { await fs.mkdir(attemptDir); break; }
        catch (error) { if (error.code !== 'EEXIST') throw error; }
      }
    }
    const context = { stage, sourceRoot, outputDir: attemptDir, sourceIdentity: structuredClone(sourceIdentity),
      conversionPath, inputs: { ...inputs }, completed: structuredClone(completed) };
    try {
      const receiptPath = checkpoint ? ownedPath(attemptDir, checkpoint.receiptPath)
        : ownedPath(attemptDir, await runStage(context));
      const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
      if (receipt.schema !== 'doppler.onboarding-stage-result/v1' || receipt.stage !== stage || receipt.passed !== true
        || receipt.source?.repository !== sourceIdentity.repository || receipt.source?.revision !== sourceIdentity.revision) {
        throw new Error(`Stage evidence does not bind the declared source: ${stage}`);
      }
      if (['model-qualification', 'capsule-qualification'].includes(stage) && receipt.physicalExecution !== true) {
        throw new Error(`Physical execution evidence required: ${stage}`);
      }
      if (await verifyStage(context, receipt) !== true) throw new Error(`Stage evidence rejected: ${stage}`);
      await verifyPinnedInputs();
      if (!checkpoint) {
        checkpoint = { schema: 'doppler.onboarding-stage-checkpoint/v1', stage, inputDigest, dependencyDigest,
          attempt: path.basename(attemptDir), receiptPath: path.relative(attemptDir, receiptPath), files: await inventoryOnboardingFiles(attemptDir) };
        await retainOnboardingJson(stageRoot, 'complete.json', checkpoint);
      }
      completed[stage] = receipt;
      stages.push({ stage, checkpointDigest: computeCanonicalSha256(checkpoint), receiptPath: path.relative(outputDir, receiptPath) });
    } catch (error) {
      if (!checkpoint) {
        try { await retainOnboardingJson(attemptDir, 'failure.json', { stage, inputDigest, dependencyDigest, message: error.message }); }
        catch (retentionError) { throw new AggregateError([error, retentionError], error.message, { cause: error }); }
      }
      throw error;
    }
  }
  const result = { schema: 'doppler.model-onboarding-execution-result/v1', inputDigest, sourceIdentity, stages, qualified: true, published: false };
  await retainOnboardingJson(outputDir, 'execution-result.json', result);
  return result;
}
