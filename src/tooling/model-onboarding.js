import fs from 'node:fs/promises';
import path from 'node:path';
import schema from '../config/forge/model-onboarding.schema.json' with { type: 'json' };
import { isPlainObject } from '../formats/plain-object.js';
import { hashBytesSha256, computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { forgeSourceTruthFromFiles } from './source-truth-inputs.js';
import { assessModelSupport } from '../converter/model-support-assessment.js';
import { materializeLineageConversionCandidate } from '../converter/lineage-lowering-forge.js';
import { retainOnboardingJson as retainJson } from './model-onboarding-custody.js';
import { runOnboardingExecution, validateOnboardingExecution } from './model-onboarding-execution.js';

function requireKeys(value, keys, label) {
  if (!isPlainObject(value) || keys.some((key) => !Object.hasOwn(value, key))
    || Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error(`${label} requires exactly: ${keys.join(', ')}.`);
  }
}

function validateConfig(config) {
  requireKeys(config, schema.required, 'Model onboarding config');
  if (config.schema !== schema.$id) throw new Error(`Model onboarding requires schema "${schema.$id}".`);
  if (!Array.isArray(config.entryPointIds) || config.entryPointIds.length === 0
    || Array.from(config.entryPointIds).some((id) => typeof id !== 'string' || !id.trim() || id.trim() !== id)
    || new Set(config.entryPointIds).size !== config.entryPointIds.length) {
    throw new Error('Model onboarding requires unique, explicit entryPointIds.');
  }
  const inputs = [config.sourceSpec, config.vocabulary];
  if (config.lineage !== null) {
    requireKeys(config.lineage, schema.properties.lineage.oneOf[1].required, 'lineage');
    inputs.push(config.lineage.recipe, config.lineage.template);
  }
  for (const input of inputs) {
    requireKeys(input, schema.$defs.input.required, 'Onboarding input');
    if (typeof input.path !== 'string' || !input.path.trim()
      || typeof input.digest !== 'string' || !new RegExp(schema.$defs.input.properties.digest.pattern).test(input.digest)) {
      throw new Error('Onboarding input requires a file path and pinned byte SHA-256 digest.');
    }
  }
}

async function readInput(input, sourceRoot) {
  const bytes = await fs.readFile(path.resolve(sourceRoot, input.path));
  if (hashBytesSha256(bytes) !== input.digest) throw new Error(`Onboarding input changed: ${input.path}`);
  return JSON.parse(bytes.toString('utf8'));
}

export async function runModelOnboarding(config, { sourceRoot, outputDir, execution }) {
  validateConfig(config);
  if (execution != null) {
    validateOnboardingExecution(execution.config);
    if (typeof execution.runStage !== 'function' || typeof execution.verifyStage !== 'function') {
      throw new Error('Explicit onboarding stage execution and verification ports required.');
    }
    execution = { ...execution, config: structuredClone(execution.config) };
  }
  if (typeof sourceRoot !== 'string' || !sourceRoot.trim() || typeof outputDir !== 'string' || !outputDir.trim()) {
    throw new Error('Model onboarding requires sourceRoot and outputDir.');
  }
  config = structuredClone(config);
  const inputDigest = computeCanonicalSha256(execution == null ? config : { config, execution: execution.config });
  await fs.mkdir(outputDir, { recursive: true });
  try {
    const previous = JSON.parse(await fs.readFile(path.join(outputDir, 'onboarding-result.json'), 'utf8'));
    if (previous.inputDigest !== inputDigest) throw new Error('Retained onboarding inputs differ. Preserve them and use a new output directory.');
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await retainJson(outputDir, 'onboarding-input.json', config);
  if (execution != null) await retainJson(outputDir, 'onboarding-execution-input.json', execution.config);
  let stage = 'source-facts';
  const outputs = {};
  try {
    const spec = await readInput(config.sourceSpec, sourceRoot);
    if (typeof spec.sourceIdentity?.revision !== 'string' || !/^[a-f0-9]{40}$/.test(spec.sourceIdentity.revision)) {
      throw new Error('Model onboarding requires an immutable source revision, not a branch or tag.');
    }
    const source = await forgeSourceTruthFromFiles(spec, sourceRoot);
    outputs.modelIR = await retainJson(outputDir, 'model-ir-receipt.json', source);
    stage = 'support-assessment';
    const vocabulary = await readInput(config.vocabulary, sourceRoot);
    const assessment = assessModelSupport({ modelIR: source.modelIR, unresolvedFacts: source.unresolvedFacts,
      entryPointIds: config.entryPointIds, vocabulary });
    outputs.assessment = await retainJson(outputDir, 'support-assessment.json', assessment);
    let status = assessment.tasks.length ? 'blocked' : 'recipe-required';
    if (assessment.tasks.length === 0 && config.lineage !== null) {
      stage = 'lineage-materialization';
      const [recipe, template] = await Promise.all([
        readInput(config.lineage.recipe, sourceRoot), readInput(config.lineage.template, sourceRoot),
      ]);
      const lineage = materializeLineageConversionCandidate({ modelIR: source.modelIR, recipe, template });
      outputs.lineage = await retainJson(outputDir, 'lineage-lowering-receipt.json', lineage);
      outputs.conversion = await retainJson(outputDir, 'conversion-config.json', lineage.conversionConfig);
      status = 'candidate-materialized';
    }
    const result = {
      schema: 'doppler.model-onboarding-result/v1', inputDigest, status, outputs,
      sourceIdentity: source.modelIR.sourceIdentity,
      manualRequirements: assessment.tasks.length ? assessment.tasks : [
        ...(config.lineage === null ? [{ kind: 'lineage-recipe', reason: 'Declare an attributable recipe and pinned template for the compatible operations.' }] : []),
        { kind: 'physical-reference-comparison', reason: 'Convert and run the candidate against unchanged source references.' },
        { kind: 'capsule-qualification', reason: 'Qualify device plans and sign the complete executable closure before publication.' },
      ],
      qualified: false, published: false,
    };
    if (execution != null && status === 'candidate-materialized') {
      stage = 'execution';
      const executionResult = await runOnboardingExecution(execution.config, { sourceRoot,
        outputDir: path.join(outputDir, 'execution'), sourceIdentity: source.modelIR.sourceIdentity,
        conversionPath: path.join(outputDir, outputs.conversion.path), conversionDigest: outputs.conversion.digest,
        runStage: execution.runStage, verifyStage: execution.verifyStage });
      result.schema = 'doppler.model-onboarding-result/v2';
      result.status = 'capsule-qualified'; result.qualified = true; result.manualRequirements = [];
      outputs.execution = await retainJson(outputDir, 'execution-summary.json', executionResult);
    }
    await retainJson(outputDir, 'onboarding-result.json', result);
    return result;
  } catch (error) {
    const failure = { schema: 'doppler.model-onboarding-failure/v1', inputDigest, stage, message: error.message, outputs };
    try {
      await retainJson(outputDir, `failure-${computeCanonicalSha256(failure).slice(7)}.json`, failure);
    } catch (retentionError) {
      throw new AggregateError([error, retentionError], error.message, { cause: error });
    }
    throw error;
  }
}
