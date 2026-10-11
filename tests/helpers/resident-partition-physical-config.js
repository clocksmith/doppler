/** Test-only input binding for the existing same-device resident diagnostic.
 * DOPPLER_PARTITION_CONFIG names JSON with schema, exact manifest modelIdentity,
 * splitLayer, generation, limits, prompts, and reference (null or {path, sha256}).
 * Reference paths are relative to that config. Sparse reference steps carry their
 * complete context; the source sampler's tokens are observations, not a promise
 * that another sampler interprets the same numeric seed identically.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { resolveResidentPartitionAllocation } from '../../src/inference/pipelines/text/resident-partition-contract.js';

const LIMIT_KEYS = ['maxTokens', 'maxPromptTokens', 'maxActivationBytes', 'maxOutputCharacters',
  'maxAttempts', 'maxConcurrentAttempts'];
const DIGEST = /^sha256:[a-f0-9]{64}$/;
export const INDEPENDENT_TOLERANCE = 0.001;

export function physicalInputDigest(bytes) {
  return 'sha256:' + createHash('sha256').update(bytes).digest('hex');
}

function exactKeys(value, keys, label) {
  assert(value && typeof value === 'object' && !Array.isArray(value), `${label} must be an object`);
  assert.deepEqual(Object.keys(value).sort(), [...keys].sort(), `${label} requires exactly its declared fields`);
}

export function resolvePhysicalPartitionConfig(manifest, modelIdentity, config = null, legacy = {}) {
  assert.match(modelIdentity, DIGEST);
  const chat = legacy.chat === true;
  if (config !== null) {
    exactKeys(config, ['schema', 'modelIdentity', 'splitLayer', 'generation', 'limits', 'prompts', 'reference'], 'Physical config');
    assert.equal(config.schema, 'doppler.resident-physical-config/v1');
    assert.equal(config.modelIdentity, modelIdentity, 'Physical config model identity differs from loaded manifest');
    assert(!legacy.chat && legacy.prompts === undefined, 'Explicit config cannot mix legacy prompt/chat overrides');
    assert(Number.isSafeInteger(config.splitLayer), 'Physical config requires an explicit splitLayer');
    exactKeys(config.limits, LIMIT_KEYS, 'Physical limits');
    if (config.reference !== null) {
      exactKeys(config.reference, ['path', 'sha256'], 'Independent reference source');
      assert(typeof config.reference.path === 'string' && config.reference.path.trim());
      assert.match(config.reference.sha256, DIGEST);
    }
  }
  const generation = resolveGenerationOptions(config !== null ? config.generation : (chat ? {
    maxTokens: 1024, maxSeqLen: 4096, temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, repetitionPenaltyWindow: 0, presencePenalty: 0, useChatTemplate: true,
  } : {
    maxTokens: 3, maxSeqLen: 128, temperature: 0, topK: 0, topP: 1,
    repetitionPenalty: 1.1, repetitionPenaltyWindow: 0, presencePenalty: 0.1, useChatTemplate: false,
  }));
  assert(generation.temperature === 0 || generation.seed !== undefined,
    'Stochastic parity requires an explicit sampling seed');
  const plan = createLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
    splitLayer: config !== null ? config.splitLayer : null,
    activationDtype: manifest.inference.session.compute.defaults.activationDtype });
  const planId = hashLayerPartitionPlan(plan);
  const limits = structuredClone(config !== null ? config.limits : { maxTokens: generation.maxTokens, maxPromptTokens: 2048,
    maxActivationBytes: 1024 * 1024, maxOutputCharacters: 1024, maxAttempts: 16, maxConcurrentAttempts: 2 });
  resolveResidentPartitionAllocation(manifest, modelIdentity, {
    model: { id: manifest.modelId, identity: modelIdentity }, plan, planId, index: 0, participantId: 'a', limits, generation,
  });
  assert(limits.maxConcurrentAttempts <= limits.maxAttempts, 'Concurrent attempts exceed attempt allocation');
  const prompts = structuredClone(config !== null ? config.prompts : legacy.prompts ?? (chat ? [
    [{ role: 'user', content: 'Reply with only the word Hello.' }],
    [{ role: 'user', content: 'What is two plus two? Answer briefly.' }],
  ] : ['The color of the sky is', 'The capital of France is']));
  assert(Array.isArray(prompts) && prompts.length > 0, 'Physical prompts must be nonempty');
  for (const prompt of prompts) {
    assert(typeof prompt === 'string' ? prompt.length > 0 : Array.isArray(prompt) && prompt.length > 0
      && prompt.every(message => message && ['system', 'user', 'assistant'].includes(message.role)
        && typeof message.content === 'string' && message.content.length > 0), 'Invalid physical prompt');
  }
  assert(limits.maxAttempts >= prompts.length + 3, 'Attempt allocation must include short, cancelled and denied-output controls');
  return { modelIdentity, plan, planId, generation, limits, prompts, chat,
    reference: config !== null ? config.reference : null };
}

export function bindPhysicalIndependentReference(bytes, source, configuration) {
  assert.equal(physicalInputDigest(bytes), source.sha256, 'Independent reference digest changed');
  const reference = JSON.parse(bytes);
  assert.equal(reference.kind, 'independent-source-model', 'Doppler parity captures are not independent accuracy references');
  assert(typeof reference.provenance?.implementation === 'string' && reference.provenance.implementation.trim(),
    'Independent reference requires its implementation provenance');
  assert.match(reference.provenance.implementationIdentity, DIGEST);
  assert.equal(reference.modelIdentity, configuration.modelIdentity, 'Independent reference model identity changed');
  assert.equal(reference.planId, configuration.planId, 'Independent reference plan changed');
  assert.deepEqual(resolveGenerationOptions(reference.generation), configuration.generation,
    'Independent reference generation changed');
  assert.deepEqual(reference.prompts, configuration.prompts, 'Independent reference prompts changed');
  assert(Array.isArray(reference.expected) && reference.expected.length === configuration.prompts.length,
    'Independent reference must cover every prompt');
  const vocabSize = configuration.plan.vocabSize;
  for (const expected of reference.expected) {
    assert(Array.isArray(expected.input) && expected.input.length > 0 && expected.input.every(token =>
      Number.isSafeInteger(token) && token >= 0 && token < vocabSize), 'Independent reference requires exact input tokens');
    assert(expected.input.length <= configuration.limits.maxPromptTokens
      && expected.input.length + configuration.generation.maxTokens <= configuration.generation.maxSeqLen,
    'Independent reference exceeds the accepted prompt/context allocation');
    assert(Array.isArray(expected.steps) && expected.steps.length > 0
      && expected.steps.length <= configuration.generation.maxTokens, 'Invalid independent reference steps');
    let previousStep = -1;
    for (const [index, step] of expected.steps.entries()) {
      assert(Number.isSafeInteger(step.step) && step.step > previousStep && step.step < configuration.generation.maxTokens,
        'Independent reference steps must be ordered unique generated-token indices');
      previousStep = step.step;
      assert(Array.isArray(step.context) && step.context.length === expected.input.length + step.step
        && step.context.every(token => Number.isSafeInteger(token) && token >= 0 && token < vocabSize),
      'Independent reference requires the exact token context at each selected step');
      assert.deepEqual(step.context.slice(0, expected.input.length), expected.input, 'Independent reference prompt tokens changed');
      assert(Number.isSafeInteger(step.tokenId) && step.tokenId >= 0 && step.tokenId < vocabSize, 'Invalid reference token');
      assert([null, 'eos-token', 'stop-token', 'stop-sequence', 'max-tokens'].includes(step.stopReason), 'Invalid reference stop reason');
      if (step.stopReason !== null) assert.equal(index, expected.steps.length - 1, 'Reference cannot continue after termination');
      if (step.stopReason === 'max-tokens') assert.equal(step.step + 1, configuration.generation.maxTokens);
      assert(typeof step.logits === 'string', 'Reference logits must be base64');
      const data = Buffer.from(step.logits, 'base64');
      assert.equal(data.toString('base64'), step.logits, 'Reference logits must use canonical base64');
      assert.equal(data.length, vocabSize * 4, 'Independent reference must contain full logits');
      const logits = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
      assert(logits.every(Number.isFinite), 'Independent reference logits must be finite');
      step.logits = logits;
    }
    const longest = expected.steps.at(-1).context;
    for (const step of expected.steps) {
      assert.deepEqual(step.context, longest.slice(0, step.context.length),
        'Independent reference contexts must describe one coherent token history');
      if (step.context.length < longest.length) assert.equal(longest[step.context.length], step.tokenId,
        'Independent reference sampled token must match its later source context');
    }
  }
  return reference;
}
