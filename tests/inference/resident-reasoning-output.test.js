import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createGenerationOutputDecoder } from '../../src/inference/generation-output.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

// Only numeric/GPU owners are stubs. Real resident admission, decoder, limits,
// continuation and attempt settlement execute together.
const fixture = globalThis.residentReasoningFixture = { token: 0, released: 0, closed: 0 };
const residentURL = new URL('../../src/inference/pipelines/text/resident-partition.js', import.meta.url).href;
const modules = {
  '../shader-scoped-pipeline.js': 'export const runPipelineOperation = async (owner, operation) => operation(owner);',
  '../../../memory/buffer-pool.js': `export const readBuffer = async () => {
    const scores = new Float32Array(6).fill(-100); scores[globalThis.residentReasoningFixture.token] = 100;
    return scores.buffer;
  }; export const releaseBuffer = () => { globalThis.residentReasoningFixture.released++; };`,
  './partition-execution.js': 'export const assertPartitionExecutionSupported = () => {}; export const executePartitionLayers = async () => ({ logits: {logitsBuffer: {}, vocabSize: 6}, timing: {} });',
  './partition-attempt.js': 'export const createPartitionAttempt = () => ({state:{currentSeqLen:0,decodeStepCount:0},close(){globalThis.residentReasoningFixture.closed++;}});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === residentURL && modules[specifier]) {
    return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { createResidentPartitionSession } = await import(residentURL);
  const contract = { schema: 'doppler.reasoning-output/v1', openingToken: { id: 0, text: '<think>' },
    closingToken: { id: 1, text: '</think>' } };
  const text = ['<think>', '</think>', 'hidden text consumes budget', 'Yes', '<eos>', 'x'];
  const tokenizer = { decode: ids => ids.map(id => text[id]).join(''),
    createRawIncrementalDecoder: () => ({ push: id => text[id], pendingText: () => '', finish: () => '' }),
    createIncrementalDecoder: () => ({ push: id => id < 2 || id === 4 ? '' : text[id], pendingText: () => '', finish: () => '' }) };
  const generation = resolveGenerationOptions({ maxTokens: 5, maxSeqLen: 12, temperature: 0,
    topK: 1, topP: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false });
  const plan = { schema: 'doppler.layer-partition-contract/v1', activationDtype: 'f32', vocabSize: 6, hiddenSize: 2,
    partitions: [{ layerRange: [0, 0] }, { layerRange: [1, 1] }] };
  const planId = computeCanonicalSha256(plan);
  const allocation = { model: { id: 'fixture', identity: 'sha256:model' }, plan, planId, index: 1,
    participantId: 'right', generation,
    limits: { maxAttempts: 8, maxConcurrentAttempts: 3, maxPromptTokens: 1, maxActivationBytes: 8, maxOutputCharacters: 30 } };
  const session = await createResidentPartitionSession({ modelPartition: { plan, index: 1 }, dopplerLoader: {} }, allocation,
    { getTokenContract: () => ({ eosTokenId: 4, reasoningOutput: contract }),
      createIncrementalDecoder: () => createGenerationOutputDecoder(tokenizer, contract, [4]) }, async () => {});
  const attempt = (id, maxTokens = 5) => ({ identity: { modelId: 'fixture', modelIdentity: 'sha256:model', planId,
    threadId: id, attemptId: id, participantA: 'left', participantB: 'right' }, step: 0, tokenPosition: 0,
    continuation: null, inputTokenIds: [5], maxTokens, generation: { ...generation, maxTokens } });
  async function advance(value, token) {
    fixture.token = token;
    const result = await session.executeGroup1({ ...value, inputTokenCount: 1, signal: new AbortController().signal,
      activation: { shape: [1, 1, 2], dtype: 'f32', seqOffset: value.tokenPosition, step: value.step,
        tensorData: new Float32Array(2) } });
    value.step++; value.tokenPosition++; value.continuation = result.continuation; value.inputTokenIds = [result.tokenId];
    return result;
  }
  const limited = attempt('limit');
  assert.equal((await advance(limited, 0)).delta, '');
  await assert.rejects(advance(limited, 2), error => error.code === 'DOPPLER_RESIDENT_OUTPUT_LIMIT');
  await assert.rejects(advance(limited, 1), /retired/);
  await session.closeAttempt({ identity: limited.identity });

  const truncated = attempt('truncated', 2);
  await advance(truncated, 0);
  const truncatedResult = await advance(truncated, 5);
  assert.equal(truncatedResult.stopReason, 'max-tokens'); assert.equal(truncatedResult.delta, '');
  assert.equal(truncatedResult.metrics.outputUsage.state, 'truncated');
  await session.closeAttempt({ identity: truncated.identity });

  const cancelled = attempt('cancelled'); await advance(cancelled, 0);
  await session.closeAttempt({ identity: cancelled.identity });
  await assert.rejects(advance(cancelled, 1), /retired/);
  const malformed = attempt('malformed'); await advance(malformed, 0);
  await assert.rejects(advance(malformed, 4), error => error.code === 'DOPPLER_GENERATION_OUTPUT_INVALID');
  await session.closeAttempt({ identity: malformed.identity });
  const valid = attempt('valid');
  for (const token of [0, 5, 1, 3]) await advance(valid, token);
  const final = await advance(valid, 4);
  assert.equal(final.stopReason, 'eos-token'); assert.equal(final.metrics.outputUsage.rawTokens, 5);
  assert.equal(final.metrics.outputUsage.rawCharacters, '<think>x</think>Yes<eos>'.length);
  await session.close();
  assert.equal(fixture.closed, 5);
} finally { hooks.deregister(); delete globalThis.residentReasoningFixture; }
console.log('resident-reasoning-output.test: raw limits, truncation, cancellation, failure and reuse passed (GPU stubbed)');
