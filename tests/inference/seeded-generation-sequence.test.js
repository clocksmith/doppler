import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createSessionController } from '../../src/client/runtime/session-controller.js';
import { sample, samplingSeedForToken } from '../../src/inference/token-sampling.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

const generation = Object.freeze({ maxTokens: 12, maxSeqLen: 32, temperature: 1,
  topK: 4, topP: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0,
  useChatTemplate: false, seed: 17 });
const expected = [0, 0, 3, 1, 2, 1, 3, 0, 1, 2, 3, 0];
const logits = () => new Float32Array(4);
const sampleAt = (seed, index) => sample(logits(), { ...generation, seed: samplingSeedForToken(seed, index) });
for (const seed of [0, 17, 0.25, 4294967295, Number.MAX_SAFE_INTEGER, Number.MAX_VALUE]) {
  assert.equal(samplingSeedForToken(seed, 0), seed);
  assert.equal(sampleAt(seed, 0), sample(logits(), { ...generation, seed }));
  assert.equal(samplingSeedForToken(seed, 1), ((seed % 4294967296) + 1) % 4294967296);
  assert.notEqual(samplingSeedForToken(seed, 1), samplingSeedForToken(seed, 2));
  assert.equal(sampleAt(seed, 3), sampleAt(seed, 3), 'retrying an output index reuses its draw');
}
assert.equal(samplingSeedForToken(4294967295, 1), 0);
assert.equal(samplingSeedForToken(undefined, 3), undefined);
for (const seed of [null, -1, NaN, Infinity]) {
  assert.throws(() => resolveGenerationOptions({ ...generation, seed }));
  assert.throws(() => samplingSeedForToken(seed, 0));
}
for (const index of [-1, 0.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
  assert.throws(() => samplingSeedForToken(17, index));
}

function controller(gpu = false, forcedFirst = false) {
  const seeds = [];
  let released = 0;
  const value = createSessionController({ async executePhase(phase, commands, request) {
    const options = request.context.generationOptions;
    seeds.push(options.seed);
    const scores = forcedFirst && phase === 'prefill' ? Float32Array.from([100, -100, -100, -100]) : logits();
    return { results: [gpu ? { tokenId: sample(scores, options), vocabSize: 4 } : { logits: scores }] };
  } }, { bindSlots() {}, writeSlot() {}, assertDeviceAvailable() {}, releaseTransient() { released++; } },
  { reset() {}, getTokenContract: () => ({}), releaseStepResult() {} });
  const plan = { phases: { prefill: [], decode: [] }, memoryLayout: {}, ...(gpu ? { tokenSelection: {} } : {}) };
  return { seeds, value, plan, released: () => released,
    stream: options => value.generateTokens(plan, { promptTokens: [1, 2], ...generation, ...options }) };
}
for (const gpu of [false, true]) {
  const fixture = controller(gpu);
  assert.deepEqual(await Array.fromAsync(fixture.stream()), expected);
  assert.deepEqual(fixture.seeds, Array.from({ length: 12 }, (_, index) => 17 + index));
  assert.deepEqual(await Array.fromAsync(fixture.stream()), expected, 'fresh generation restarts its own stream');
  assert.deepEqual(await Array.fromAsync(controller(gpu, true).stream()), [0, ...expected.slice(1)],
    'forced output still advances the generated-token index');
  assert.deepEqual(await Array.fromAsync(fixture.stream({ temperature: 0 })), Array(12).fill(0));
  const aborted = new AbortController();
  const active = fixture.stream({ signal: aborted.signal });
  assert.equal((await active.next()).value, expected[0]);
  aborted.abort();
  await assert.rejects(active.next(), /aborted/);
  assert.deepEqual(await Array.fromAsync(fixture.stream()), expected);
  assert.equal(fixture.released(), 5);
}
const left = controller(), right = controller();
const streams = [left.stream(), right.stream()];
for (const token of expected) {
  assert.equal((await streams[1].next()).value, token);
  assert.equal((await streams[0].next()).value, token);
}
for (const stream of streams) await stream.return();

// Stub only GPU execution/storage. Real resident admission, continuation,
// sampling, attempt isolation, cancellation and settlement remain exercised.
const residentURL = new URL('../../src/inference/pipelines/text/resident-partition.js', import.meta.url).href;
const decodeURL = new URL('../../src/inference/pipelines/text/generator/decode-runtime.js', import.meta.url).href;
const modules = {
  '../shader-scoped-pipeline.js': 'export const runPipelineOperation = async (owner, operation) => operation(owner);',
  '../../../memory/buffer-pool.js': 'export const readBuffer = async () => new Float32Array(4).buffer; export const releaseBuffer = () => {};',
  './partition-execution.js': 'export const assertPartitionExecutionSupported = () => {}; export const executePartitionLayers = async () => ({ logits: {logitsBuffer: {}, vocabSize: 4}, timing: {} });',
  './partition-attempt.js': 'export const createPartitionAttempt = () => ({state:{currentSeqLen:0,decodeStepCount:0},close(){}});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === decodeURL && specifier === '../generator-steps.js') {
    const original = new URL('../../src/inference/pipelines/text/generator-steps.js', import.meta.url).href;
    return { url: `data:text/javascript,${encodeURIComponent(`
      export * from ${JSON.stringify(original)};
      export const decodeStep = async (state, ids, options) => {
        state.seeds.push(options.seed);
        return options.seed;
      };
    `)}`, shortCircuit: true };
  }
  if (context.parentURL === residentURL && modules[specifier]) {
    return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { PipelineGenerator } = await import('../../src/inference/pipelines/text/generator.js');
  const pipeline = Object.create(PipelineGenerator.prototype);
  pipeline._state = { modelConfig: {}, seeds: [] };
  pipeline._getDecodeHelpers = () => ({});
  const ids = [1, 2], options = { ...generation, promptTokenCount: ids.length };
  for (const token of expected) {
    const count = ids.length - options.promptTokenCount;
    assert.equal(pipeline._sampleNextTokenFromLogits(logits(), ids, options), token);
    assert.equal(await pipeline._decodeStep(ids, options), 17 + count,
      'direct pipeline CPU and fused GPU decode receive the same indexed seed');
    ids.push(token);
  }
  assert.equal(options.seed, 17);
  assert.deepEqual(pipeline._state.seeds, Array.from({ length: 12 }, (_, index) => 17 + index));
  for (const index of [0, 7, 11]) {
    assert.equal(pipeline._sampleNextTokenFromLogits(logits(), [1, 2, ...expected.slice(0, index)],
      { ...options, temperature: 0 }), 0);
  }
  const { createResidentPartitionSession } = await import(residentURL);
  const plan = { schema: 'doppler.layer-partition-contract/v1', activationDtype: 'f32',
    vocabSize: 4, hiddenSize: 2, partitions: [{ layerRange: [0, 0] }, { layerRange: [1, 1] }] };
  const planId = computeCanonicalSha256(plan);
  const allocation = { model: { id: 'fixture', identity: 'sha256:model' }, plan, planId, index: 1,
    participantId: 'right', generation: resolveGenerationOptions(generation),
    limits: { maxAttempts: 5, maxConcurrentAttempts: 3, maxPromptTokens: 2, maxActivationBytes: 16, maxOutputCharacters: 100 } };
  const session = await createResidentPartitionSession({ modelPartition: { plan, index: 1 }, dopplerLoader: {} },
    allocation, { getTokenContract: () => ({}), createIncrementalDecoder: () => ({ push: String, pendingText: () => '', finish: () => '' }) }, async () => {});
  const originalDigest = session.getDescriptor().generationDigest;
  const attempt = id => ({ identity: { modelId: 'fixture', modelIdentity: 'sha256:model', planId,
    threadId: id, attemptId: id, participantA: 'left', participantB: 'right' }, continuation: null,
    step: 0, tokenPosition: 0, ids: [1, 2], output: [] });
  const a = attempt('a'), b = attempt('b');
  async function advance(value, signal = new AbortController().signal) {
    const count = value.ids.length;
    const result = await session.executeGroup1({ ...value, inputTokenIds: value.ids, inputTokenCount: count,
      maxTokens: 12, generation: allocation.generation, signal,
      activation: { shape: [1, count, 2], dtype: 'f32', seqOffset: value.tokenPosition, step: value.step,
        tensorData: new Float32Array(count * 2) } });
    value.output.push(result.tokenId); value.ids = [result.tokenId];
    value.continuation = result.continuation; value.step++; value.tokenPosition += count;
    return result;
  }
  for (let index = 0; index < expected.length; index++) {
    await advance(b); await advance(a);
    assert.equal(a.output[index], expected[index]); assert.equal(b.output[index], expected[index]);
  }
  assert.equal(session.getDescriptor().generationDigest, originalDigest);
  assert.equal(allocation.generation.seed, 17);
  await session.closeAttempt({ identity: a.identity }); await session.closeAttempt({ identity: b.identity });
  const cancelled = attempt('cancelled');
  await advance(cancelled); await session.closeAttempt({ identity: cancelled.identity });
  await assert.rejects(advance(cancelled), /retired/);
  const restarted = attempt('restart');
  for (const token of expected) assert.equal((await advance(restarted)).tokenId, token);
  await session.close();
} finally { hooks.deregister(); }
assert.equal(generation.seed, 17);
console.log('seeded-generation-sequence: advancing isolated CPU/GPU-boundary/resident draws, retries, limits and greedy controls passed');
