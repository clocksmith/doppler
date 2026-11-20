import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { log } from '../../src/debug/index.js';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator/generator.js';

const profiler = { enabled: true };
const policy = { enabled: true, routeChunkTokens: 8 };
const cache = { sourceReads: 9, sourceReadBytes: 1000, sourceReadWaitMs: 30, uploadBytes: 900,
  cacheHits: 4, cacheMisses: 3, initialQueueIdleForWeightsMs: 2,
  phaseProfiles: { prefill: { preparations: 4, prepareWallMs: 30 } } };
const readback = { calls: 8, wallMs: 25 };
let snapshots = 0, cacheReady = true;
const state = { stats: {}, currentSeqLen: 0,
  runtimeConfig: { shared: { debug: { profiler } }, inference: { session: { prefillTokenChunkSize: 4 } } },
  dopplerLoader: { getRoutedExpertPolicy: () => policy,
    getRoutedExpertCacheStats() { snapshots++; return cacheReady ? cache : null; },
    getRoutedReadbackStats: () => ({ phases: { prefill: readback } }) } };
const generator = new PipelineGenerator(state);
const inputIds = [1, 2, 3, 4, 5, 6, 7, 8, 9];
generator._resolvePromptOrInputIds = () => inputIds;
const logits = Float32Array.of(0.5, 0.25);
const chunks = [];
generator._prefill = async (ids, opts) => {
  cacheReady = true;
  chunks.push(ids.length);
  cache.sourceReads++;
  cache.sourceReadBytes += 100;
  cache.sourceReadWaitMs += 12;
  cache.uploadBytes += 60;
  cache.cacheHits += 2;
  cache.cacheMisses++;
  cache.initialQueueIdleForWeightsMs++;
  cache.phaseProfiles.prefill.preparations++;
  cache.phaseProfiles.prefill.prepareWallMs += 25;
  readback.calls++;
  readback.wallMs += 9;
  return opts._returnHidden ? { numTokens: ids.length } : logits;
};
generator._commitPrefillHiddenChunk = async () => {};
const receipts = [], info = log.info;
log.info = (category, message, data) => { if (category === 'Perf' && message === 'Routed prefill') receipts.push(data); };
try {
  for (let i = 0; i < 2; i++) {
    const result = await generator._prefillPromptToLogits('private prompt', {}, 'generate');
    assert.equal(result.logits, logits);
    assert.deepEqual(result.inputIds, inputIds);
    const { inputMs, prefillMs, ...observation } = receipts[i];
    assert.equal(inputMs, result.phase.inputMs);
    assert.equal(prefillMs, result.phase.prefillMs);
    assert.deepEqual(observation, { promptTokens: 9, computedTokens: 9, reusedTokens: 0,
      tokenChunkLimit: 4, routeChunkLimit: 8, preparations: 3, prepareWallMs: 75,
      routeReadbacks: 3, routeReadbackWallMs: 27, sourceReads: 3, sourceReadBytes: 300, sourceReadWaitMs: 36,
      uploadBytes: 180, cacheHits: 6, cacheMisses: 3, initialQueueIdleForWeightsMs: 3 });
  }
  assert.deepEqual(chunks, [4, 4, 1, 4, 4, 1]);
  assert.equal(snapshots, 4);
  profiler.enabled = false;
  await generator._prefillPromptToLogits('private prompt', {}, 'generate');
  profiler.enabled = true;
  policy.enabled = false;
  await generator._prefillPromptToLogits('private prompt', {}, 'generate');
  assert.equal(snapshots, 4);
  assert.equal(receipts.length, 2);
  policy.enabled = true;
  cacheReady = false;
  for (const key of Object.keys(cache)) {
    if (typeof cache[key] === 'number') cache[key] = 0;
  }
  cache.phaseProfiles = { prefill: { preparations: 0, prepareWallMs: 0 } };
  await generator._prefillPromptToLogits('private prompt', {}, 'generate');
  assert.equal(receipts[2].sourceReadBytes, 300, 'The first request sees the lazily created bank');
  // Exercise the normal streaming generation entry point with deterministic CPU logits.
  const runtimeConfig = createDopplerConfig().runtime;
  runtimeConfig.shared.debug.profiler = profiler;
  runtimeConfig.inference.session.prefillTokenChunkSize = 4;
  Object.assign(state, { isLoaded: true, debug: false, runtimeConfig, batchingStats: {}, weights: new Map(),
    tokenizer: { decode: () => 'yes', getSpecialTokens: () => ({ eos: 15 }),
      createIncrementalDecoder: () => ({ push: () => 'yes', finish: () => '' }) },
    modelConfig: { vocabSize: 16, chatTemplateEnabled: false, stopTokenIds: [] }, manifest: { modelType: 'transformer' },
    executionPlanState: { activePlanId: 'primary', primaryPlan: { id: 'primary', activationDtype: 'f32',
      defaultMaxTokens: 1, defaultBatchSize: 1, readbackInterval: 1, defaultStopCheckMode: 'batch',
      defaultDisableCommandBatching: false, defaultDisableMultiTokenDecode: false,
      ringTokens: 0, ringStop: false, ringStaging: false }, fallbackPlan: null } });
  const expected = [];
  for await (const token of generator.generate('private prompt', { maxTokens: 1, temperature: 0, topK: 1, topP: 1 })) expected.push(token);
  assert.equal(expected.join(''), 'yes');
  assert.equal(receipts.at(-1).computedTokens, 9);
  cache.phaseProfiles = {};
  generator._prefill = async () => logits;
  await generator._prefillPromptToLogits('private prompt', {}, 'generate');
  assert.equal(receipts[4].prepareWallMs, null, 'A cache without preparation profiling has no timing sample');
  assert.equal(receipts[4].preparations, null);
  generator._prefill = async () => { throw Error('cancelled prefill'); };
  await assert.rejects(generator._prefillPromptToLogits('private prompt', {}, 'generate'), /cancelled prefill/);
  assert.equal(receipts.length, 5);
} finally {
  log.info = info;
}
console.log('Routed request prefill observation: scoped counters, repeated requests, token chunks, disabled and failed requests passed.');
