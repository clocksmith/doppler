import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { PipelineGenerator } = await import('../../src/inference/pipelines/text/generator.js');
const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const { compileExecutionPlanState } = await import('../../src/inference/pipelines/text/execution-plan.js');

// =============================================================================
// Behavioral parity test: generateTokenIds() vs generate()
//
// Constructs a PipelineGenerator with a minimal stub state, overrides the
// GPU-coupled methods (_prefillPromptToLogits, _decodeNextTokenViaLogits) to
// return predictable logits, then verifies both generation surfaces produce
// identical behavior for: EOS, abort, stop sequences, stats, and cleanup.
// =============================================================================

const EOS_TOKEN = 2;
const VOCAB_SIZE = 32;

function makeLogitsForToken(tokenId) {
  const logits = new Float32Array(VOCAB_SIZE);
  logits[tokenId] = 10.0;
  return logits;
}

function createFakeTokenizer() {
  return {
    encode(text) {
      return [1, 5, 7];
    },
    decode(ids, skipSpecial, skipFallback) {
      return ids.map((id) => `[${id}]`).join('');
    },
    getSpecialTokens() {
      return { eos: EOS_TOKEN, pad: 0 };
    },
    getVocabSize() {
      return VOCAB_SIZE;
    },
  };
}

function createMinimalState(overrides = {}) {
  const runtimeConfig = createDopplerConfig({
    runtime: {
      inference: {
        batching: {},
        compute: {
          activationDtype: 'f32',
        },
        generation: {
          maxTokens: overrides.maxTokens ?? 10,
          disableMultiTokenDecode: true,
        },
        session: {
          decodeLoop: {
            batchSize: 1,
            stopCheckMode: 'batch',
            readbackInterval: 1,
            readbackMode: 'sequential',
            ringTokens: 1,
            ringStop: 1,
            ringStaging: 1,
            disableCommandBatching: true,
          },
        },
        sampling: {
          temperature: 0,
          topK: 1,
          topP: 1,
          repetitionPenalty: 1.0,
        },
        chatTemplate: { enabled: false },
      },
    },
  }).runtime;

  const executionPlanState = compileExecutionPlanState({
    runtimeConfig: { inference: runtimeConfig.inference, shared: runtimeConfig.shared },
    resolvedKernelPath: null,
    kernelPathSource: 'none',
  });

  return {
    tokenizer: createFakeTokenizer(),
    kvCache: null,
    linearAttentionRuntime: { schemaVersion: 1, layers: new Map() },
    convLayerStates: new Map(),
    moeRouter: null,
    speculativeDecoder: null,
    decodeBuffers: null,
    decodeRing: null,
    finitenessBuffer: null,
    emulation: null,
    debugFlags: {},
    decodeStepCount: 0,
    resolvedKernelPath: null,
    kernelPathSource: 'none',
    executionPlanState,
    disableRecordedLogits: false,
    disableFusedDecode: false,
    manifest: null,
    modelConfig: {
      vocabSize: VOCAB_SIZE,
      hiddenSize: 64,
      numHeads: 4,
      headDim: 16,
      numLayers: 2,
      stopTokenIds: overrides.stopTokenIds ?? [EOS_TOKEN],
      chatTemplateType: null,
      layerTypes: null,
    },
    weights: new Map(),
    expertWeights: new Map(),
    isLoaded: true,
    isGenerating: false,
    currentSeqLen: 0,
    currentTokenIds: null,
    runtimeConfig,
    dopplerLoader: null,
    gpuContext: null,
    useGPU: false,
    memoryContext: null,
    storageContext: null,
    stats: {
      prefillTimeMs: 0,
      decodeTimeMs: 0,
      ttftMs: 0,
      prefillTokens: 0,
      decodeTokens: 0,
      memoryUsageBytes: 0,
      tokensGenerated: 0,
      totalTimeMs: 0,
      decodeRecordMs: 0,
      decodeSubmitWaitMs: 0,
      decodeReadbackWaitMs: 0,
      decodeProfileSteps: [],
      attentionInputs: [],
    },
    batchingStats: {
      batchedForwardCalls: 0,
      unbatchedForwardCalls: 0,
      totalBatchedTimeMs: 0,
      totalUnbatchedTimeMs: 0,
      gpuSubmissions: 0,
    },
    baseUrl: null,
    ropeFreqsCos: null,
    ropeFreqsSin: null,
    ropeLocalCos: null,
    ropeLocalSin: null,
    debug: false,
    layerPipelinePlan: null,
    useTiedEmbeddings: false,
    embeddingVocabSize: null,
    embeddingTranspose: false,
    layerRouterWeights: null,
    lora: null,
  };
}

function stubGenerator(gen, tokenSequence) {
  let decodeCallIndex = 0;

  gen._prefillPromptToLogits = async function (_prompt, _opts, _label) {
    const inputIds = [1, 5, 7];
    const logits = makeLogitsForToken(tokenSequence[0]);
    return { inputIds, logits };
  };

  gen._decodeNextTokenViaLogits = async function (_currentIds, _opts) {
    decodeCallIndex++;
    const nextTokenIdx = decodeCallIndex;
    if (nextTokenIdx < tokenSequence.length) {
      return tokenSequence[nextTokenIdx];
    }
    return EOS_TOKEN;
  };
}

// === Test 1: EOS stop behavior — both paths stop at EOS ===
{
  const tokenSeq = [10, 11, EOS_TOKEN, 99];
  const state1 = createMinimalState({ maxTokens: 20 });
  const gen1 = new PipelineGenerator(state1);
  stubGenerator(gen1, tokenSeq);
  const result1 = await gen1.generateTokenIds('test', { useChatTemplate: false });
  assert.deepStrictEqual(
    result1.tokenIds,
    [10, 11, EOS_TOKEN],
    'generateTokenIds must stop at EOS token'
  );

  const state2 = createMinimalState({ maxTokens: 20 });
  const gen2 = new PipelineGenerator(state2);
  stubGenerator(gen2, tokenSeq);
  const genTokens = [];
  for await (const chunk of gen2.generate('test', { useChatTemplate: false })) {
    void chunk;
  }
  // generate() tracks stats.tokensGenerated via _runDecodeLoop
  assert.equal(
    state2.stats.tokensGenerated,
    3,
    'generate() must also produce 3 tokens before EOS stop'
  );
  assert.deepStrictEqual(
    result1.tokenIds.length,
    state2.stats.tokensGenerated,
    'Both paths must generate the same count'
  );
}
console.log('  ok: EOS stop parity');

// === Test 1b: immediate EOS on first sampled token must not decode one extra token ===
{
  const immediateStopSeq = [EOS_TOKEN, 19, 20];
  const state1 = createMinimalState({ maxTokens: 20 });
  const gen1 = new PipelineGenerator(state1);
  stubGenerator(gen1, immediateStopSeq);
  const result1 = await gen1.generateTokenIds('test', { useChatTemplate: false });
  assert.deepStrictEqual(
    result1.tokenIds,
    [EOS_TOKEN],
    'generateTokenIds must stop immediately when the first sampled token is EOS'
  );

  const state2 = createMinimalState({ maxTokens: 20 });
  const gen2 = new PipelineGenerator(state2);
  let yieldedCount = 0;
  stubGenerator(gen2, immediateStopSeq);
  for await (const _ of gen2.generate('test', { useChatTemplate: false })) {
    yieldedCount++;
  }
  assert.equal(yieldedCount, 1, 'generate() must emit only the immediate EOS token');
  assert.equal(state2.stats.tokensGenerated, 1, 'generate() must not decode past an immediate EOS token');
}
console.log('  ok: immediate EOS first-token parity');

// === Test 2: maxTokens cap — both paths respect the limit ===
{
  const neverEnds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
  const state1 = createMinimalState({ maxTokens: 5, stopTokenIds: [] });
  const gen1 = new PipelineGenerator(state1);
  stubGenerator(gen1, neverEnds);
  const result1 = await gen1.generateTokenIds('test', { useChatTemplate: false });
  assert.equal(result1.tokenIds.length, 5, 'generateTokenIds must respect maxTokens');

  const state2 = createMinimalState({ maxTokens: 5, stopTokenIds: [] });
  const gen2 = new PipelineGenerator(state2);
  stubGenerator(gen2, neverEnds);
  for await (const _ of gen2.generate('test', { useChatTemplate: false })) { void _; }
  assert.equal(state2.stats.tokensGenerated, 5, 'generate() must also respect maxTokens');
}
console.log('  ok: maxTokens cap parity');

// === Test 3: Abort signal — both paths stop when aborted ===
{
  const neverEnds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
  const controller = new AbortController();

  const state1 = createMinimalState({ maxTokens: 20, stopTokenIds: [] });
  const gen1 = new PipelineGenerator(state1);
  let callCount1 = 0;
  const origDecode1 = gen1._decodeNextTokenViaLogits;
  gen1._prefillPromptToLogits = async () => ({ inputIds: [1, 5, 7], logits: makeLogitsForToken(10) });
  gen1._decodeNextTokenViaLogits = async function (...args) {
    callCount1++;
    if (callCount1 >= 3) controller.abort();
    return neverEnds[callCount1] ?? 19;
  };
  const result1 = await gen1.generateTokenIds('test', { useChatTemplate: false, signal: controller.signal });
  // First token from prefill + some decode tokens before abort kicks in
  assert.ok(result1.tokenIds.length < 20, `generateTokenIds must stop on abort (got ${result1.tokenIds.length})`);
  const abortedLen1 = result1.tokenIds.length;

  const controller2 = new AbortController();
  const state2 = createMinimalState({ maxTokens: 20, stopTokenIds: [] });
  const gen2 = new PipelineGenerator(state2);
  let callCount2 = 0;
  gen2._prefillPromptToLogits = async () => ({ inputIds: [1, 5, 7], logits: makeLogitsForToken(10) });
  gen2._decodeNextTokenViaLogits = async function (...args) {
    callCount2++;
    if (callCount2 >= 3) controller2.abort();
    return neverEnds[callCount2] ?? 19;
  };
  for await (const _ of gen2.generate('test', { useChatTemplate: false, signal: controller2.signal })) { void _; }
  const abortedLen2 = state2.stats.tokensGenerated;

  assert.ok(abortedLen2 < 20, `generate() must stop on abort (got ${abortedLen2})`);
  // Both should stop at approximately the same point (within 1 token due to yield timing)
  assert.ok(
    Math.abs(abortedLen1 - abortedLen2) <= 1,
    `Abort parity: generateTokenIds=${abortedLen1}, generate=${abortedLen2}`
  );
}
console.log('  ok: abort signal parity');

// === Test 4: Stop sequences — both paths respect string stop ===
{
  // Tokens 10, 11, 12 — tokenizer.decode will produce "[10][11][12]"
  // Stop sequence: "[12]"
  const tokenSeq = [10, 11, 12, 13, 14];
  const state1 = createMinimalState({ maxTokens: 20, stopTokenIds: [] });
  const gen1 = new PipelineGenerator(state1);
  stubGenerator(gen1, tokenSeq);
  const result1 = await gen1.generateTokenIds('test', {
    useChatTemplate: false,
    stopSequences: ['[12]'],
  });
  assert.ok(
    result1.tokenIds.includes(12),
    'generateTokenIds must include the stop-sequence trigger token'
  );
  assert.ok(
    result1.tokenIds.length <= 3,
    `generateTokenIds must stop at stop sequence (got ${result1.tokenIds.length})`
  );

  const state2 = createMinimalState({ maxTokens: 20, stopTokenIds: [] });
  const gen2 = new PipelineGenerator(state2);
  stubGenerator(gen2, tokenSeq);
  for await (const _ of gen2.generate('test', { useChatTemplate: false, stopSequences: ['[12]'] })) { void _; }
  assert.ok(
    state2.stats.tokensGenerated <= 3,
    `generate() must also stop at stop sequence (got ${state2.stats.tokensGenerated})`
  );
}
console.log('  ok: stop sequence parity');

// === Test 5: Stats fields set correctly ===
{
  const tokenSeq = [10, 11, EOS_TOKEN];
  const state1 = createMinimalState({ maxTokens: 10 });
  const gen1 = new PipelineGenerator(state1);
  stubGenerator(gen1, tokenSeq);
  const result1 = await gen1.generateTokenIds('test', { useChatTemplate: false });

  assert.ok(Number.isFinite(result1.stats.prefillTimeMs), 'prefillTimeMs must be set');
  assert.ok(Number.isFinite(result1.stats.decodeTimeMs), 'decodeTimeMs must be set');
  assert.ok(Number.isFinite(result1.stats.totalTimeMs), 'totalTimeMs must be set');
  assert.ok(Number.isFinite(result1.stats.ttftMs), 'ttftMs must be set');
  assert.equal(result1.stats.tokensGenerated, 3, 'tokensGenerated must match');
  assert.equal(result1.stats.decodeTokens, 3, 'decodeTokens must match');
  assert.equal(result1.stats.prefillTokens, 3, 'prefillTokens must match input length');
}
console.log('  ok: stats fields');

// === Test 5b: self-speculation honors configured burst tokens ===
{
  const tokenSeq = [10, 11, 12, 13, 14, 15];
  const state = createMinimalState({ maxTokens: 5, stopTokenIds: [] });
  const gen = new PipelineGenerator(state);
  let decodeCallCount = 0;
  gen._prefillPromptToLogits = async function () {
    return { inputIds: [1, 5, 7], logits: makeLogitsForToken(tokenSeq[0]) };
  };
  gen._decodeNextTokenViaLogits = async function () {
    decodeCallCount += 1;
    return tokenSeq[decodeCallCount] ?? EOS_TOKEN;
  };

  const result = await gen.generateTokenIds('test', {
    useChatTemplate: false,
    speculation: {
      mode: 'self',
      tokens: 3,
      verify: 'greedy',
      threshold: null,
      rollbackOnReject: true,
    },
  });

  assert.deepStrictEqual(
    result.tokenIds,
    [10, 11, 12, 13, 14],
    'self-speculation must emit one base token plus the configured speculative burst'
  );
  assert.equal(decodeCallCount, 4, 'self-speculation must decode one base token plus three speculative tokens');
  assert.equal(result.stats.speculationAttempts, 3, 'self-speculation must record one attempt per speculative token');
  assert.equal(result.stats.speculationAccepted, 3, 'self-speculation must record one accept per speculative token');
}
console.log('  ok: self-speculation burst honors configured tokens');

// === Test 6: Cleanup — isGenerating is false after success and after error ===
{
  const state1 = createMinimalState({ maxTokens: 5 });
  const gen1 = new PipelineGenerator(state1);
  stubGenerator(gen1, [10, EOS_TOKEN]);
  await gen1.generateTokenIds('test', { useChatTemplate: false });
  assert.equal(state1.isGenerating, false, 'isGenerating must be false after success');

  const state2 = createMinimalState({ maxTokens: 5 });
  const gen2 = new PipelineGenerator(state2);
  gen2._prefillPromptToLogits = async () => { throw new Error('deliberate prefill failure'); };
  try {
    await gen2.generateTokenIds('test', { useChatTemplate: false });
    assert.fail('should have thrown');
  } catch (e) {
    assert.equal(e.message, 'deliberate prefill failure');
  }
  assert.equal(state2.isGenerating, false, 'isGenerating must be false after error');
}
console.log('  ok: cleanup on success and throw');

// === Test 7: Double-generation guard ===
{
  const state1 = createMinimalState({ maxTokens: 5 });
  state1.isGenerating = true;
  const gen1 = new PipelineGenerator(state1);
  await assert.rejects(
    () => gen1.generateTokenIds('test', { useChatTemplate: false }),
    /Generation already in progress/,
    'generateTokenIds must reject when already generating'
  );
}
console.log('  ok: double-generation guard');

console.log('generate-token-ids-behavioral-parity.test: ok');
