import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from 'file:///home/x/deco/d4da/sites/d4da-com/doppler/src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { PipelineGenerator } = await import('file:///home/x/deco/d4da/sites/d4da-com/doppler/src/inference/pipelines/text/generator.js');
const { createDopplerConfig } = await import('file:///home/x/deco/d4da/sites/d4da-com/doppler/src/config/schema/index.js');
const { compileExecutionPlanState } = await import('file:///home/x/deco/d4da/sites/d4da-com/doppler/src/inference/pipelines/text/execution-plan.js');

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

// Streaming observes the first token, every decode token, and immediate EOS.
for (const sequence of [[9, 10, EOS_TOKEN], [EOS_TOKEN]]) {
  const state = createMinimalState();
  const gen = new PipelineGenerator(state);
  stubGenerator(gen, sequence);
  const observed = [];
  const result = await gen.generateTokenIds('stream', {
    useChatTemplate: false,
    onToken: (id, text) => observed.push({ id, text }),
  });
  assert.deepEqual(observed, result.tokenIds.map((id) => ({ id, text: '' })));
  assert.deepEqual(result.tokenIds, sequence);
}

// A first-token observer failure still releases generation ownership.
{
  const state = createMinimalState();
  const gen = new PipelineGenerator(state);
  stubGenerator(gen, [9, EOS_TOKEN]);
  await assert.rejects(gen.generateTokenIds('stream', {
    useChatTemplate: false,
    onToken() { throw new Error('observer failed'); },
  }), /observer failed/);
  assert.equal(state.isGenerating, false);
  assert.deepEqual((await gen.generateTokenIds('retry', { useChatTemplate: false })).tokenIds, [9, EOS_TOKEN]);
}


console.log('Hosted first-token, EOS, callback error cleanup, and retry: passed');
