import assert from 'node:assert/strict';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import {
  applyEntropyBoundStep,
  applyEntropyBoundStatsStep,
  createSeededRandom,
  denoiseCanvas,
  denoiseCanvasWithStatsProvider,
  resolveDenoisingTemperature,
} from '../../src/inference/pipelines/diffusion-gemma/index.js';

const diffusionGemma = {
  canvasLength: 3,
  maxDenoisingSteps: 2,
  maxNewTokens: 3,
  tMin: 0.4,
  tMax: 0.8,
  entropyBound: 100,
  confidenceThreshold: 0.01,
  stabilityThreshold: 1,
  padTokenId: 0,
  eosTokenIds: [7],
  boiTokenId: null,
  eoiTokenId: null,
  imageTokenId: null,
  selfConditioning: true,
  decoderCacheMode: 'encoder_kv_readonly_canvas_concat',
  router: {
    scaleHiddenStates: true,
    normalizeTopK: true,
    perExpertScale: true,
  },
  vocabSize: 8,
};

const config = {
  ...diffusionGemma,
};

function buildLogits(tokenIds, vocabSize) {
  const logits = new Float32Array(tokenIds.length * vocabSize);
  logits.fill(-12);
  for (let i = 0; i < tokenIds.length; i += 1) {
    logits[(i * vocabSize) + tokenIds[i]] = 12;
  }
  return logits;
}

assert.equal(resolveDenoisingTemperature(config, 2), 0.8);
assert.equal(resolveDenoisingTemperature(config, 1), 0.6000000000000001);

{
  const step = applyEntropyBoundStep(
    Int32Array.from([1, 1, 1]),
    buildLogits([2, 3, 4], config.vocabSize),
    config,
    {
      temperature: 0.8,
      random: createSeededRandom(123),
    }
  );
  assert.deepEqual([...step.argmaxCanvas], [2, 3, 4]);
  assert.deepEqual([...step.canvas], [2, 3, 4]);
  assert.equal(step.processedLogits[2], 15);
  assert.equal(step.acceptedCount, 3);
  assert.ok(step.meanEntropy < 0.01);
}

{
  const step = applyEntropyBoundStatsStep(
    Int32Array.from([1, 1, 1]),
    {
      argmaxCanvas: Int32Array.from([2, 3, 4]),
      entropies: new Float32Array([0.001, 0.002, 0.003]),
    },
    config,
    {
      temperature: 0.8,
      random: createSeededRandom(123),
    }
  );
  assert.deepEqual([...step.argmaxCanvas], [2, 3, 4]);
  assert.deepEqual([...step.canvas], [2, 3, 4]);
  assert.equal(step.processedLogits, null);
  assert.equal(step.acceptedCount, 3);
  assert.ok(step.meanEntropy < 0.01);
}

{
  const result = await denoiseCanvasWithStatsProvider(config, {
    initialCanvas: [1, 1, 1],
    random: createSeededRandom(456),
    statsProvider: async () => ({
      argmaxCanvas: Int32Array.from([5, 6, 7]),
      entropies: new Float32Array([0.001, 0.002, 0.003]),
      selfConditioningLogits: { logitsBuffer: {}, release() {} },
    }),
  });
  assert.deepEqual([...result.argmaxCanvas], [5, 6, 7]);
  assert.equal(result.selfConditioningLogits?.logitsBuffer != null, true);
  assert.equal(result.lastStep.step, 1, 'stabilityThreshold=1 requires one matching previous denoise pass');
  assert.equal(result.stepsRun, 2);
}

{
  const result = await denoiseCanvas(config, {
    initialCanvas: [1, 1, 1],
    random: createSeededRandom(456),
    logitsProvider: async () => buildLogits([5, 6, 7], config.vocabSize),
  });
  assert.deepEqual([...result.argmaxCanvas], [5, 6, 7]);
  assert.ok(result.selfConditioningLogits instanceof Float32Array);
  assert.equal(result.lastStep.step, 1, 'stabilityThreshold=1 requires one matching previous denoise pass');
  assert.equal(result.stepsRun, 2);
}

{
  const manifest = {
    modelId: 'diffusiongemma-fixture',
    modelType: 'diffusion_gemma',
    architecture: {
      numLayers: 1,
      hiddenSize: 4,
      intermediateSize: 8,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 4,
      vocabSize: 8,
      maxSeqLen: 32,
    },
    inference: {
      ...DEFAULT_MANIFEST_INFERENCE,
      diffusionGemma,
    },
    tokenizer: {
      type: 'bundled',
      vocabSize: 8,
      file: 'tokenizer.json',
    },
  };
  const tokenizer = {
    encode: () => [1, 2],
    decode: (ids) => ids.map((id) => `<${id}>`).join(''),
  };
  const logitsProvider = async () => buildLogits([3, 4, 7], config.vocabSize);
  const pipeline = await createPipeline(manifest, {
    diffusionGemma: {
      tokenizer,
      logitsProvider,
    },
  });

  const tokenIds = await pipeline.generateTokenIds('prompt', { maxNewTokens: 3, seed: 789 });
  assert.deepEqual([...tokenIds], [3, 4], 'generation should stop before yielding eos');

  const chunks = [];
  for await (const chunk of pipeline.generate('prompt', { maxNewTokens: 2, seed: 789 })) {
    chunks.push(chunk);
  }
  assert.deepEqual(chunks, ['<3>', '<4>']);
  await pipeline.unload();
}

{
  const manifest = {
    modelId: 'diffusiongemma-core-fixture',
    modelType: 'diffusion_gemma',
    architecture: {
      numLayers: 1,
      hiddenSize: 4,
      intermediateSize: 8,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 4,
      vocabSize: 8,
      maxSeqLen: 32,
    },
    inference: {
      ...DEFAULT_MANIFEST_INFERENCE,
      diffusionGemma: {
        ...diffusionGemma,
        canvasLength: 2,
        maxDenoisingSteps: 1,
        maxNewTokens: 4,
      },
    },
    tokenizer: {
      type: 'bundled',
      vocabSize: 8,
      file: 'tokenizer.json',
    },
  };
  const tokenizer = {
    encode: () => [1, 2],
    decode: (ids) => ids.map((id) => `<${id}>`).join(''),
  };
  const corePipeline = {
    initialized: false,
    loadedManifest: null,
    resetCalls: [],
    prefillInputs: [],
    logitsCalls: [],
    async initialize() {
      this.initialized = true;
    },
    async loadModel(loadedManifest) {
      this.loadedManifest = loadedManifest;
    },
    resetToSeqLen(seqLen) {
      this.resetCalls.push(seqLen);
    },
    async prefillKVOnly(_prompt, options) {
      this.prefillInputs.push([...options.inputIds]);
    },
    async computeDiffusionGemmaCanvasLogits(args, options) {
      this.logitsCalls.push({
        canvas: [...args.canvas],
        selfConditioningLogits: args.selfConditioningLogits,
        useChatTemplate: options.useChatTemplate,
        internalGenerate: options.__internalGenerate,
      });
      const callIndex = this.logitsCalls.length;
      return buildLogits(callIndex === 1 ? [3, 4] : [5, 7], config.vocabSize);
    },
    async unload() {
      this.unloaded = true;
    },
  };
  const pipeline = await createPipeline(manifest, {
    diffusionGemma: {
      tokenizer,
      corePipeline,
    },
  });

  const tokenIds = await pipeline.generateTokenIds('prompt', { maxNewTokens: 4, seed: 321 });
  assert.deepEqual([...tokenIds], [3, 4, 5], 'internal core path should append canvas tokens and stop on eos');
  assert.equal(corePipeline.initialized, true);
  assert.equal(corePipeline.loadedManifest, manifest);
  assert.deepEqual(corePipeline.resetCalls, [0]);
  assert.deepEqual(corePipeline.prefillInputs, [[1, 2], [3, 4]]);
  assert.equal(corePipeline.logitsCalls.length, 2);
  assert.equal(corePipeline.logitsCalls[0].useChatTemplate, false);
  assert.equal(corePipeline.logitsCalls[0].internalGenerate, true);
  assert.equal(corePipeline.logitsCalls[0].selfConditioningLogits, null);
  assert.equal(corePipeline.logitsCalls[1].selfConditioningLogits, null);
  await pipeline.unload();
  assert.equal(corePipeline.unloaded, undefined, 'provided core pipeline ownership stays with caller');
}

{
  const manifest = {
    modelId: 'diffusiongemma-core-stats-fixture',
    modelType: 'diffusion_gemma',
    architecture: {
      numLayers: 1,
      hiddenSize: 4,
      intermediateSize: 8,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 4,
      vocabSize: 8,
      maxSeqLen: 32,
    },
    inference: {
      ...DEFAULT_MANIFEST_INFERENCE,
      diffusionGemma: {
        ...diffusionGemma,
        canvasLength: 4,
        maxDenoisingSteps: 4,
        maxNewTokens: 4,
      },
    },
    tokenizer: {
      type: 'bundled',
      vocabSize: 8,
      file: 'tokenizer.json',
    },
  };
  const tokenizer = {
    encode: () => [1, 2],
    decode: (ids) => ids.map((id) => `<${id}>`).join(''),
  };
  const corePipeline = {
    statsCalls: [],
    async initialize() {},
    async loadModel() {},
    resetToSeqLen() {},
    async prefillKVOnly() {},
    async computeDiffusionGemmaCanvasStep(args, options) {
      this.statsCalls.push({
        canvasLength: args.canvas.length,
        temperature: args.temperature,
        useChatTemplate: options.useChatTemplate,
        internalGenerate: options.__internalGenerate,
      });
      return {
        argmaxCanvas: Int32Array.from([3, 7]),
        entropies: new Float32Array([0.001, 0.002]),
        selfConditioningLogits: { logitsBuffer: {}, release() {} },
      };
    },
  };
  const pipeline = await createPipeline(manifest, {
    diffusionGemma: {
      tokenizer,
      corePipeline,
    },
  });

  const tokenIds = await pipeline.generateTokenIds('prompt', {
    maxNewTokens: 2,
    canvasLength: 2,
    maxDenoisingSteps: 1,
    seed: 321,
  });
  assert.deepEqual([...tokenIds], [3], 'stats path should honor explicit compact canvas and stop on eos');
  assert.equal(corePipeline.statsCalls.length, 1);
  assert.equal(corePipeline.statsCalls[0].canvasLength, 2);
  assert.equal(corePipeline.statsCalls[0].useChatTemplate, false);
  assert.equal(corePipeline.statsCalls[0].internalGenerate, true);
  await assert.rejects(
    () => pipeline.generateTokenIds('prompt', { canvasLength: 5 }),
    /canvasLength 5 exceeds manifest canvasLength 4/
  );
  await pipeline.unload();
}

console.log('diffusion-gemma-sampling.test: ok');
