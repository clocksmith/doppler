import assert from 'node:assert/strict';

const originalFetch = globalThis.fetch;

const { StructuredJsonHeadPipeline } = await import('../../src/inference/pipelines/structured/json-head-pipeline.js');
const { SpeculativeDecoder } = await import('../../src/inference/speculative.js');
const { Tokenizer } = await import('../../src/inference/tokenizer.js');
const { discoverModels } = await import('../../src/inference/test-harness.js');
const { TieredKVCache } = await import('../../src/inference/kv-cache/tiered.js');
const { evolveNetwork } = await import('../../src/inference/network-evolution.js');
const { parseModelConfigFromManifest } = await import('../../src/inference/pipelines/text/config.js');
const {
  createLinearAttentionRuntime,
  runLinearAttentionLayer,
} = await import('../../src/inference/pipelines/text/linear-attention.js');

function createStructuredPipeline(manifest, runtimeConfig = {}) {
  const pipeline = new StructuredJsonHeadPipeline();
  pipeline.manifest = manifest;
  pipeline.runtimeConfig = runtimeConfig;
  pipeline.reset = () => {};
  pipeline.generate = async function* generate() {
    yield '{"ok":true}';
  };
  return pipeline;
}

try {
  {
    const pipeline = createStructuredPipeline({
      modelId: 'dream-structured',
      inference: {
        dream: {
          maxTokens: 32,
          temperature: 0,
          maxOutputChars: 4096,
        },
      },
    }, {
      inference: {
        dream: {
          maxTokens: 16,
          temperature: 0,
          maxOutputChars: 4096,
        },
      },
    });

    await assert.rejects(
      () => pipeline.inferJSON({ prompt: 'test' }),
      /manifest\.inference\.structuredJsonHead is required/
    );
  }

  {
    assert.throws(
      () => new SpeculativeDecoder({
        numDraftTokens: 2,
        maxRejectionRetries: 1,
        enableTreeDraft: false,
        temperature: 1,
      }),
      /requires randomSeed/
    );
  }

  {
    const tokenizer = new Tokenizer();
    await assert.rejects(
      () => tokenizer.initialize({
        modelId: 'runtime-fallback-model',
        tokenizer: {
          type: 'huggingface',
          allowArchFallback: true,
        },
      }),
      /requires explicit tokenizer\.hfModel or tokenizer\.modelId/
    );
  }

  {
    globalThis.fetch = async () => {
      throw new Error('catalog unavailable');
    };

    await assert.rejects(
      () => discoverModels(),
      /no explicit fallback model list was provided/
    );

    const models = await discoverModels(['explicit-model']);
    assert.deepEqual(models, [{ id: 'explicit-model', name: 'explicit-model' }]);
  }

  {
    assert.throws(
      () => new TieredKVCache({
        numLayers: 1,
        numHeads: 2,
        headDim: 32,
        maxSeqLen: 128,
        useGPU: true,
        layout: 'tiered',
        pageSize: 32,
        kvDtype: 'f16',
        tiering: {
          hotWindow: 32,
          coldPageSize: 64,
          mode: 'int8',
          compression: { mode: 'int8', blockSize: 1 },
          gating: { mode: 'auto', minAluBwRatio: 1.5 },
        },
      }),
      /requires an explicit measured ALU\/BW ratio/
    );
  }

  {
    assert.throws(
      () => parseModelConfigFromManifest({
        modelId: 'lfm2-mismatch',
        modelType: 'text',
        quantization: 'f16',
        architecture: {
          hiddenSize: 4,
          numLayers: 1,
          numHeads: 1,
          numKVHeads: 1,
          headDim: 4,
          intermediateSize: 8,
          vocabSize: 16,
        },
        eos_token_id: 1,
        tensors: {
          'layers.0.feed_forward.w1.weight': { shape: [12, 4] },
        },
        inference: {
          presetId: 'lfm2',
          attention: {
            queryPreAttnScalar: 4,
            queryKeyNorm: false,
            attentionBias: false,
            causal: true,
            slidingWindow: null,
            attnLogitSoftcapping: null,
          },
          normalization: {
            rmsNormWeightOffset: false,
            rmsNormEps: 1e-6,
            postAttentionNorm: false,
            preFeedforwardNorm: false,
            postFeedforwardNorm: false,
          },
          ffn: {
            activation: 'silu',
            gatedActivation: true,
            swigluLimit: null,
          },
          rope: {
            ropeTheta: 10000,
            ropeScalingFactor: 1,
            ropeScalingType: null,
            ropeLocalTheta: null,
            mropeInterleaved: false,
            mropeSection: null,
            partialRotaryFactor: null,
            yarnBetaFast: null,
            yarnBetaSlow: null,
            yarnOriginalMaxPos: null,
          },
          output: {
            tieWordEmbeddings: false,
            scaleEmbeddings: false,
            embeddingTranspose: false,
            finalLogitSoftcapping: null,
            embeddingVocabSize: null,
          },
          layerPattern: {
            type: 'global',
            globalPattern: null,
            period: null,
            offset: null,
          },
          chatTemplate: {
            type: null,
            enabled: false,
          },
        },
      }),
      /FFN tensors imply 12/
    );
  }

  {
    await assert.rejects(
      () => evolveNetwork({
        populationSize: 2,
        generations: 1,
        eliteCount: 1,
        evaluate: async () => 1,
        randomGenome: () => ({ topology: { type: 'chain' }, nodes: [], edges: [] }),
      }),
      /requires an explicit random\(\) source/
    );
  }

  {
    const baseOptions = {
      layerIdx: 0,
      numTokens: 1,
      hiddenSize: 2,
      currentSeqLen: 0,
      activationDtype: 'f32',
      kernelPath: null,
      linearRuntime: createLinearAttentionRuntime(),
      getWeightBuffer(weight) {
        return weight;
      },
      getNormWeightBuffer(weight) {
        return weight;
      },
    };
    const layerWeights = {
      qkvProj: new Float32Array(12),
      oProj: new Float32Array(4),
      linearInProjZ: new Float32Array(4),
      linearInProjA: new Float32Array(2),
      linearInProjB: new Float32Array(2),
      linearConv1D: new Float32Array(72),
      linearDtBias: new Float32Array(1),
      linearALog: new Float32Array(1),
      linearNorm: new Float32Array(2),
    };
    const inputTensor = {
      dtype: 'f32',
      shape: [1, 2],
      buffer: { size: 8 },
    };

    await assert.rejects(
      () => runLinearAttentionLayer(inputTensor, layerWeights, {
        ...baseOptions,
        config: {
          linearNumKeyHeads: 1,
          linearNumValueHeads: 1,
          linearKeyHeadDim: 2,
          linearValueHeadDim: 2,
        },
      }),
      /requires linearConvKernelDim/
    );

    await assert.rejects(
      () => runLinearAttentionLayer(inputTensor, layerWeights, {
        ...baseOptions,
        config: {
          linearNumKeyHeads: 1,
          linearNumValueHeads: 1,
          linearKeyHeadDim: 2,
          linearValueHeadDim: 2,
          linearConvKernelDim: 12,
        },
      }),
      /requires a positive rmsNormEps/
    );
  }
} finally {
  globalThis.fetch = originalFetch;
}

console.log('runtime-contract-regressions.test: ok');
