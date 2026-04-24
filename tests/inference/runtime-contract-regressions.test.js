import assert from 'node:assert/strict';
import { createExecutionV1Session } from '../helpers/execution-v1-fixtures.js';

const originalFetch = globalThis.fetch;

const { StructuredJsonHeadPipeline } = await import('../../src/inference/pipelines/structured/json-head-pipeline.js');
const { SpeculativeDecoder } = await import('../../src/inference/speculative.js');
const { Tokenizer } = await import('../../src/inference/tokenizer.js');
const { discoverModels } = await import('../../src/inference/test-harness.js');
const { TieredKVCache } = await import('../../src/inference/kv-cache/tiered.js');
const { createKVCache } = await import('../../src/inference/pipelines/text/init.js');
const { resolveAttentionFrequencyBaseDim } = await import('../../src/inference/pipelines/text/layer.js');
const { evolveNetwork } = await import('../../src/inference/network-evolution.js');
const { parseModelConfigFromManifest } = await import('../../src/inference/pipelines/text/config.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { extractExecutionContractFacts } = await import('../../src/config/execution-contract-check.js');
const { SUPPORTED_EXECUTION_V1_OPS } = await import('../../src/config/supported-operations.js');
const {
  DEFAULT_KVCACHE_CONFIG,
  EXECUTION_V1_SCHEMA_ID,
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} = await import('../../src/config/schema/index.js');
const {
  createLinearAttentionRuntime,
  runLinearAttentionLayer,
} = await import('../../src/inference/pipelines/text/linear-attention.js');

function createKernelCapsOnlyDevice() {
  return {
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() {
      return {};
    },
    createBuffer() {
      throw new Error('createBuffer should not be called in this regression test.');
    },
    createCommandEncoder() {
      throw new Error('createCommandEncoder should not be called in this regression test.');
    },
  };
}

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
    for (const op of ['gate_proj', 'up_proj', 'activation', 'down_proj']) {
      assert.equal(
        SUPPORTED_EXECUTION_V1_OPS.has(op),
        true,
        `${op} must be recognized for split-FFN execution-v1 manifests`
      );
    }

    const digest = 'sha256:0000000000000000000000000000000000000000000000000000000000000000';
    const facts = extractExecutionContractFacts({
      modelId: 'split-ffn-contract',
      architecture: {
        headDim: 256,
        maxSeqLen: 1024,
      },
      inference: {
        schema: EXECUTION_V1_SCHEMA_ID,
        session: {
          kvcache: {
            layout: 'contiguous',
            tiering: { mode: 'off' },
          },
          decodeLoop: {
            batchSize: 1,
            disableCommandBatching: false,
          },
        },
        execution: {
          kernels: {
            gemv: { kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4', digest },
            gelu: { kernel: 'gelu.wgsl', entry: 'main', digest },
          },
          decode: [
            ['gate_proj', 'gemv', 'layer.{L}.mlp.gate_proj'],
            ['up_proj', 'gemv', 'layer.{L}.mlp.up_proj'],
            ['activation', 'gelu'],
            ['down_proj', 'gemv', 'layer.{L}.mlp.down_proj'],
          ],
        },
      },
    });
    assert.equal(facts.steps.length, 4);
  }

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
          numAttentionHeads: 1,
          numKeyValueHeads: 1,
          headDim: 4,
          intermediateSize: 8,
          vocabSize: 16,
        },
        eos_token_id: 1,
        tensors: {
          'layers.0.feed_forward.w1.weight': { shape: [12, 4] },
        },
        inference: {
          attention: {
            queryPreAttnScalar: 4,
            queryKeyNorm: false,
            valueNorm: false,
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
            useDoubleWideMlp: false,
            swigluLimit: null,
          },
          rope: {
            ropeTheta: 10000,
            ropeScalingFactor: 1,
            ropeScalingType: null,
            ropeLocalTheta: null,
            ropeLocalScalingType: null,
            ropeLocalScalingFactor: 1,
            mropeInterleaved: false,
            mropeSection: null,
            partialRotaryFactor: null,
            ropeLocalPartialRotaryFactor: null,
            ropeFrequencyBaseDim: null,
            ropeLocalFrequencyBaseDim: null,
            yarnBetaFast: null,
            yarnBetaSlow: null,
            yarnOriginalMaxPos: null,
            ropeLocalYarnBetaFast: null,
            ropeLocalYarnBetaSlow: null,
            ropeLocalYarnOriginalMaxPos: null,
          },
          output: {
            tieWordEmbeddings: false,
            scaleEmbeddings: false,
            embeddingTranspose: false,
            finalLogitSoftcapping: null,
            embeddingVocabSize: null,
            embeddingPostprocessor: null,
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
    assert.throws(
      () => parseModelConfigFromManifest({
        modelId: 'unknown-chat-template',
        modelType: 'text',
        quantization: 'f16',
        architecture: {
          hiddenSize: 4,
          numLayers: 1,
          numAttentionHeads: 1,
          numKeyValueHeads: 1,
          headDim: 4,
          intermediateSize: 8,
          vocabSize: 16,
        },
        eos_token_id: 1,
        inference: {
          attention: {
            queryPreAttnScalar: 4,
            queryKeyNorm: false,
            valueNorm: false,
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
            useDoubleWideMlp: false,
            swigluLimit: null,
          },
          rope: {
            ropeTheta: 10000,
            ropeScalingFactor: 1,
            ropeScalingType: null,
            ropeLocalTheta: null,
            ropeLocalScalingType: null,
            ropeLocalScalingFactor: 1,
            mropeInterleaved: false,
            mropeSection: null,
            partialRotaryFactor: null,
            ropeLocalPartialRotaryFactor: null,
            ropeFrequencyBaseDim: null,
            ropeLocalFrequencyBaseDim: null,
            yarnBetaFast: null,
            yarnBetaSlow: null,
            yarnOriginalMaxPos: null,
            ropeLocalYarnBetaFast: null,
            ropeLocalYarnBetaSlow: null,
            ropeLocalYarnOriginalMaxPos: null,
          },
          output: {
            tieWordEmbeddings: false,
            scaleEmbeddings: false,
            embeddingTranspose: false,
            finalLogitSoftcapping: null,
            embeddingVocabSize: null,
            embeddingPostprocessor: null,
          },
          layerPattern: {
            type: 'global',
            globalPattern: null,
            period: null,
            offset: null,
          },
        chatTemplate: {
          type: 'mystery-template',
          enabled: true,
        },
        session: createExecutionV1Session(),
      },
    }),
      /not a known formatter type/
    );
  }

  {
    const parsed = parseModelConfigFromManifest({
      modelId: 'gemma4-e2b-contract',
      modelType: 'text',
      quantization: 'f16',
      architecture: {
        hiddenSize: 1536,
        numLayers: 35,
        numAttentionHeads: 8,
        numKeyValueHeads: 1,
        headDim: 256,
        globalHeadDim: 512,
        intermediateSize: 6144,
        vocabSize: 262144,
        maxSeqLen: 131072,
        ropeTheta: 1000000,
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262144,
        numKvSharedLayers: 20,
      },
      eos_token_id: 1,
      inference: {
        attention: {
          queryPreAttnScalar: 1,
          queryKeyNorm: true,
          valueNorm: true,
          attentionBias: false,
          causal: true,
          slidingWindow: 512,
          attnLogitSoftcapping: null,
        },
        normalization: {
          rmsNormWeightOffset: false,
          rmsNormEps: 1e-6,
          postAttentionNorm: true,
          preFeedforwardNorm: true,
          postFeedforwardNorm: true,
        },
        ffn: {
          activation: 'gelu',
          gatedActivation: true,
          useDoubleWideMlp: true,
          swigluLimit: null,
        },
        rope: {
          ropeTheta: 1000000,
          ropeScalingFactor: 1,
          ropeScalingType: null,
          ropeLocalTheta: 10000,
          ropeLocalScalingType: null,
          ropeLocalScalingFactor: 1,
          mropeInterleaved: false,
          mropeSection: null,
          partialRotaryFactor: 0.25,
          ropeLocalPartialRotaryFactor: null,
          ropeFrequencyBaseDim: 512,
          ropeLocalFrequencyBaseDim: null,
          yarnBetaFast: null,
          yarnBetaSlow: null,
          yarnOriginalMaxPos: null,
          ropeLocalYarnBetaFast: null,
          ropeLocalYarnBetaSlow: null,
          ropeLocalYarnOriginalMaxPos: null,
        },
        output: {
          tieWordEmbeddings: true,
          scaleEmbeddings: true,
          embeddingTranspose: false,
          finalLogitSoftcapping: 30,
          embeddingVocabSize: null,
          embeddingPostprocessor: null,
        },
        layerPattern: {
          type: 'custom',
          globalPattern: null,
          period: null,
          offset: null,
          layerTypes: Array.from({ length: 35 }, (_, index) => index % 5 === 4 ? 'full_attention' : 'sliding_attention'),
        },
        chatTemplate: {
          type: 'gemma4',
          enabled: true,
        },
        session: createExecutionV1Session(),
      },
    });

    assert.equal(parsed.globalHeadDim, 512);
    assert.equal(parsed.hiddenSizePerLayerInput, 256);
    assert.equal(parsed.numKvSharedLayers, 20);
    assert.equal(parsed.ropeRotaryDim, 128);
    assert.equal(parsed.ropeLocalRotaryDim, 256);
    assert.equal(parsed.ropeFrequencyBaseDim, 512);
    assert.equal(parsed.ropeLocalFrequencyBaseDim, 256);
    assert.equal(resolveAttentionFrequencyBaseDim(parsed, 'full_attention'), 512);
    assert.equal(resolveAttentionFrequencyBaseDim(parsed, 'sliding_attention'), 256);
    assert.equal(parsed.decodeStrategy, 'incremental');
    assert.equal(parsed.vocabSizePerLayerInput, 262144);
    assert.equal(parsed.maxIntermediateSize, 12288);
    assert.equal(parsed.intermediateSizes[14], 6144);
    assert.equal(parsed.intermediateSizes[15], 12288);
  }

  {
    const parsed = parseModelConfigFromManifest({
      modelId: 'gemma4-every-n-contract',
      modelType: 'text',
      quantization: 'f16',
      architecture: {
        hiddenSize: 1536,
        numLayers: 35,
        numAttentionHeads: 8,
        numKeyValueHeads: 1,
        headDim: 256,
        globalHeadDim: 512,
        intermediateSize: 6144,
        intermediateSizes: Array.from({ length: 35 }, (_, index) => index % 15 === 14 ? 12288 : 6144),
        vocabSize: 262144,
        maxSeqLen: 131072,
        ropeTheta: 1000000,
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262144,
        numKvSharedLayers: 20,
      },
      eos_token_id: 1,
      inference: {
        attention: {
          queryPreAttnScalar: 1,
          queryKeyNorm: true,
          valueNorm: true,
          attentionBias: false,
          causal: true,
          slidingWindow: 512,
          attnLogitSoftcapping: null,
        },
        normalization: {
          rmsNormWeightOffset: false,
          rmsNormEps: 1e-6,
          postAttentionNorm: true,
          preFeedforwardNorm: true,
          postFeedforwardNorm: true,
        },
        ffn: {
          activation: 'gelu',
          gatedActivation: true,
          useDoubleWideMlp: true,
          swigluLimit: null,
        },
        rope: {
          ropeTheta: 1000000,
          ropeScalingFactor: 1,
          ropeScalingType: null,
          ropeLocalTheta: 10000,
          ropeLocalScalingType: null,
          ropeLocalScalingFactor: 1,
          mropeInterleaved: false,
          mropeSection: null,
          partialRotaryFactor: 0.25,
          ropeLocalPartialRotaryFactor: null,
          ropeFrequencyBaseDim: 512,
          ropeLocalFrequencyBaseDim: null,
          yarnBetaFast: null,
          yarnBetaSlow: null,
          yarnOriginalMaxPos: null,
          ropeLocalYarnBetaFast: null,
          ropeLocalYarnBetaSlow: null,
          ropeLocalYarnOriginalMaxPos: null,
        },
        output: {
          tieWordEmbeddings: true,
          scaleEmbeddings: true,
          embeddingTranspose: false,
          finalLogitSoftcapping: 30,
          embeddingVocabSize: null,
          embeddingPostprocessor: null,
        },
        layerPattern: {
          type: 'every_n',
          globalPattern: null,
          period: 5,
          offset: 4,
          layerTypes: null,
        },
        chatTemplate: {
          type: 'gemma4',
          enabled: true,
        },
        session: createExecutionV1Session(),
      },
    });

    assert.equal(parsed.layerTypes.length, 35);
    assert.equal(parsed.layerTypes[0], 'sliding_attention');
    assert.equal(parsed.layerTypes[4], 'full_attention');
    assert.equal(parsed.layerTypes[9], 'full_attention');
    assert.equal(parsed.decodeStrategy, 'incremental');
  }

  {
    assert.throws(
      () => parseModelConfigFromManifest({
        modelId: 'gemma4-missing-pooling-kernel',
        modelType: 'text',
        quantization: 'f16',
        architecture: {
          hiddenSize: 1536,
          numLayers: 35,
          numAttentionHeads: 8,
          numKeyValueHeads: 1,
          headDim: 256,
          globalHeadDim: 512,
          intermediateSize: 6144,
          vocabSize: 262144,
          maxSeqLen: 131072,
          ropeTheta: 1000000,
          hiddenSizePerLayerInput: 256,
          vocabSizePerLayerInput: 262144,
          numKvSharedLayers: 20,
        },
        eos_token_id: 1,
        config: {
          vision_config: {
            vision_architecture: 'gemma4',
            model_type: 'gemma4_vision',
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            num_hidden_layers: 16,
            patch_size: 16,
            position_embedding_size: 10240,
            default_output_length: 280,
            rope_parameters: {
              rope_theta: 100.0,
            },
            rms_norm_eps: 1e-6,
            hidden_activation: 'gelu_pytorch_tanh',
            standardize: false,
            use_clipped_linears: true,
          },
        },
        inference: {
          attention: {
            queryPreAttnScalar: 1,
            queryKeyNorm: true,
            valueNorm: true,
            attentionBias: false,
            causal: true,
            slidingWindow: 512,
            attnLogitSoftcapping: null,
          },
          normalization: {
            rmsNormWeightOffset: false,
            rmsNormEps: 1e-6,
            postAttentionNorm: true,
            preFeedforwardNorm: true,
            postFeedforwardNorm: true,
          },
          ffn: {
            activation: 'gelu',
            gatedActivation: true,
            useDoubleWideMlp: true,
            swigluLimit: null,
          },
          rope: {
            ropeTheta: 1000000,
            ropeScalingFactor: 1,
            ropeScalingType: null,
            ropeLocalTheta: 10000,
            ropeLocalScalingType: null,
            ropeLocalScalingFactor: 1,
            mropeInterleaved: false,
            mropeSection: null,
            partialRotaryFactor: 0.25,
            ropeLocalPartialRotaryFactor: null,
            ropeFrequencyBaseDim: 512,
            ropeLocalFrequencyBaseDim: null,
            yarnBetaFast: null,
            yarnBetaSlow: null,
            yarnOriginalMaxPos: null,
            ropeLocalYarnBetaFast: null,
            ropeLocalYarnBetaSlow: null,
            ropeLocalYarnOriginalMaxPos: null,
          },
          output: {
            tieWordEmbeddings: true,
            scaleEmbeddings: true,
            embeddingTranspose: false,
            finalLogitSoftcapping: 30,
            embeddingVocabSize: null,
            embeddingPostprocessor: null,
          },
          layerPattern: {
            type: 'custom',
            globalPattern: null,
            period: null,
            offset: null,
            layerTypes: Array.from({ length: 35 }, (_, index) => index % 5 === 4 ? 'full_attention' : 'sliding_attention'),
          },
          chatTemplate: {
            type: 'gemma4',
            enabled: true,
          },
          session: createExecutionV1Session(),
        },
      }),
      /vision_config\.pooling_kernel_size/
    );
  }

  {
    assert.throws(
      () => parseModelConfigFromManifest({
        modelId: 'gemma4-unsupported-vision-activation',
        modelType: 'text',
        quantization: 'f16',
        architecture: {
          hiddenSize: 1536,
          numLayers: 35,
          numAttentionHeads: 8,
          numKeyValueHeads: 1,
          headDim: 256,
          globalHeadDim: 512,
          intermediateSize: 6144,
          vocabSize: 262144,
          maxSeqLen: 131072,
          ropeTheta: 1000000,
          hiddenSizePerLayerInput: 256,
          vocabSizePerLayerInput: 262144,
          numKvSharedLayers: 20,
        },
        eos_token_id: 1,
        config: {
          vision_config: {
            vision_architecture: 'gemma4',
            model_type: 'gemma4_vision',
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            num_hidden_layers: 16,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            default_output_length: 280,
            rope_parameters: {
              rope_theta: 100.0,
            },
            rms_norm_eps: 1e-6,
            hidden_activation: 'silu',
            standardize: false,
            use_clipped_linears: true,
          },
        },
        inference: {
          attention: {
            queryPreAttnScalar: 1,
            queryKeyNorm: true,
            valueNorm: true,
            attentionBias: false,
            causal: true,
            slidingWindow: 512,
            attnLogitSoftcapping: null,
          },
          normalization: {
            rmsNormWeightOffset: false,
            rmsNormEps: 1e-6,
            postAttentionNorm: true,
            preFeedforwardNorm: true,
            postFeedforwardNorm: true,
          },
          ffn: {
            activation: 'gelu',
            gatedActivation: true,
            useDoubleWideMlp: true,
            swigluLimit: null,
          },
          rope: {
            ropeTheta: 1000000,
            ropeScalingFactor: 1,
            ropeScalingType: null,
            ropeLocalTheta: 10000,
            ropeLocalScalingType: null,
            ropeLocalScalingFactor: 1,
            mropeInterleaved: false,
            mropeSection: null,
            partialRotaryFactor: 0.25,
            ropeLocalPartialRotaryFactor: null,
            ropeFrequencyBaseDim: 512,
            ropeLocalFrequencyBaseDim: null,
            yarnBetaFast: null,
            yarnBetaSlow: null,
            yarnOriginalMaxPos: null,
            ropeLocalYarnBetaFast: null,
            ropeLocalYarnBetaSlow: null,
            ropeLocalYarnOriginalMaxPos: null,
          },
          output: {
            tieWordEmbeddings: true,
            scaleEmbeddings: true,
            embeddingTranspose: false,
            finalLogitSoftcapping: 30,
            embeddingVocabSize: null,
            embeddingPostprocessor: null,
          },
          layerPattern: {
            type: 'custom',
            globalPattern: null,
            period: null,
            offset: null,
            layerTypes: Array.from({ length: 35 }, (_, index) => index % 5 === 4 ? 'full_attention' : 'sliding_attention'),
          },
          chatTemplate: {
            type: 'gemma4',
            enabled: true,
          },
          session: createExecutionV1Session(),
        },
      }),
      /unsupported Gemma 4 vision hidden_activation/
    );
  }

  {
    assert.throws(
      () => createKVCache({
        numLayers: 35,
        numKVHeads: 1,
        headDim: 256,
        globalHeadDim: 512,
        maxSeqLen: 131072,
        slidingWindow: 512,
        attnLogitSoftcapping: null,
        layerTypes: Array.from({ length: 35 }, (_, index) => index % 5 === 4 ? 'full_attention' : 'sliding_attention'),
        numKvSharedLayers: 20,
        decodeStrategy: 'incremental',
      }, false, false, DEFAULT_KVCACHE_CONFIG),
      /requires GPU execution/
    );
  }

  {
    setDevice(createKernelCapsOnlyDevice(), { platformConfig: null });
    try {
      const runtimeKV = {
        ...DEFAULT_KVCACHE_CONFIG,
        maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
      };

      const slidingOnlyCache = createKVCache({
        numLayers: 1,
        numKVHeads: 1,
        headDim: 8,
        maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
        slidingWindow: 1024,
        attnLogitSoftcapping: null,
        layerTypes: ['sliding_attention'],
      }, false, false, runtimeKV);
      assert.equal(slidingOnlyCache.layout, 'paged');

      const fullAttentionCache = createKVCache({
        numLayers: 1,
        numKVHeads: 1,
        headDim: 8,
        maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
        slidingWindow: 1024,
        attnLogitSoftcapping: null,
        layerTypes: ['sliding_attention', 'full_attention'],
      }, false, false, runtimeKV);
      assert.equal(fullAttentionCache.layout, 'contiguous');
    } finally {
      setDevice(null);
    }
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
