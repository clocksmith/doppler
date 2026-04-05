import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const { setDevice } = await import('../../src/gpu/device.js');
const { destroyBufferPool, acquireBuffer, uploadData } = await import('../../src/memory/buffer-pool.js');
const { createDopplerConfig, DEFAULT_KVCACHE_CONFIG } = await import('../../src/config/schema/index.js');
const { parseModelConfigFromManifest } = await import('../../src/inference/pipelines/text/config.js');
const { createKVCache } = await import('../../src/inference/pipelines/text/init.js');
const { PipelineGenerator } = await import('../../src/inference/pipelines/text/generator.js');
const { compileExecutionPlanState } = await import('../../src/inference/pipelines/text/execution-plan.js');
const { MixedGeometryKVCache } = await import('../../src/inference/kv-cache/mixed-geometry.js');

let nextBufferId = 0;

function createMockGPUBuffer(descriptor) {
  const size = descriptor?.size ?? 0;
  return {
    __dopplerFakeGPUBuffer: true,
    label: descriptor?.label ?? '',
    size,
    usage: descriptor?.usage ?? 0,
    destroyed: false,
    _id: nextBufferId++,
    _arrayBuffer: new ArrayBuffer(size),
    destroy() {
      this.destroyed = true;
    },
    mapAsync() {
      return Promise.resolve();
    },
    getMappedRange(offset = 0, sizeBytes = this.size - offset) {
      return this._arrayBuffer.slice(offset, offset + sizeBytes);
    },
    unmap() {},
  };
}

function writeBytes(buffer, offset, data) {
  const source = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const target = new Uint8Array(buffer._arrayBuffer, offset, source.byteLength);
  target.set(source);
}

function createMockDevice() {
  return {
    queue: {
      submit(commandBuffers) {
        for (const commandBuffer of commandBuffers) {
          for (const op of commandBuffer.ops ?? []) {
            const src = new Uint8Array(op.src._arrayBuffer, op.srcOffset, op.size);
            const dst = new Uint8Array(op.dst._arrayBuffer, op.dstOffset, op.size);
            dst.set(src);
          }
        }
      },
      writeBuffer(buffer, offset, data) {
        writeBytes(buffer, offset, data);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 28,
      maxBufferSize: 1 << 28,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 16,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createBuffer(descriptor) {
      return createMockGPUBuffer(descriptor);
    },
    createCommandEncoder() {
      const ops = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ src, srcOffset, dst, dstOffset, size });
        },
        finish() {
          return { ops };
        },
      };
    },
  };
}

function createGemma4EveryNManifest() {
  return {
    modelId: 'gemma4-every-n-prefill-embedding',
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
    },
  };
}

function createTestState(modelConfig, runtimeConfig, kvCache, executionPlanState) {
  return {
    tokenizer: null,
    kvCache,
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
    executionV1State: null,
    disableRecordedLogits: false,
    disableFusedDecode: false,
    manifest: { modelType: 'text' },
    modelConfig,
    weights: new Map([
      ['final_norm', new Float32Array(modelConfig.hiddenSize).fill(1)],
    ]),
    expertWeights: new Map(),
    isLoaded: true,
    isGenerating: false,
    currentSeqLen: 0,
    currentTokenIds: null,
    runtimeConfig,
    dopplerLoader: null,
    gpuContext: null,
    useGPU: true,
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
      prefillProfileSteps: [],
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
    useTiedEmbeddings: true,
    embeddingVocabSize: null,
    embeddingTranspose: false,
    embeddingPostprocessor: null,
    layerRouterWeights: null,
    lora: null,
  };
}

setDevice(createMockDevice(), { platformConfig: null });
destroyBufferPool();

let result = null;
let state = null;

try {
  const modelConfig = parseModelConfigFromManifest(createGemma4EveryNManifest());
  assert.equal(modelConfig.decodeStrategy, 'incremental');
  assert.equal(modelConfig.layerTypes.length, modelConfig.numLayers);
  assert.equal(modelConfig.layerTypes[4], 'full_attention');

  const runtimeConfig = createDopplerConfig({
    runtime: {
      inference: {
        compute: {
          activationDtype: 'f32',
        },
        generation: {
          embeddingMode: 'last',
        },
        chatTemplate: {
          enabled: false,
        },
        session: {
          decodeLoop: {
            batchSize: 1,
            stopCheckMode: 'batch',
            readbackInterval: 1,
            readbackMode: 'sequential',
            disableCommandBatching: false,
          },
          kvcache: {
            ...DEFAULT_KVCACHE_CONFIG,
            maxSeqLen: 32,
            kvDtype: 'f32',
            layout: 'contiguous',
            tiering: {
              ...DEFAULT_KVCACHE_CONFIG.tiering,
              mode: 'off',
            },
            quantization: {
              ...DEFAULT_KVCACHE_CONFIG.quantization,
              mode: 'none',
            },
          },
        },
      },
    },
  }).runtime;

  const executionPlanState = compileExecutionPlanState({
    runtimeConfig: {
      inference: runtimeConfig.inference,
      shared: runtimeConfig.shared,
    },
    resolvedKernelPath: null,
    kernelPathSource: 'none',
  });

  const kvCache = createKVCache(modelConfig, true, false, runtimeConfig.inference);
  assert.ok(kvCache instanceof MixedGeometryKVCache);

  state = createTestState(modelConfig, runtimeConfig, kvCache, executionPlanState);

  const generator = new PipelineGenerator(state);
  generator._prefillToHidden = async function (inputIds) {
    state.kvCache.truncate(inputIds.length);
    const hidden = new Float32Array(inputIds.length * modelConfig.hiddenSize);
    for (let i = 0; i < hidden.length; i++) {
      hidden[i] = (i % 17) + 1;
    }
    const currentHiddenBuffer = acquireBuffer(hidden.byteLength, undefined, 'mixed_geometry_prefill_hidden');
    uploadData(currentHiddenBuffer, hidden);
    return {
      numTokens: inputIds.length,
      config: modelConfig,
      startPos: 0,
      activationDtype: 'f32',
      activationBytes: Float32Array.BYTES_PER_ELEMENT,
      currentRecorder: null,
      recordProfile: async () => {},
      currentHiddenBuffer,
    };
  };

  result = await generator.prefillWithEmbedding('', {
    inputIds: [11, 22, 33],
    useChatTemplate: false,
  });

  assert.equal(state.currentSeqLen, 3);
  assert.equal(result.seqLen, 3);
  assert.deepEqual(result.tokens, [11, 22, 33]);
  assert.equal(result.embeddingMode, 'last');
  assert.ok(result.embedding instanceof Float32Array);
  assert.equal(result.embedding.length, modelConfig.hiddenSize);
  assert.ok(result.cache instanceof MixedGeometryKVCache);
  assert.equal(result.cache.getGPUBuffers(0)?.layout, 'ring');
  assert.equal(result.cache.getGPUBuffers(0)?.seqLen, 3);
  assert.equal(result.cache.getGPUBuffers(4)?.layout, 'contiguous');
  assert.equal(result.cache.getGPUBuffers(4)?.seqLen, 3);
  assert.ok(result.linearAttention?.layers instanceof Map);
} finally {
  result?.cache?.destroy?.();
  state?.kvCache?.destroy?.();
  destroyBufferPool();
  setDevice(null);
}

console.log('mixed-geometry-prefill-with-embedding.test: ok');
