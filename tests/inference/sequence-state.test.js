import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { getRuntimeConfig } from '../../src/config/runtime.js';
import { destroyBufferPool, getBufferPool } from '../../src/memory/buffer-pool.js';
import { createKVCache } from '../../src/inference/pipelines/text/init.js';
import { createLayerPartitionPlan } from '../../src/config/layer-partition-plan.js';
import {
  beginGenerationContext,
  createGenerationContext,
  finishGenerationContext,
  generationContextInput,
  prepareGenerationContextPrefill,
  resetSequenceState,
  supportsGenerationContext,
} from '../../src/inference/pipelines/text/sequence-state.js';

const truncated = [];
const state = { isGenerating: false, currentSeqLen: 8, kvCache: { truncate: n => truncated.push(n) } };
for (const invalid of ['invalid', '2', NaN, Infinity, -Infinity, -1, 1.9, null, undefined, {}, Number.MAX_SAFE_INTEGER + 1]) {
  assert.throws(() => resetSequenceState(state, invalid), /non-negative safe integer/);
  assert.equal(state.currentSeqLen, 8);
  assert.deepEqual(truncated, []);
}
assert.throws(() => resetSequenceState(state, 9), /exceeds/);
state.isGenerating = true;
assert.throws(() => resetSequenceState(state, 0), /in progress/);
assert.deepEqual(truncated, []);
state.isGenerating = false;
resetSequenceState(state, 3);
resetSequenceState(state, 0);
assert.deepEqual(truncated, [3, 0]);
assert.equal(state.currentSeqLen, 0);
const failure = { isGenerating: false, currentSeqLen: 8, kvCache: { truncate() { throw new Error('cache failed'); } } };
assert.throws(() => resetSequenceState(failure, 2), /cache failed/);
assert.equal(failure.currentSeqLen, 8);

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

class FakeBuffer {
  constructor({ size, usage = 0, label = '' }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.bytes = new Uint8Array(size);
    this.destroyed = false;
  }
  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  return {
    lost: new Promise(() => {}),
    features: new Set(['shader-f16']),
    limits: {
      maxStorageBufferBindingSize: 16 * 1024 * 1024,
      maxBufferSize: 16 * 1024 * 1024,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 10,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
      minStorageBufferOffsetAlignment: 16,
    },
    queue: {
      submit(commands) {
        for (const cmd of commands) {
          for (const op of cmd.ops || []) op();
        }
      },
      writeBuffer(buffer, dstOffset, src) {
        const view = ArrayBuffer.isView(src)
          ? new Uint8Array(src.buffer, src.byteOffset, src.byteLength)
          : new Uint8Array(src);
        buffer.bytes.set(view, dstOffset);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBuffer({ size, usage, label }) {
      return new FakeBuffer({ size, usage, label });
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createCommandEncoder() {
      const ops = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push(() => {
            dst.bytes.set(src.bytes.subarray(srcOffset, srcOffset + size), dstOffset);
          });
        },
        clearBuffer(dst, offset = 0, size = dst.size - offset) {
          ops.push(() => {
            dst.bytes.fill(0, offset, offset + size);
          });
        },
        finish() {
          return { ops };
        },
      };
    },
  };
}

try {
  destroyBufferPool();
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });

  const plan = createLayerPartitionPlan({
    modelId: 'test-hybrid-conv-hc',
    numLayers: 2,
    hiddenSize: 8,
    vocabSize: 32,
    splitLayer: 1,
    activationDtype: 'f32',
  });
  const manifest = {
    modelId: 'test-hybrid-conv-hc',
    architecture: { numLayers: 2, hiddenSize: 8, vocabSize: 32 },
  };
  const modelConfig = {
    numLayers: 2,
    hiddenSize: 8,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 8,
    maxSeqLen: 16,
    vocabSize: 32,
    layerTypes: ['conv', 'full_attention'],
    hyperConnections: { streamCount: 4, streamWidth: 8 },
  };
  const runtimeConfig = structuredClone(getRuntimeConfig());
  runtimeConfig.inference.compute.activationDtype = 'f32';
  runtimeConfig.inference.session.kvcache.layout = 'contiguous';
  runtimeConfig.inference.session.kvcache.kvDtype = 'f32';
  runtimeConfig.inference.session.kvcache.maxSeqLen = 16;

  const kvCache = createKVCache(modelConfig, true, false, runtimeConfig.inference, [0, 0]);

  const convBuffer = device.createBuffer({ size: 32, label: 'L0.conv_state' });
  const pleConvBuffer = device.createBuffer({ size: 64, label: 'L0.ple_conv' });
  const qsaRawBuffer = device.createBuffer({ size: 16 * 32, label: 'L0.qsa_raw' });
  const qsaBlockBuffer = device.createBuffer({ size: 4 * 32, label: 'L0.qsa_block' });

  let pleHistoryTokens = [];
  const qsaLayer = { rawKeys: qsaRawBuffer, blockKeys: qsaBlockBuffer, length: 0, reservedEnd: 0 };
  const hyperSession = {
    layers: new Map([
      [0, {
        conv: {
          buffer: pleConvBuffer,
          reset() {
            pleConvBuffer.bytes.fill(0);
          },
        },
        history: {
          get length() {
            return pleHistoryTokens.length;
          },
          reset() {
            pleHistoryTokens = [];
          },
          append(tokens) {
            pleHistoryTokens.push(...tokens);
            return new Uint32Array(tokens.length);
          },
          checkpoint() {
            return { epoch: 0, length: pleHistoryTokens.length };
          },
          restore(saved) {
            pleHistoryTokens.length = saved.length;
          },
        },
      }],
    ]),
    qsa: {
      contract: { headDim: 8, compressRatio: 4, layers: [0] },
      capacity: 16,
      epoch: 0,
      layers: new Map([[0, qsaLayer]]),
      length(idx) {
        return this.layers.get(idx).length;
      },
      reset() {
        qsaLayer.length = 0;
        qsaLayer.reservedEnd = 0;
        qsaRawBuffer.bytes.fill(0);
        qsaBlockBuffer.bytes.fill(0);
        this.epoch += 1;
      },
      checkpoint() {
        return { epoch: this.epoch, lengths: { 0: qsaLayer.length } };
      },
      restore(saved) {
        qsaLayer.length = saved.lengths[0];
        qsaLayer.reservedEnd = saved.lengths[0];
        this.epoch += 1;
      },
    },
    reset() {
      this.qsa.reset();
      for (const layer of this.layers.values()) {
        layer.history.reset();
        layer.conv.reset();
      }
    },
    assertLength(expected) {
      assert.equal(pleHistoryTokens.length, expected);
      assert.equal(qsaLayer.length, expected);
    },
    async checkpoint() {
      const conv = device.createBuffer({ size: pleConvBuffer.size, label: 'tx_ple_conv' });
      conv.bytes.set(pleConvBuffer.bytes);
      return {
        epoch: 0,
        qsa: this.qsa.checkpoint(),
        layers: new Map([[0, { history: this.layers.get(0).history.checkpoint(), conv }]]),
      };
    },
    async restore(snapshot) {
      pleConvBuffer.bytes.set(snapshot.layers.get(0).conv.bytes);
      this.layers.get(0).history.restore(snapshot.layers.get(0).history);
      this.qsa.restore(snapshot.qsa);
    },
    releaseCheckpoint(snapshot) {
      for (const layer of snapshot.layers.values()) layer.conv.destroy();
    },
  };

  const pipelineState = {
    useGPU: true,
    debug: false,
    manifest,
    modelConfig,
    runtimeConfig,
    modelPartition: { plan, index: 0 },
    kvCache,
    linearAttentionRuntime: { schemaVersion: 4, layers: new Map() },
    convLayerStates: new Map([[0, { convStateGPU: convBuffer, hiddenSize: 8, kernelSize: 2 }]]),
    hyperConnectionSession: hyperSession,
    tokenizer: { encode: text => Array.from(text, c => c.charCodeAt(0) % 32) },
    currentSeqLen: 0,
    decodeStepCount: 0,
    lora: null,
    stats: { reusedPrefixTokens: 0, prefillComputedTokens: 0 },
  };

  assert.equal(supportsGenerationContext(pipelineState), true, 'Partitioned conv + hyperConnection state should support GenerationContext');
  assert.equal(
    supportsGenerationContext({ ...pipelineState, convLayerStates: new Map() }),
    false,
    'Missing owned convStateGPU buffer must reject GenerationContext'
  );
  assert.equal(
    supportsGenerationContext({ ...pipelineState, hyperConnectionSession: null }),
    false,
    'Missing hyperConnectionSession must reject GenerationContext'
  );

  const context = createGenerationContext(pipelineState, { maxBytes: 1024 * 1024 });
  const options = { inputIds: [1, 2, 3, 4, 5], useChatTemplate: false, executionPlan: null, speculation: null };
  const generator = {
    _state: pipelineState,
    _resolvePrefillTokenChunkSize: () => null,
    async _prefillToHidden(ids) {
      for (const token of ids) {
        convBuffer.bytes[0] = (convBuffer.bytes[0] + token) & 0xff;
        pleConvBuffer.bytes[0] = (pleConvBuffer.bytes[0] + token * 3) & 0xff;
      }
      hyperSession.layers.get(0).history.append(ids);
      const prevLen = qsaLayer.length;
      qsaLayer.length = prevLen + ids.length;
      qsaLayer.reservedEnd = qsaLayer.length;
      qsaRawBuffer.bytes.fill(7, prevLen * 32, qsaLayer.length * 32);
      if (qsaLayer.length >= 4) qsaBlockBuffer.bytes.fill(9, 0, Math.floor(qsaLayer.length / 4) * 32);
      return { numTokens: ids.length, startPos: pipelineState.currentSeqLen };
    },
    async _commitPrefillHiddenChunk(res) {
      pipelineState.currentSeqLen = res.startPos + res.numTokens;
    },
  };

  beginGenerationContext(pipelineState, context, options);
  generationContextInput(pipelineState, options.inputIds);
  const checkpoint = await prepareGenerationContextPrefill(generator, '', options.inputIds, options);
  assert.equal(checkpoint, 4, 'Checkpoint should park first 4 tokens');
  assert.equal(pipelineState.currentSeqLen, 4);
  const expectedConvByte = (1 + 2 + 3 + 4) & 0xff;
  const expectedPleByte = ((1 + 2 + 3 + 4) * 3) & 0xff;
  assert.equal(convBuffer.bytes[0], expectedConvByte);
  assert.equal(pleConvBuffer.bytes[0], expectedPleByte);

  // Mutate working state as if token 5 and decode tokens ran, then finish context.
  convBuffer.bytes[0] = 250;
  pleConvBuffer.bytes[0] = 251;
  await finishGenerationContext(pipelineState, [1, 2, 3, 4, 5, 6], true);
  assert.equal(context.getStats().status, 'cached');
  assert.equal(context.getStats().cachedTokens, 4);
  assert.ok(context.getStats().allocatedBytes > 0);

  // Simulate next generation start resetting the working session and mutating conv state.
  hyperSession.reset();
  convBuffer.bytes.fill(0);
  const extendedOptions = { inputIds: [1, 2, 3, 4, 5, 6, 7], useChatTemplate: false, executionPlan: null, speculation: null };
  beginGenerationContext(pipelineState, context, extendedOptions);
  const suffix = generationContextInput(pipelineState, extendedOptions.inputIds);
  assert.deepEqual(suffix, [5, 6, 7]);
  assert.equal(pipelineState.stats.reusedPrefixTokens, 4);
  const secondCheckpoint = await prepareGenerationContextPrefill(generator, '', extendedOptions.inputIds, extendedOptions);
  assert.equal(secondCheckpoint, 6);
  assert.equal(convBuffer.bytes[0], (expectedConvByte + 5 + 6) & 0xff, 'Conv state must resume from parked 4-token snapshot');
  assert.equal(pleConvBuffer.bytes[0], (expectedPleByte + (5 + 6) * 3) & 0xff, 'PLE conv state must resume from parked 4-token snapshot');
  assert.deepEqual(pleHistoryTokens, [1, 2, 3, 4, 5, 6], 'PLE token history must resume from parked 4-token snapshot');
  assert.equal(qsaLayer.length, 6, 'QSA length must resume from parked 4-token snapshot');
  assert.equal(qsaRawBuffer.bytes[0], 7, 'QSA rawKeys prefix must be restored on GPU');
  assert.equal(qsaBlockBuffer.bytes[0], 9, 'QSA blockKeys prefix must be restored on GPU');
  await finishGenerationContext(pipelineState, [1, 2, 3, 4, 5, 6, 7, 8], true);
  assert.equal(context.getStats().cachedTokens, 6);

  context.close();
  assert.equal(context.getStats().allocatedBytes, 0);
} finally {
  destroyBufferPool();
  setDevice(null, { platformConfig: null });
}
console.log('sequence-state.test: ok');
