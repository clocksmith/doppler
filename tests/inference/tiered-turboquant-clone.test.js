import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
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

const { TieredKVCache } = await import('../../src/inference/kv-cache/tiered.js');
const { setDevice } = await import('../../src/gpu/device.js');

class FakeBuffer {
  constructor({ label = '', size, usage }) {
    this.label = label;
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.data = new Uint8Array(size);
  }

  destroy() {
    this.destroyed = true;
  }
}

function copyBytes(src, srcOffset, dst, dstOffset, size) {
  dst.data.set(src.data.subarray(srcOffset, srcOffset + size), dstOffset);
}

function writePattern(buffer, size, seed) {
  for (let i = 0; i < size; i++) {
    buffer.data[i] = (seed + i) & 0xff;
  }
}

function createFakeDevice() {
  return {
    queue: {
      submit(commandBuffers) {
        for (const buffer of commandBuffers) {
          for (const op of buffer.ops) {
            if (op.type === 'copyBufferToBuffer') {
              copyBytes(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
            }
          }
        }
      },
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        buffer.data.set(bytes, offset);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(['shader-f16']),
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
    createBuffer(descriptor) {
      return new FakeBuffer(descriptor);
    },
    createCommandEncoder() {
      const ops = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ type: 'copyBufferToBuffer', src, srcOffset, dst, dstOffset, size });
        },
        finish() {
          return { ops };
        },
      };
    },
  };
}

const device = createFakeDevice();
setDevice(device, { platformConfig: null });

try {
  const cache = new TieredKVCache({
    numLayers: 1,
    numHeads: 1,
    headDim: 64,
    maxSeqLen: 16,
    useGPU: true,
    layout: 'tiered',
    kvDtype: 'f16',
    pageSize: 4,
    tiering: {
      mode: 'turboquant_prod',
      hotWindow: 4,
      coldPageSize: 4,
      coldDtype: 'f16',
      compression: {
        mode: 'turboquant_prod',
        blockSize: 1,
        bitWidth: 4,
        prodMode: true,
      },
      gating: {
        mode: 'force_on',
        minAluBwRatio: 0,
      },
    },
  });

  const layer = cache.coldLayers[0];
  const usedTokens = 3;
  const mseBytes = usedTokens * cache.numHeads * cache.msePackedStride * 4;
  const residualBytes = usedTokens * cache.numHeads * cache.residualPackedStride * 4;
  const scaleBytes = usedTokens * cache.numHeads * 2;

  layer.seqLen = usedTokens;
  cache.currentSeqLen = 7;
  cache.totalTokensSeen = 7;

  writePattern(layer.keysPackedGPU, mseBytes, 11);
  writePattern(layer.valuesPackedGPU, mseBytes, 23);
  writePattern(layer.scalesKGPU, scaleBytes, 37);
  writePattern(layer.scalesVGPU, scaleBytes, 41);
  writePattern(layer.residualKGPU, residualBytes, 53);
  writePattern(layer.residualVGPU, residualBytes, 67);
  writePattern(layer.residualNormsKGPU, scaleBytes, 79);
  writePattern(layer.residualNormsVGPU, scaleBytes, 97);

  const cloned = cache.clone();
  const clonedLayer = cloned.coldLayers[0];

  assert.equal(cloned.currentSeqLen, cache.currentSeqLen);
  assert.equal(cloned.totalTokensSeen, cache.totalTokensSeen);
  assert.equal(clonedLayer.seqLen, layer.seqLen);

  assert.deepEqual(
    Array.from(clonedLayer.keysPackedGPU.data.slice(0, mseBytes)),
    Array.from(layer.keysPackedGPU.data.slice(0, mseBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.valuesPackedGPU.data.slice(0, mseBytes)),
    Array.from(layer.valuesPackedGPU.data.slice(0, mseBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.scalesKGPU.data.slice(0, scaleBytes)),
    Array.from(layer.scalesKGPU.data.slice(0, scaleBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.scalesVGPU.data.slice(0, scaleBytes)),
    Array.from(layer.scalesVGPU.data.slice(0, scaleBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.residualKGPU.data.slice(0, residualBytes)),
    Array.from(layer.residualKGPU.data.slice(0, residualBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.residualVGPU.data.slice(0, residualBytes)),
    Array.from(layer.residualVGPU.data.slice(0, residualBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.residualNormsKGPU.data.slice(0, scaleBytes)),
    Array.from(layer.residualNormsKGPU.data.slice(0, scaleBytes))
  );
  assert.deepEqual(
    Array.from(clonedLayer.residualNormsVGPU.data.slice(0, scaleBytes)),
    Array.from(layer.residualNormsVGPU.data.slice(0, scaleBytes))
  );

  assert.equal(cloned.rotationMatrixBuffer, cache.rotationMatrixBuffer);
  assert.equal(cloned.codebookCentroidsBuffer, cache.codebookCentroidsBuffer);
  assert.equal(cloned.qjlMatrixBuffer, cache.qjlMatrixBuffer);

  cloned.destroy();
  cache.destroy();
} finally {
  setDevice(null);
}

console.log('tiered-turboquant-clone.test: ok');
