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

globalThis.GPUMapMode = {
  READ: 1 << 0,
  WRITE: 1 << 1,
};

const { runProbes } = await import('../../src/inference/pipelines/text/probes.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { destroyBufferPool } = await import('../../src/memory/buffer-pool.js');
const { setTrace } = await import('../../src/debug/config.js');
const { trace } = await import('../../src/debug/trace.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.data = new Uint8Array(size);
  }

  async mapAsync() {}

  getMappedRange(offset = 0, size = this.size - offset) {
    return this.data.slice(offset, offset + size).buffer;
  }

  unmap() {}

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  return {
    queue: {
      submit(commandBuffers) {
        for (const buffer of commandBuffers) {
          for (const op of buffer.ops) {
            if (op.type === 'copyBufferToBuffer') {
              const bytes = op.src.data.subarray(op.srcOffset, op.srcOffset + op.size);
              op.dst.data.set(bytes, op.dstOffset);
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
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
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

configurePerfGuards({
  allowGPUReadback: true,
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
});

const device = createFakeDevice();
setDevice(device, { platformConfig: null });
setTrace(['embed']);

const messages = [];
const originalEmbed = trace.embed;
trace.embed = (message) => {
  messages.push(message);
};

try {
  const buffer = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  new Float32Array(buffer.data.buffer).set([1.5, 2.5]);

  await runProbes('embed_out', buffer, {
    layerIdx: null,
    numTokens: 1,
    hiddenSize: 2,
    probes: [
      {
        stage: 'embed_out',
        dims: [0, 1],
        tokens: [0],
      },
    ],
    recorder: null,
    dtype: 'f32',
  });

  assert.equal(messages.length, 1);
  assert.match(messages[0], /values=\[0=1\.5000, 1=2\.5000\]/);
} finally {
  trace.embed = originalEmbed;
  destroyBufferPool();
  setTrace(false);
  setDevice(null);
}

console.log('probes-readback.test: ok');
