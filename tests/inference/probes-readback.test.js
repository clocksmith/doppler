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
const { OperatorEventEmitter } = await import('../../src/inference/pipelines/text/operator-events.js');
const { createDefaultCaptureConfig } = await import('../../src/debug/capture-policy.js');
const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
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
const attnMessages = [];
const originalEmbed = trace.embed;
const originalAttn = trace.attn;
trace.embed = (message) => {
  messages.push(message);
};
trace.attn = (_layerIdx, message) => {
  attnMessages.push(message);
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

  await runProbes('post_input_norm', buffer, {
    layerIdx: 0,
    numTokens: 1,
    hiddenSize: 2,
    probes: [
      {
        id: 'L0_norm',
        stage: 'post_input_norm',
        layers: [0],
        dims: [0, 1],
        tokens: [0],
      },
    ],
    recorder: null,
    dtype: 'f32',
  });

  assert.equal(attnMessages.length, 1);
  assert.match(attnMessages[0], /PROBE L0_norm stage=post_input_norm token=0 values=\[0=1\.5000, 1=2\.5000\]/);

  const diagnostics = {
    enabled: true,
    captureConfig: {
      ...createDefaultCaptureConfig(),
      enabled: true,
      defaultLevel: 'none',
    },
    emitter: new OperatorEventEmitter({
      modelHash: 'probe-readback-test',
      runtimeConfigHash: 'runtime',
      executionPlanHash: 'plan',
    }),
  };
  const cpuBuffer = new Float32Array([3.5, 4.5]);

  await runProbes('embed_out', cpuBuffer, {
    numTokens: 1,
    hiddenSize: 2,
    probes: [],
    operatorDiagnostics: diagnostics,
    dtype: 'f32',
    phase: 'prefill',
  });

  await runProbes('logits', cpuBuffer, {
    numTokens: 1,
    hiddenSize: 2,
    probes: [],
    recorder: {},
    operatorDiagnostics: diagnostics,
    dtype: 'f32',
    phase: 'decode',
  });

  await runProbes('per_layer_embed_out', cpuBuffer, {
    numTokens: 1,
    hiddenSize: 2,
    probes: [],
    operatorDiagnostics: diagnostics,
    dtype: 'f32',
    phase: 'prefill',
    layerIdx: 3,
  });

  assert.equal(diagnostics.emitter.length, 3);
  assert.deepEqual(
    diagnostics.emitter.getTimeline().map((record) => ({
      opId: record.opId,
      phase: record.phase,
      capturePolicy: record.capturePolicy,
    })),
    [
      {
        opId: 'embed.out',
        phase: 'prefill',
        capturePolicy: 'none',
      },
      {
        opId: 'logits.out',
        phase: 'decode',
        capturePolicy: 'none',
      },
      {
        opId: 'layer.3.per_layer_embed.out',
        phase: 'prefill',
        capturePolicy: 'none',
      },
    ]
  );

  const recorderDiagnostics = {
    enabled: true,
    captureConfig: {
      ...createDefaultCaptureConfig(),
      enabled: true,
      defaultLevel: 'slice',
    },
    emitter: new OperatorEventEmitter({
      modelHash: 'probe-readback-test',
      runtimeConfigHash: 'runtime',
      executionPlanHash: 'plan',
    }),
  };
  const recorder = new CommandRecorder(device, 'probe_capture');

  await runProbes('embed_out', buffer, {
    numTokens: 1,
    hiddenSize: 2,
    probes: [],
    recorder,
    operatorDiagnostics: recorderDiagnostics,
    dtype: 'f32',
    phase: 'decode',
  });

  assert.equal(recorderDiagnostics.emitter.length, 1);
  assert.equal(recorderDiagnostics.emitter.getTimeline()[0].capture.sample, null);

  await recorder.submitAndWait();

  assert.deepEqual(
    recorderDiagnostics.emitter.getTimeline()[0].capture.sample,
    [1.5, 2.5]
  );
} finally {
  trace.embed = originalEmbed;
  trace.attn = originalAttn;
  destroyBufferPool();
  setTrace(false);
  setDevice(null);
}

console.log('probes-readback.test: ok');
