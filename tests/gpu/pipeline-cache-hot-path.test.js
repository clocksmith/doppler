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

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  let pipelineCount = 0;
  let pipelineLayoutCount = 0;
  let pipelineBindGroupLayoutCount = 0;

  const queue = {
    submit() {},
    writeBuffer() {},
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

  return {
    queue,
    lost: new Promise(() => {}),
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
    get pipelineCount() {
      return pipelineCount;
    },
    get pipelineLayoutCount() {
      return pipelineLayoutCount;
    },
    get pipelineBindGroupLayoutCount() {
      return pipelineBindGroupLayoutCount;
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createBindGroupLayout(descriptor) {
      return descriptor;
    },
    createPipelineLayout(descriptor) {
      pipelineLayoutCount += 1;
      return descriptor;
    },
    createShaderModule(descriptor) {
      return {
        descriptor,
        getCompilationInfo() {
          return Promise.resolve({ messages: [] });
        },
      };
    },
    createComputePipelineAsync(descriptor) {
      pipelineCount += 1;
      return Promise.resolve({
        descriptor,
        getBindGroupLayout(index) {
          pipelineBindGroupLayoutCount += 1;
          return { label: `auto-layout-${index}` };
        },
      });
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() {
          return {};
        },
      };
    },
    destroy() {},
  };
}

const { setDevice } = await import('../../src/gpu/device.js');
const { clearShaderCaches } = await import('../../src/gpu/kernels/shader-cache.js');
const {
  clearPipelineCaches,
  getCachedPipeline,
  getPipelineBindGroupLayout,
  getPipelineFast,
} = await import('../../src/gpu/kernels/pipeline-cache.js');

const device = createFakeDevice();

try {
  setDevice(device, { platformConfig: null });
  clearShaderCaches();
  clearPipelineCaches();

  const first = await getPipelineFast('scale', 'default', null, {
    A: 4,
    B: true,
  });
  assert.equal(device.pipelineCount, 1);

  const reordered = getCachedPipeline('scale', 'default', {
    B: true,
    A: 4,
  });
  assert.equal(reordered, first);

  const fast = await getPipelineFast('scale', 'default', null, {
    A: 4,
    B: true,
  });
  assert.equal(fast, first);
  assert.equal(device.pipelineCount, 1);

  const emptyConstants = await getPipelineFast('scale', 'default', null, {});
  assert.notEqual(emptyConstants, first);
  assert.equal(device.pipelineCount, 2);

  const noConstants = await getPipelineFast('scale', 'default');
  assert.notEqual(noConstants, emptyConstants);
  assert.equal(device.pipelineCount, 3);

  const autoLayoutA = getPipelineBindGroupLayout(first, 0);
  const autoLayoutB = getPipelineBindGroupLayout(first, 0);
  assert.equal(autoLayoutB, autoLayoutA);
  assert.equal(device.pipelineBindGroupLayoutCount, 1);
} finally {
  clearPipelineCaches();
  clearShaderCaches();
  setDevice(null, { platformConfig: null });
}

console.log('pipeline-cache-hot-path.test: ok');
