import assert from 'node:assert/strict';

const { resolveKernelPath } = await import('../../src/config/kernel-path-loader.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { resolveQ4KConfig } = await import('../../src/inference/pipelines/text/init.js');

function createFakeDevice() {
  const features = new Set(['shader-f16', 'subgroups']);
  return {
    queue: {
      submit() {},
    },
    features,
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    createBuffer() {
      return {
        __dopplerFakeGPUBuffer: true,
        size: 0,
        usage: 0,
        destroy() {},
      };
    },
    createBindGroup() {
      return {};
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
    createShaderModule() {
      return {};
    },
    lost: new Promise(() => {}),
  };
}

const manifest = {
  modelId: 'unit-gemma3-q4k',
  quantization: 'Q4_K_M',
  quantizationInfo: {
    layout: 'row',
  },
};

const defaultPath = resolveKernelPath('gemma3-q4k-dequant-f32a-online');
const f32WeightPath = resolveKernelPath('gemma3-q4k-dequant-f32w-f32a-online');

setDevice(createFakeDevice(), { platformConfig: null });

try {
  assert.deepEqual(resolveQ4KConfig(manifest, defaultPath, 'config', false), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: false,
  });

  assert.deepEqual(resolveQ4KConfig(manifest, defaultPath, 'config', true), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: true,
  });

  assert.deepEqual(resolveQ4KConfig(manifest, f32WeightPath, 'config', false), {
    useFusedQ4K: false,
    q4kLayout: 'row',
    keepF32Weights: true,
  });
} finally {
  setDevice(null, { platformConfig: null });
}

console.log('q4k-loader-kernel-path-contract.test: ok');
