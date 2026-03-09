import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

if (!globalThis.GPUBufferUsage) {
  globalThis.GPUBufferUsage = {
    MAP_READ: 1 << 0,
    MAP_WRITE: 1 << 1,
    COPY_SRC: 1 << 2,
    COPY_DST: 1 << 3,
    INDEX: 1 << 4,
    VERTEX: 1 << 5,
    UNIFORM: 1 << 6,
    STORAGE: 1 << 7,
    INDIRECT: 1 << 8,
    QUERY_RESOLVE: 1 << 9,
  };
}
if (!globalThis.GPUMapMode) {
  globalThis.GPUMapMode = {
    READ: 1 << 0,
    WRITE: 1 << 1,
  };
}

const { resolveKernelPathState } = await import('../../src/inference/pipelines/text/model-load.js');
const NO_SUBGROUP_CAPABILITIES = Object.freeze({
  hasSubgroups: false,
  hasF16: false,
});
const FULL_CAPABILITIES = Object.freeze({
  hasSubgroups: true,
  hasF16: true,
});

function createRuntimeConfig(kernelPath) {
  return {
    inference: {
      compute: { activationDtype: 'f32' },
      kvcache: { kvDtype: 'f32' },
      session: {
        compute: {
          defaults: {
            outputDtype: 'f32',
          },
        },
      },
      kernelPath,
      kernelPathSource: 'execution-v0',
      kernelPathPolicy: {
        mode: 'capability-aware',
        sourceScope: ['execution-v0'],
        onIncompatible: 'remap',
      },
    },
  };
}

const modelConfig = { kernelPath: null };
const manifest = { modelId: 'model-load-execution-v0-test', inference: {} };

{
  const runtimeConfig = {
    inference: {
      compute: { activationDtype: 'f16' },
      kvcache: { kvDtype: 'f16' },
      session: {
        compute: {
          defaults: {
            outputDtype: 'f16',
          },
        },
      },
      kernelPath: 'gemma2-q4k-dequant-f32a',
      kernelPathPolicy: {
        mode: 'capability-aware',
        sourceScope: ['config'],
        onIncompatible: 'remap',
      },
    },
  };
  assert.throws(
    () => resolveKernelPathState({
      manifest,
      runtimeConfig,
      modelConfig,
      kernelCapabilities: FULL_CAPABILITIES,
    }),
    /requires explicit matching runtime dtypes/
  );
}

{
  const runtimeConfig = createRuntimeConfig({
    id: 'inline-execution-v0-subgroup',
    name: 'Inline execution v0 subgroup',
    activationDtype: 'f32',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main' },
      ],
    },
  });

  assert.throws(
    () => resolveKernelPathState({
      manifest,
      runtimeConfig,
      modelConfig,
      kernelCapabilities: NO_SUBGROUP_CAPABILITIES,
    }),
    /ExecutionV0.*unsupported GPU features.*matmul_gemv_subgroup\.wgsl#main_vec4/
  );
}

{
  const runtimeConfig = createRuntimeConfig({
    id: 'inline-execution-v0-portable',
    name: 'Inline execution v0 portable',
    activationDtype: 'f32',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f32.wgsl', entry: 'main' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f32.wgsl', entry: 'main' },
      ],
    },
  });

  const resolved = resolveKernelPathState({
    manifest,
    runtimeConfig,
    modelConfig,
    kernelCapabilities: NO_SUBGROUP_CAPABILITIES,
  });
  assert.equal(resolved.kernelPathSource, 'execution-v0');
  assert.equal(resolved.resolvedKernelPath?.id, 'inline-execution-v0-portable');
}

{
  const runtimeConfig = {
    inference: {
      compute: { activationDtype: 'f32' },
      kvcache: { kvDtype: 'f32' },
    },
  };
  const lfm2Manifest = {
    modelId: 'lfm2-kernelpath-dtype-mismatch-test',
    quantizationInfo: {
      compute: 'f32',
    },
    inference: {
      presetId: 'lfm2',
    },
  };
  const lfm2ModelConfig = {
    kernelPath: 'gemma3-q4k-dequant-f16a-online',
  };

  assert.throws(
    () => resolveKernelPathState({
      manifest: lfm2Manifest,
      runtimeConfig,
      modelConfig: lfm2ModelConfig,
      kernelCapabilities: FULL_CAPABILITIES,
    }),
    /Manifest kernel path dtype mismatch/
  );
}

{
  const runtimeConfig = {
    inference: {
      compute: { activationDtype: 'f16' },
      kvcache: { kvDtype: 'f16' },
      session: {
        compute: {
          defaults: {
            outputDtype: 'f16',
          },
        },
      },
    },
  };
  const manifestKernelPath = {
    modelId: 'manifest-kernelpath-runtime-defaults-test',
    quantizationInfo: {
      compute: 'f32',
    },
    inference: {},
  };
  const manifestModelConfig = {
    kernelPath: 'gemma2-q4k-dequant-f32a',
  };

  const resolved = resolveKernelPathState({
    manifest: manifestKernelPath,
    runtimeConfig,
    modelConfig: manifestModelConfig,
    kernelCapabilities: FULL_CAPABILITIES,
  });
  assert.equal(resolved.resolvedKernelPath?.id, 'gemma2-q4k-dequant-f32a-nosubgroups');
  assert.equal(resolved.runtimeConfig.inference.compute.activationDtype, 'f32');
  assert.equal(resolved.runtimeConfig.inference.kvcache.kvDtype, 'f32');
  assert.equal(
    resolved.runtimeConfig.inference.session.compute.defaults.outputDtype,
    'f32'
  );
}

console.log('model-load-execution-v0-kernel-path.test: ok');
