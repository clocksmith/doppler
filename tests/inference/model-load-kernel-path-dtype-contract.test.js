import assert from 'node:assert/strict';

const { resolveKernelPathState } = await import('../../src/inference/pipelines/text/model-load.js');

function createKernelPath(activationDtype = 'f32') {
  return {
    id: `inline-${activationDtype}`,
    name: `Inline ${activationDtype.toUpperCase()}`,
    activationDtype,
    outputDtype: activationDtype,
    kvDtype: activationDtype,
    decode: {
      steps: [
        { op: 'attention', kernel: 'attention_streaming_f16kv.wgsl' },
      ],
    },
    prefill: {
      steps: [
        { op: 'attention', kernel: 'attention_streaming_f16kv.wgsl' },
      ],
    },
  };
}

function createRuntimeConfig(dtype) {
  return {
    inference: {
      kernelPath: null,
      compute: {
        activationDtype: dtype,
      },
      session: {
        compute: {
          defaults: {
            activationDtype: dtype,
            outputDtype: dtype,
          },
        },
        kvcache: {
          kvDtype: dtype,
        },
      },
    },
  };
}

function createManifest(compute = 'f32') {
  return {
    modelId: 'model-load-kernel-path-dtype-contract',
    quantizationInfo: {
      compute,
    },
    inference: {},
  };
}

{
  assert.throws(
    () => resolveKernelPathState({
      manifest: createManifest('f32'),
      runtimeConfig: createRuntimeConfig('f16'),
      modelConfig: {
        kernelPath: createKernelPath('f32'),
      },
    }),
    /Runtime dtype auto-rewrites are not allowed/
  );
}

{
  const runtimeConfig = createRuntimeConfig('f32');
  const result = resolveKernelPathState({
    manifest: createManifest('f32'),
    runtimeConfig,
    modelConfig: {
      kernelPath: createKernelPath('f32'),
    },
  });

  assert.strictEqual(result.runtimeConfig, runtimeConfig);
  assert.equal(result.resolvedKernelPath?.activationDtype, 'f32');
  assert.equal(result.kernelPathSource, 'model');
}

console.log('model-load-kernel-path-dtype-contract.test: ok');
