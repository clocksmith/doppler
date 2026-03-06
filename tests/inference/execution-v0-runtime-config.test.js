import assert from 'node:assert/strict';
import { applyExecutionV0RuntimeConfig } from '../../src/inference/pipelines/text/execution-v0.js';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { buildKernelRefFromKernelEntry } from '../../src/config/kernels/kernel-ref.js';

function kernelRef(kernel, entry = 'main') {
  return buildKernelRefFromKernelEntry(kernel, entry);
}

{
  const runtimeConfig = {
    inference: {
      compute: { activationDtype: 'f16' },
      modelOverrides: { attention: { causal: false } },
    },
  };
  const manifest = {
    modelId: 'runtime-merge-model',
    architecture: { numLayers: 2 },
    inference: {
      schema: 'doppler.execution/v0',
      model: {
        attention: {
          causal: true,
          slidingWindow: 1024,
        },
      },
      sessionDefaults: {
        compute: {
          defaults: {
            activationDtype: 'f16',
            mathDtype: 'f16',
            accumDtype: 'f32',
            outputDtype: 'f16',
          },
          kernelProfiles: [
            { kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main') },
          ],
        },
        kvcache: {
          kvDtype: 'f16',
        },
        decodeLoop: null,
      },
      execution: {
        steps: [
          {
            id: 'attn',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
          },
        ],
        policies: {
          precisionPrecedence: 'step_then_kernel_profile_then_session_default',
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
      },
    },
  };

  const resolved = applyExecutionV0RuntimeConfig({ runtimeConfig, manifest });
  assert.ok(resolved.executionV0State);
  assert.equal(resolved.runtimeConfig.inference.compute.activationDtype, 'f16');
  assert.equal(
    resolved.runtimeConfig.inference.modelOverrides.attention.causal,
    false
  );
  assert.equal(
    resolved.runtimeConfig.inference.modelOverrides.attention.slidingWindow,
    1024
  );
  assert.equal(
    resolved.runtimeConfig.inference.kernelPath.id,
    'runtime-merge-model-execution-v0'
  );
  assert.equal(
    resolved.runtimeConfig.inference.kernelPathSource,
    'execution-v0'
  );
  assert.equal(
    resolved.executionV0State.resolvedSources.steps.attn['precision.inputDtype'].source,
    'derived'
  );
}

{
  const runtimeConfig = { inference: { compute: { activationDtype: 'f16' } } };
  const manifest = { inference: { attention: { causal: true } } };
  const resolved = applyExecutionV0RuntimeConfig({ runtimeConfig, manifest });
  assert.equal(resolved.executionV0State, null);
  assert.equal(resolved.runtimeConfig, runtimeConfig);
}

{
  const runtimeConfig = createDopplerConfig({}).runtime;
  const manifest = {
    modelId: 'execution-v0-kv-manifest',
    architecture: { numLayers: 1 },
    inference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: {
        compute: {
          defaults: {
            activationDtype: 'f32',
            mathDtype: 'f32',
            accumDtype: 'f32',
            outputDtype: 'f32',
          },
          kernelProfiles: [
            { kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main') },
          ],
        },
        kvcache: {
          kvDtype: 'f16',
        },
      },
      execution: {
        steps: [
          {
            id: 'prefill-attn',
            phase: 'prefill',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16kv.wgsl',
            entry: 'main',
            kvIO: {
              readDtype: 'f16',
              writeDtype: 'f16',
            },
            kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
          },
        ],
        policies: {
          precisionPrecedence: 'step_then_kernel_profile_then_session_default',
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
      },
    },
  };

  const resolved = applyExecutionV0RuntimeConfig({ runtimeConfig, manifest });
  assert.ok(resolved.executionV0State);
  assert.equal(resolved.runtimeConfig.inference.kvcache.kvDtype, 'f16');
  assert.equal(resolved.runtimeConfig.inference.kernelPath.kvDtype, 'f16');
}

{
  assert.throws(
    () => applyExecutionV0RuntimeConfig({
      runtimeConfig: { inference: {} },
      manifest: {
        inference: {
          schema: 'bad-schema',
          execution: {
            steps: [
              {
                id: 'attn',
                phase: 'both',
                section: 'layer',
                op: 'attention',
                src: 'state',
                dst: 'state',
                layers: 'all',
                kernel: 'attention_streaming_f16.wgsl',
              },
            ],
            policies: {
              precisionPrecedence: 'step_then_kernel_profile_then_session_default',
              unsupportedPrecision: 'error',
              dtypeTransition: 'require_cast_step',
              unresolvedKernel: 'error',
            },
          },
        },
      },
    }),
    /manifest\.inference\.schema must be "doppler\.execution\/v0"/
  );
}

{
  const config = createDopplerConfig({
    runtime: {
      inference: {
        session: {
          compute: {
            defaults: {
              activationDtype: 'f32',
            },
          },
          decodeLoop: {
            batchSize: 8,
            stopCheckMode: 'batch',
          },
        },
        executionPatch: {
          set: [{ id: 'attn', entry: 'main' }],
        },
      },
    },
  });

  assert.equal(config.runtime.inference.session.compute.defaults.activationDtype, 'f32');
  assert.equal(config.runtime.inference.session.decodeLoop.batchSize, 8);
  assert.equal(config.runtime.inference.executionPatch.set.length, 1);
  assert.equal(config.runtime.inference.executionPatch.remove.length, 0);
  assert.equal(config.runtime.inference.executionPatch.add.length, 0);
}
