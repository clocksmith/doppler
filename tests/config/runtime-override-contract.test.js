import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { mergeKernelPathPolicy } from '../../src/config/merge-helpers.js';
import { validateModelOverrides, validateRuntimeConfig } from '../../src/config/param-validator.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

const originalRuntime = structuredClone(getRuntimeConfig());

try {
  assert.throws(
    () => setRuntimeConfig('invalid'),
    /runtime overrides must be an object when provided/
  );

  // inference.session.maxNewTokens is silently ignored by the config merge but is
  // never wired to generation limits (which now read inference.generation.maxTokens).
  // It must be rejected fast so callers don't get unbounded 300s+ runs.
  assert.throws(
    () => setRuntimeConfig({
      inference: {
        session: {
          maxNewTokens: 30,
        },
      },
    }),
    /inference\.session\.maxNewTokens.*inference\.generation\.maxTokens/,
    'session.maxNewTokens must throw with redirect to generation.maxTokens'
  );

  assert.throws(
    () => setRuntimeConfig({
      inference: {
        generation: {
          disableCommandBatching: true,
        },
      },
    }),
    /inference\.generation\.disableCommandBatching.*inference\.session\.decodeLoop\.disableCommandBatching/,
    'generation.disableCommandBatching must throw with redirect to session.decodeLoop.disableCommandBatching'
  );

  assert.throws(
    () => setRuntimeConfig({
      inference: {
        batching: null,
      },
    }),
    /runtime\.inference\.batching must not be null/
  );

  assert.throws(
    () => setRuntimeConfig({
      inference: {
        modelOverrides: [],
      },
    }),
    /runtime\.inference\.modelOverrides must be an object when provided/
  );

  assert.throws(
    () => setRuntimeConfig({
      inference: {
        modelOverrides: {
          inference: {
            rope: {
              ropeTheta: 1,
            },
          },
        },
      },
    }),
    /runtime\.inference\.modelOverrides may only override vision_config, audio_config[\s\S]*inference\.rope\.ropeTheta/
  );

  assert.doesNotThrow(() => validateModelOverrides({
    vision_config: {
      image_size: null,
    },
    audio_config: {
      sample_rate: 16000,
    },
  }));

  assert.throws(
    () => setRuntimeConfig({
      inference: {
        kernelPathPolicy: null,
      },
    }),
    /runtime\.inference\.kernelPathPolicy must not be null/
  );

  assert.throws(
    () => setRuntimeConfig({
      inference: {
        kernelPath: 'gemma3-q4k-dequant-f32a-online',
      },
    }),
    /runtime\.inference\.kernelPath no longer accepts string registry IDs/
  );

  assert.throws(
    () => createDopplerConfig({
      runtime: {
        inference: {
          kernelPath: 7,
        },
      },
    }),
    /runtime\.inference\.kernelPath must be an inline kernel path object or null/
  );

  assert.throws(
    () => createDopplerConfig({
      runtime: {
        inference: {
          batching: null,
        },
      },
    }),
    /runtime\.inference\.batching must not be null/
  );

  assert.throws(
    () => mergeKernelPathPolicy(
      {
        mode: 'locked',
        sourceScope: ['manifest'],
        onIncompatible: 'error',
      },
      null
    ),
    /runtime\.inference\.kernelPathPolicy must not be null/
  );

  assert.throws(
    () => createDopplerConfig({
      runtime: {
        inference: {
          kernelPathPolicy: {
            sourceScope: ['manifest'],
            allowSources: ['config'],
          },
        },
      },
    }),
    /sourceScope and runtime\.inference\.kernelPathPolicy\.allowSources must match exactly/
  );

  const runtimeConfig = createDopplerConfig().runtime;
  assert.equal(
    runtimeConfig.inference.generation.maxTokens,
    256,
    'Default runtime config must own the text generation max token budget.'
  );
  assert.equal(
    runtimeConfig.inference.generation.multimodalMaxTokens,
    512,
    'Default runtime config must own the multimodal generation max token budget.'
  );
  assert.equal(
    runtimeConfig.inference.kvcache.quantization.bitWidth,
    4,
    'Default runtime config must own quantized KV bit width.'
  );
  assert.equal(
    runtimeConfig.inference.kvcache.tiering.compression.bitWidth,
    4,
    'Default runtime config must own tiered KV compression bit width.'
  );
  assert.equal(
    runtimeConfig.shared.debug.profiler.maxHistoryLabels,
    1024,
    'Default runtime config must own profiler label history length.'
  );
  assert.deepEqual(
    runtimeConfig.inference.kernelPathPolicy,
    {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      allowSources: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    },
    'Default runtime config must preserve capability-aware remap policy for execution-v1 manifests.'
  );
  runtimeConfig.inference.kernelPathPolicy = {
    mode: 'locked',
    sourceScope: ['manifest'],
    allowSources: ['config'],
    onIncompatible: 'error',
  };
  assert.throws(
    () => validateRuntimeConfig(runtimeConfig),
    /sourceScope and runtime\.inference\.kernelPathPolicy\.allowSources must match exactly/
  );

  const legacyGenerationDisableCommandBatchingRuntime = createDopplerConfig().runtime;
  legacyGenerationDisableCommandBatchingRuntime.inference.generation.disableCommandBatching = true;
  assert.throws(
    () => validateRuntimeConfig(legacyGenerationDisableCommandBatchingRuntime),
    /runtime\.inference\.generation\.disableCommandBatching.*runtime\.inference\.session\.decodeLoop\.disableCommandBatching/
  );

  const invalidGenerationMaxTokensRuntime = createDopplerConfig().runtime;
  invalidGenerationMaxTokensRuntime.inference.generation.maxTokens = 0;
  assert.throws(
    () => validateRuntimeConfig(invalidGenerationMaxTokensRuntime),
    /runtime\.inference\.generation\.maxTokens/
  );

  const invalidMultimodalMaxTokensRuntime = createDopplerConfig().runtime;
  invalidMultimodalMaxTokensRuntime.inference.generation.multimodalMaxTokens = 0;
  assert.throws(
    () => validateRuntimeConfig(invalidMultimodalMaxTokensRuntime),
    /runtime\.inference\.generation\.multimodalMaxTokens/
  );

  const updatedRuntime = setRuntimeConfig({
    inference: {
      batching: {
        batchSize: 7,
      },
    },
  });
  assert.equal(updatedRuntime.inference.batching.batchSize, 7);

  const nullKernelPathRuntime = setRuntimeConfig({
    inference: {
      kernelPath: null,
    },
  });
  assert.equal(nullKernelPathRuntime.inference.kernelPath, null);

  const allowlistedModelOverrideRuntime = setRuntimeConfig({
    inference: {
      modelOverrides: {
        vision_config: {
          image_size: 384,
        },
      },
    },
  });
  assert.equal(
    allowlistedModelOverrideRuntime.inference.modelOverrides.vision_config.image_size,
    384
  );

  const nullScalarSchemaConfig = createDopplerConfig({
    runtime: {
      loading: {
        allowF32UpcastNonMatmul: null,
      },
      inference: {
        prompt: null,
        modelOverrides: null,
      },
    },
  });
  assert.equal(nullScalarSchemaConfig.runtime.loading.allowF32UpcastNonMatmul, null);
  assert.equal(nullScalarSchemaConfig.runtime.inference.prompt, null);
  assert.equal(nullScalarSchemaConfig.runtime.inference.modelOverrides, null);

  const nullSessionSubtreeConfig = createDopplerConfig({
    runtime: {
      inference: {
        session: {
          kvcache: null,
        },
      },
    },
  });
  assert.equal(nullSessionSubtreeConfig.runtime.inference.session.kvcache, null);

  const resetRuntime = setRuntimeConfig(null);
  assert.deepEqual(resetRuntime, createDopplerConfig().runtime);
} finally {
  setRuntimeConfig(originalRuntime);
}

console.log('runtime-override-contract.test: ok');
