import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const { applyModelBatchingRuntimeDefaults } = await import('../../src/inference/pipelines/text/model-load.js');

function createRuntimeConfig() {
  return structuredClone(createDopplerConfig().runtime);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-default-batching-test',
    modelType: 'transformer',
    inference: { presetId: 'lfm2' },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);

  assert.strictEqual(nextRuntime, runtimeConfig);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-manifest-session-defaults-test',
    modelType: 'transformer',
    inference: {
      presetId: 'lfm2',
      sessionDefaults: {
        decodeLoop: {
          batchSize: 16,
          stopCheckMode: 'batch',
          readbackInterval: 4,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.batchSize, 16);
  assert.equal(nextRuntime.inference.batching.readbackInterval, 4);
  assert.equal(nextRuntime.inference.batching.ringTokens, runtimeConfig.inference.batching.ringTokens);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'qwen3-manifest-disable-recorder-default-test',
    modelType: 'transformer',
    inference: {
      presetId: 'qwen3',
      sessionDefaults: {
        decodeLoop: {
          batchSize: 4,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          disableCommandBatching: true,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.generation.disableCommandBatching, true);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-manifest-ring-defaults-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          ringTokens: 4,
          ringStop: 2,
          ringStaging: 3,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.batchSize, 8);
  assert.equal(nextRuntime.inference.batching.readbackInterval, 8);
  assert.equal(nextRuntime.inference.batching.ringTokens, 4);
  assert.equal(nextRuntime.inference.batching.ringStop, 2);
  assert.equal(nextRuntime.inference.batching.ringStaging, 3);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-manifest-ring-disable-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          ringTokens: null,
          ringStop: null,
          ringStaging: null,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.ringTokens, null);
  assert.equal(nextRuntime.inference.batching.ringStop, null);
  assert.equal(nextRuntime.inference.batching.ringStaging, null);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'non-lfm2-batching-test',
    modelType: 'transformer',
    inference: { presetId: 'gemma3' },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.strictEqual(nextRuntime, runtimeConfig);
}

{
  const runtimeConfig = createRuntimeConfig();
  runtimeConfig.inference.batching.batchSize = 16;
  const manifest = {
    modelId: 'lfm2-conflicting-batching-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /Manifest decodeLoop defaults cannot be merged after runtime batching overrides were already resolved/
  );
}

{
  const runtimeConfig = createRuntimeConfig();
  runtimeConfig.inference.batching.batchSize = 16;
  const manifest = {
    modelId: 'execution-v0-manifest-batching-defaults-test',
    modelType: 'transformer',
    inference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.strictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.batchSize, 16);
}

{
  const runtimeConfig = createRuntimeConfig();
  runtimeConfig.inference.generation.disableCommandBatching = true;
  const manifest = {
    modelId: 'lfm2-conflicting-disable-command-batching-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          disableCommandBatching: false,
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /Manifest decodeLoop\.disableCommandBatching conflicts with runtime\.inference\.generation\.disableCommandBatching/
  );
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-invalid-decode-loop-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          readbackInterval: 8,
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /inference\.sessionDefaults\.decodeLoop\.stopCheckMode must be "batch" or "per-token"/
  );
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-invalid-ring-tokens-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          ringTokens: 0,
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /inference\.sessionDefaults\.decodeLoop\.ringTokens must be a positive integer or null/
  );
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-invalid-disable-command-batching-type-test',
    modelType: 'transformer',
    inference: {
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          disableCommandBatching: 'yes',
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /inference\.sessionDefaults\.decodeLoop\.disableCommandBatching must be a boolean when provided/
  );
}

console.log('model-load-batching-defaults.test: ok');
