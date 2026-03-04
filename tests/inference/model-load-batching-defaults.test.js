import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const { applyModelBatchingRuntimeDefaults } = await import('../../src/inference/pipelines/text/model-load.js');

function createRuntimeConfig() {
  return createDopplerConfig().runtime;
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-default-batching-test',
    modelType: 'transformer',
    inference: { presetId: 'lfm2' },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, {
    numLayers: 16,
    hiddenSize: 2048,
  });

  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.batchSize, 8);
  assert.equal(nextRuntime.inference.batching.stopCheckMode, 'batch');
  assert.equal(nextRuntime.inference.batching.readbackInterval, 8);
  assert.equal(nextRuntime.inference.batching.maxTokens, runtimeConfig.inference.batching.maxTokens);
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
  runtimeConfig.shared.tooling.intent = 'calibrate';
  const manifest = {
    modelId: 'lfm2-calibrate-batching-test',
    modelType: 'transformer',
    inference: { presetId: 'lfm2' },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.batchSize, 8);
  assert.equal(nextRuntime.inference.batching.readbackInterval, 8);
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
  runtimeConfig.inference.batching.readbackInterval = 8;
  const manifest = {
    modelId: 'lfm2-custom-batching-test',
    modelType: 'transformer',
    inference: { presetId: 'lfm2' },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.strictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.batching.batchSize, 16);
  assert.equal(nextRuntime.inference.batching.readbackInterval, 8);
}

console.log('model-load-batching-defaults.test: ok');
