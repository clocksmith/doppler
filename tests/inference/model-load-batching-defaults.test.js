import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { applyModelBatchingRuntimeDefaults } = await import('../../src/inference/pipelines/text/model-load.js');
const { createDopplerConfig } = await import('../../src/config/schema/index.js');

function createRuntimeConfig(session = {}) {
  return {
    inference: {
      session: structuredClone(session),
    },
  };
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-default-batching-test',
    modelType: 'transformer',
    inference: {},
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);

  assert.strictEqual(nextRuntime, runtimeConfig);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-manifest-session-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 16,
          stopCheckMode: 'batch',
          readbackInterval: 4,
          readbackMode: 'sequential',
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.deepEqual(nextRuntime.inference.session.decodeLoop, {
    batchSize: 16,
    stopCheckMode: 'batch',
    readbackInterval: 4,
    readbackMode: 'sequential',
    submitLatencyThresholdMs: null,
  });
  assert.equal(nextRuntime.inference.generation, undefined);
}

{
  const explicitRuntimeOverrides = {
    inference: {
      compute: {
        activationDtype: 'f32',
      },
    },
  };
  const runtimeConfig = createDopplerConfig({
    runtime: explicitRuntimeOverrides,
  }).runtime;
  const manifest = {
    modelId: 'manifest-session-kvtype-preserved-test',
    modelType: 'transformer',
    inference: {
      session: {
        kvcache: {
          kvDtype: 'f32',
          layout: 'contiguous',
          pageSize: 256,
          tiering: {
            mode: 'off',
          },
          quantization: {
            mode: 'none',
          },
        },
        decodeLoop: {
          batchSize: 16,
          stopCheckMode: 'batch',
          readbackInterval: 4,
          readbackMode: 'sequential',
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null, explicitRuntimeOverrides);
  assert.equal(nextRuntime.inference.session.kvcache.kvDtype, 'f32');
}

{
  const runtimeConfig = createRuntimeConfig({
    decodeLoop: {
      batchSize: 4,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      disableCommandBatching: true,
    },
  });
  const manifest = {
    modelId: 'qwen3-runtime-session-override-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 4,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          readbackMode: 'sequential',
          disableCommandBatching: false,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.session.decodeLoop.disableCommandBatching, true);
  assert.equal(nextRuntime.inference.generation, undefined);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-manifest-ring-defaults-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
          ringTokens: 4,
          ringStop: 2,
          ringStaging: 3,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.deepEqual(nextRuntime.inference.session.decodeLoop, {
    batchSize: 8,
    stopCheckMode: 'batch',
    readbackInterval: 8,
    readbackMode: 'sequential',
    submitLatencyThresholdMs: null,
    ringTokens: 4,
    ringStop: 2,
    ringStaging: 3,
  });
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-manifest-ring-disable-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
          ringTokens: null,
          ringStop: null,
          ringStaging: null,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.deepEqual(nextRuntime.inference.session.decodeLoop, {
    batchSize: 8,
    stopCheckMode: 'batch',
    readbackInterval: 8,
    readbackMode: 'sequential',
    submitLatencyThresholdMs: null,
    ringTokens: null,
    ringStop: null,
    ringStaging: null,
  });
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'non-lfm2-batching-test',
    modelType: 'transformer',
    inference: {},
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.strictEqual(nextRuntime, runtimeConfig);
}

{
  const runtimeConfig = createRuntimeConfig({
    decodeLoop: {
      batchSize: 16,
      stopCheckMode: 'batch',
      readbackInterval: 2,
      readbackMode: 'sequential',
    },
  });
  const manifest = {
    modelId: 'lfm2-runtime-override-wins-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.session.decodeLoop.batchSize, 16);
  assert.equal(nextRuntime.inference.session.decodeLoop.readbackInterval, 2);
}

{
  const runtimeConfig = createRuntimeConfig({
    decodeLoop: {
      batchSize: 8,
      stopCheckMode: 'batch',
      readbackInterval: 8,
      readbackMode: 'sequential',
      disableCommandBatching: true,
    },
  });
  const manifest = {
    modelId: 'lfm2-disable-command-batching-override-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
          disableCommandBatching: false,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.equal(nextRuntime.inference.session.decodeLoop.disableCommandBatching, true);
  assert.equal(nextRuntime.inference.generation, undefined);
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-invalid-decode-loop-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          readbackInterval: 8,
          readbackMode: 'sequential',
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /inference\.session\.decodeLoop\.stopCheckMode must be "batch" or "per-token"/
  );
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-invalid-ring-tokens-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
          ringTokens: 0,
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /inference\.session\.decodeLoop\.ringTokens must be a positive integer or null/
  );
}

{
  const runtimeConfig = createRuntimeConfig();
  const manifest = {
    modelId: 'lfm2-invalid-disable-command-batching-type-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
          disableCommandBatching: 'yes',
        },
      },
    },
  };

  assert.throws(
    () => applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null),
    /inference\.session\.decodeLoop\.disableCommandBatching must be a boolean when provided/
  );
}

{
  const runtimeConfig = createRuntimeConfig({
    decodeLoop: {
      batchSize: 8,
      stopCheckMode: 'batch',
      readbackInterval: 4,
      readbackMode: 'sequential',
      disableCommandBatching: true,
    },
  });
  const manifest = {
    modelId: 'lfm2-full-runtime-override-test',
    modelType: 'transformer',
    inference: {
      session: {
        decodeLoop: {
          batchSize: 16,
          stopCheckMode: 'batch',
          readbackInterval: 4,
          readbackMode: 'sequential',
          disableCommandBatching: false,
        },
      },
    },
  };

  const nextRuntime = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
  assert.notStrictEqual(nextRuntime, runtimeConfig);
  assert.equal(nextRuntime.inference.session.decodeLoop.batchSize, 8);
  assert.equal(nextRuntime.inference.session.decodeLoop.disableCommandBatching, true);
  assert.equal(nextRuntime.inference.generation, undefined);
}

console.log('model-load-batching-defaults.test: ok');
