import assert from 'node:assert/strict';

import {
  applyRuntimeForRun,
  runBrowserManifest,
} from '../../src/inference/browser-harness.js';
import {
  getRuntimeConfig,
  setRuntimeConfig,
} from '../../src/config/runtime.js';

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const originalFetch = globalThis.fetch;

try {
  const fetchPayloads = new Map([
    ['https://example.test/runtime/base.json', {
      runtime: {
        inference: {
          prompt: 'chain',
          batching: {
            maxTokens: 16,
            batchSize: 2,
            stopCheckMode: 'per-batch',
          },
        },
      },
    }],
    ['https://example.test/presets/modes/debug.json', {
      runtime: {
        inference: {
          prompt: 'preset',
          batching: {
            maxTokens: 12,
            batchSize: 4,
            stopCheckMode: 'per-token',
          },
        },
      },
    }],
    ['https://example.test/runtime/override.json', {
      runtime: {
        inference: {
          prompt: 'url',
          batching: {
            batchSize: 6,
            readbackInterval: 3,
          },
        },
      },
    }],
    ['https://example.test/runtime/kernel-path-base.json', {
      runtime: {
        inference: {
          kernelPath: 'gemma2-q4k-fused-f32a',
        },
      },
    }],
  ]);

  globalThis.fetch = async (url) => {
    const key = String(url);
    const payload = fetchPayloads.get(key);
    if (!payload) {
      return new Response('not found', { status: 404 });
    }
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: {
        'content-type': 'application/json',
      },
    });
  };

  setRuntimeConfig({
    shared: {
      tooling: {
        baseline: true,
      },
    },
  });

  await applyRuntimeForRun({
    command: 'verify',
    suite: 'inference',
    modelId: 'gemma3-270m',
    configChain: ['https://example.test/runtime/base.json'],
    runtimePreset: 'modes/debug',
    runtimeConfigUrl: 'https://example.test/runtime/override.json',
    runtimeConfig: {
      inference: {
        prompt: 'inline',
        batching: {
          maxTokens: 8,
        },
      },
    },
  }, {
    baseUrl: 'https://example.test/presets',
  });

  const runtimeConfig = getRuntimeConfig();
  assert.equal(runtimeConfig.shared.tooling.baseline, true);
  assert.equal(runtimeConfig.shared.tooling.intent, 'verify');
  assert.equal(runtimeConfig.shared.harness.mode, 'inference');
  assert.equal(runtimeConfig.shared.harness.modelId, 'gemma3-270m');
  assert.equal(runtimeConfig.inference.prompt, 'inline');
  assert.equal(runtimeConfig.inference.batching.maxTokens, 8);
  assert.equal(runtimeConfig.inference.batching.batchSize, 6);
  assert.equal(runtimeConfig.inference.batching.readbackInterval, 3);
  assert.equal(runtimeConfig.inference.batching.stopCheckMode, 'per-token');

  const baselineRuntime = {
    shared: {
      tooling: {
        baseline: 'manifest',
      },
    },
    inference: {
      prompt: 'baseline',
    },
  };
  setRuntimeConfig(clone(baselineRuntime));

  await assert.rejects(
    () => runBrowserManifest({
      runs: [
        {
          suite: 'inference',
          command: 'verify',
          runtimeConfig: {
            invalid: true,
          },
        },
      ],
    }),
    /runtimeConfig is missing runtime fields/
  );

  const restoredRuntime = getRuntimeConfig();
  assert.equal(restoredRuntime.shared.tooling.baseline, 'manifest');
  assert.equal(restoredRuntime.inference.prompt, 'baseline');

  setRuntimeConfig({
    inference: {
      kernelPath: 'baseline-kernel-path',
    },
  });

  await applyRuntimeForRun({
    command: 'verify',
    suite: 'inference',
    modelId: 'gemma3-270m',
    configChain: ['https://example.test/runtime/kernel-path-base.json'],
    runtimeConfig: {
      inference: {
        kernelPath: null,
      },
    },
  });

  assert.equal(getRuntimeConfig().shared.tooling.intent, 'verify');
  assert.equal(getRuntimeConfig().shared.harness.mode, 'inference');
  assert.equal(getRuntimeConfig().shared.harness.modelId, 'gemma3-270m');
  assert.equal(getRuntimeConfig().inference.kernelPath, null);
} finally {
  globalThis.fetch = originalFetch;
  setRuntimeConfig(null);
}

console.log('browser-harness-runtime-input-composition.test: ok');
