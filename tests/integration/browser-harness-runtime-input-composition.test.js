import assert from 'node:assert/strict';

import {
  applyRuntimeForRun,
  runBrowserManifest,
} from '../../src/inference/browser-harness.js';
import { resolveRuntime } from '../../src/inference/browser-harness-runtime-helpers.js';
import { parseRuntimeOverridesFromURL } from '../../src/inference/test-harness.js';
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
    ['https://example.test/presets/profiles/verbose-trace.json', {
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
    workload: 'inference',
    modelId: 'gemma3-270m',
    configChain: ['https://example.test/runtime/base.json'],
    runtimeProfile: 'profiles/verbose-trace',
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
  assert.equal(runtimeConfig.shared.harness.mode, 'verify');
  assert.equal(runtimeConfig.shared.harness.workload, 'inference');
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
          workload: 'inference',
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
    workload: 'inference',
    modelId: 'gemma3-270m',
    configChain: ['https://example.test/runtime/kernel-path-base.json'],
    runtimeConfig: {
      inference: {
        kernelPath: null,
      },
    },
  });

  assert.equal(getRuntimeConfig().shared.tooling.intent, 'verify');
  assert.equal(getRuntimeConfig().shared.harness.mode, 'verify');
  assert.equal(getRuntimeConfig().shared.harness.workload, 'inference');
  assert.equal(getRuntimeConfig().shared.harness.modelId, 'gemma3-270m');
  assert.equal(getRuntimeConfig().inference.kernelPath, null);

  setRuntimeConfig({
    shared: {
      tooling: {
        carried: true,
      },
    },
    inference: {
      kernelPath: 'gemma3-q4k-dequant-f16a-online',
      batching: {
        maxTokens: 5,
      },
    },
  });

  const resolvedRuntime = resolveRuntime({});
  assert.equal(resolvedRuntime.runtimeConfig.shared.tooling.carried, true);
  assert.equal(resolvedRuntime.runtimeConfig.inference.kernelPath, 'gemma3-q4k-dequant-f16a-online');
  assert.equal(resolvedRuntime.runtimeConfig.inference.batching.maxTokens, 5);

  const malformedRuntimeConfig = new URLSearchParams();
  malformedRuntimeConfig.set('runtimeConfig', '{');
  assert.throws(
    () => parseRuntimeOverridesFromURL(malformedRuntimeConfig),
    /Failed to parse runtimeConfig URL parameter/
  );

  const nonObjectRuntimeConfig = new URLSearchParams();
  nonObjectRuntimeConfig.set('runtimeConfig', '123');
  assert.throws(
    () => parseRuntimeOverridesFromURL(nonObjectRuntimeConfig),
    /runtimeConfig must be a JSON object/
  );

  const nonStringConfigChain = new URLSearchParams();
  nonStringConfigChain.set('configChain', '["debug",1]');
  assert.throws(
    () => parseRuntimeOverridesFromURL(nonStringConfigChain),
    /configChain must be an array of non-empty strings/
  );

  const malformedConfigChain = new URLSearchParams();
  malformedConfigChain.set('configChain', '[');
  assert.throws(
    () => parseRuntimeOverridesFromURL(malformedConfigChain),
    /Failed to parse configChain URL parameter/
  );
} finally {
  globalThis.fetch = originalFetch;
  setRuntimeConfig(null);
}

console.log('browser-harness-runtime-input-composition.test: ok');
