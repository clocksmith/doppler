import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { mergeKernelPathPolicy } from '../../src/config/merge-helpers.js';
import { validateRuntimeConfig } from '../../src/config/param-validator.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

const originalRuntime = structuredClone(getRuntimeConfig());

try {
  assert.throws(
    () => setRuntimeConfig('invalid'),
    /runtime overrides must be an object when provided/
  );

  // inference.session.maxNewTokens is silently ignored by the config merge but is
  // never wired to the decode loop (which reads inference.batching.maxTokens).
  // It must be rejected fast so callers don't get unbounded 300s+ runs.
  assert.throws(
    () => setRuntimeConfig({
      inference: {
        session: {
          maxNewTokens: 30,
        },
      },
    }),
    /inference\.session\.maxNewTokens.*inference\.batching\.maxTokens/,
    'session.maxNewTokens must throw with redirect to batching.maxTokens'
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
        kernelPathPolicy: null,
      },
    }),
    /runtime\.inference\.kernelPathPolicy must not be null/
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

  const resetRuntime = setRuntimeConfig(null);
  assert.deepEqual(resetRuntime, createDopplerConfig().runtime);
} finally {
  setRuntimeConfig(originalRuntime);
}

console.log('runtime-override-contract.test: ok');
