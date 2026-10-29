import assert from 'node:assert/strict';

import { kernelTrace } from '../../src/inference/pipelines/text/kernel-trace.js';
import { applyPipelineContexts, restorePipelineContexts } from '../../src/inference/pipelines/context.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';

// Partial options are a configuration error, not a cue to recreate schema defaults.
assert.throws(() => kernelTrace.enable({ layers: [] }), /requires resolved runtime\.shared\.debug\.kernelTrace/);
assert.throws(() => kernelTrace.enable(), /requires resolved runtime\.shared\.debug\.kernelTrace/);
assert.equal(kernelTrace.enabled, false);

// The pipeline context bridge enables tracing with the resolved config values.
{
  const runtimeConfig = { shared: { debug: { kernelTrace: { enabled: true, layers: [0, 1] } } } };
  const { enabled, ...expected } = resolveRuntimeConfig(runtimeConfig).shared.debug.kernelTrace;
  assert.equal(enabled, true);
  const target = {};
  applyPipelineContexts(target, { runtimeConfig });
  try {
    assert.equal(kernelTrace.enabled, true);
    assert.deepEqual(kernelTrace.captureState().options, expected);
  } finally {
    restorePipelineContexts(target);
    kernelTrace.disable();
  }
}

console.log('kernel-trace-options.test: ok');
