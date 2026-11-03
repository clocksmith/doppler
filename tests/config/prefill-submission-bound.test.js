import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { validateRuntimeConfig } from '../../src/config/param-validator.js';
import { resolvePrefillMaxInFlightChunks } from '../../src/inference/pipelines/text/generator/prefill-policy.js';

const defaults = createDopplerConfig().runtime;
assert.doesNotThrow(() => resolvePrefillMaxInFlightChunks({ runtimeConfig: defaults }));

const runtimeConfig = createDopplerConfig({
  runtime: { inference: { session: { prefillMaxInFlightChunks: 3 } } },
}).runtime;
validateRuntimeConfig(runtimeConfig);
assert.equal(resolvePrefillMaxInFlightChunks({ runtimeConfig }), 3, 'Explicit runtime scheduling reaches its consumer');

for (const invalid of [0, -1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1, null]) {
  const config = createDopplerConfig({
    runtime: { inference: { session: { prefillMaxInFlightChunks: invalid } } },
  }).runtime;
  assert.throws(() => validateRuntimeConfig(config), error =>
    error.code === 'DOPPLER_CONFIG_INVALID' && /prefillMaxInFlightChunks/.test(error.message));
}
console.log('prefill-submission-bound: passed');
