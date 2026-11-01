import assert from 'node:assert/strict';
import { ERROR_CODES, createDopplerError } from '../../src/errors/doppler-error.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { validateCallTimeOptions, validateRuntimeConfig, validateUnsupportedSessionOptions } from '../../src/config/param-validator.js';
import { createDopplerProvider } from '../../src/client/provider.js';

const cause = new Error('underlying device refusal');
const coded = createDopplerError(ERROR_CODES.GPU_DEVICE_FAILED, 'device failed', { cause });
assert.equal(coded.code, ERROR_CODES.GPU_DEVICE_FAILED);
assert.equal(coded.cause, cause);
assert.match(coded.message, /device failed/);
assert.throws(() => createDopplerError('unknown', 'not registered'), /unknown error code/);

for (const [action, message] of [
  [() => resolveRuntimeConfig([]), /runtime overrides must be an object/],
  [() => resolveRuntimeConfig({ debug: {} }), /runtime.debug is removed/],
  [() => resolveRuntimeConfig({ inference: { kernelPathPolicy: { mode: 'typo' } } }), /mode must be/],
  [() => validateCallTimeOptions({ slidingWindow: 5 }), /Cannot override at call-time/],
  [() => validateUnsupportedSessionOptions({ useOrtFlashPrefillAttention: true }), /unsupported/],
  [() => validateRuntimeConfig({ shared: { ecosystem: { schemaVersion: 7 } } }), /schemaVersion/],
  [() => createDopplerProvider(null), /requires a config object/],
  [() => createDopplerProvider({ policy: { mode: 'typo' } }), /Invalid policy mode/],
  [() => createDopplerProvider({ policy: { mode: 'local-only' } }), /local.model or local.handle/],
  [() => createDopplerProvider({ policy: { mode: 'cloud-only' }, fallback: {} }), /fallback.provider/],
]) {
  assert.throws(action, error => {
    assert.equal(error.code, ERROR_CODES.CONFIG_INVALID);
    assert.match(error.message, message);
    return true;
  });
}
assert.throws(() => resolveRuntimeConfig({ inference: { kernelPathPolicy: { mode: 'typo' } } }), error => {
  assert(error.cause instanceof Error);
  assert.match(error.cause.message, /DopplerConfigError:/);
  return true;
});
const programmerError = new Error('getter failure');
assert.throws(() => resolveRuntimeConfig({ get inference() { throw programmerError; } }), error => error === programmerError);
console.log('public error contract: stable config/device codes, messages and cause preservation passed');

const { normalizeToToolingCommandError } = await import('../../src/tooling/command-envelope.js');
const configError = createDopplerError(ERROR_CODES.CONFIG_INVALID, 'bad public config', { cause });
const envelopeError = normalizeToToolingCommandError(configError, { surface: 'node' });
assert.equal(envelopeError.code, ERROR_CODES.CONFIG_INVALID);
assert.equal(envelopeError.cause, configError);
assert.equal(envelopeError.cause.cause, cause);
assert.equal(envelopeError.message, configError.message);
