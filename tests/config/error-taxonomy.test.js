import assert from 'node:assert/strict';
import { DopplerErrorBase, ERROR_CODES, createDopplerError } from '../../src/errors/doppler-error.js';
import { GenerationError, GENERATION_CONTRACT } from '../../src/config/generation-contract.js';
import { AbortError } from '../../src/inference/pipelines/text/contracts/abort-contract.js';
import { FinitenessError } from '../../src/inference/pipelines/text/policy/finiteness-guard-status.js';
import { QuotaExceededError } from '../../src/storage/quota.js';
import { ToolingCommandError, normalizeToToolingCommandError } from '../../src/tooling/command-envelope.js';
import { DopplerNodeWebGPUError } from '../../src/tooling/node-webgpu.js';
import { CapsuleServeError } from '../../src/cli/serve/capsule-http.js';
import { ServeRequestError } from '../../src/cli/doppler-serve.js';
import { CapsuleReleaseStateError } from '../../src/config/capsule/capsule-release-events.js';

const cause = new Error('original');
for (const error of [
  new GenerationError('invalidRequest', 'invalid', { cause }), new AbortError('stopped'),
  new FinitenessError('not finite'), new QuotaExceededError(1024, 32),
  new CapsuleReleaseStateError(cause, { sequence: 0, digest: null }),
]) {
  assert(error instanceof DopplerErrorBase);
  assert(error instanceof Error);
  assert.equal(error.name, error.constructor.name);
}
const generated = new GenerationError('invalidRequest', 'invalid', { cause });
assert.equal(generated.code, GENERATION_CONTRACT.errors.invalidRequest);
assert.equal(generated.cause, cause);
assert.equal(new AbortError().code, 'ABORT_ERR');
const quota = new QuotaExceededError(1024, 32);
assert.equal(quota.required, 1024); assert.equal(quota.available, 32); assert.equal(quota.shortfall, 992);
assert.equal(new CapsuleServeError('ORIGIN_DENIED', 'denied', 403).statusCode, 403);
const coded = createDopplerError(ERROR_CODES.GPU_OOM, 'allocation failed', { cause });
assert(coded instanceof DopplerErrorBase);
assert.equal(coded.name, 'Error');
assert.equal(coded.message, `[${ERROR_CODES.GPU_OOM}] allocation failed`);
assert.equal(coded.cause, cause);
assert.equal(coded.code, ERROR_CODES.GPU_OOM);
assert.equal(Object.isFrozen(ERROR_CODES), true);
assert.throws(() => Object.assign(ERROR_CODES, { GPU_OOM: 'changed' }), TypeError);
// Host tooling owns command/HTTP envelopes; normalize the runtime code and cause
// at that boundary rather than adding an errors dependency to those owners.
const command = normalizeToToolingCommandError(coded);
assert(command instanceof ToolingCommandError);
assert.equal(command.code, coded.code); assert.equal(command.cause, coded);
assert.equal(new DopplerNodeWebGPUError('DOPPLER_PROVIDER_ALREADY_ACTIVE', 'active').code, 'DOPPLER_PROVIDER_ALREADY_ACTIVE');
assert.equal(new ServeRequestError('bad input', 422).statusCode, 422);
assert.equal(new Error('defect') instanceof DopplerErrorBase, false);
assert.throws(() => createDopplerError('UNKNOWN', 'unregistered'), /unknown error code/);
console.log('error-taxonomy: common local identity preserves domain classes, names, codes, fields and causes');
