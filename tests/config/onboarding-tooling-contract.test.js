import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const result = spawnSync(process.execPath, ['tools/onboarding-tooling.js', 'check', '--json'], {
  cwd: process.cwd(),
  encoding: 'utf8',
});

assert.notEqual(result.status, null);
const payload = JSON.parse(result.stdout);
const issueCodes = new Set(
  Array.isArray(payload?.issues)
    ? payload.issues.map((issue) => issue?.code).filter((code) => typeof code === 'string')
    : []
);

assert.ok(
  !issueCodes.has('COMPARE_METRIC_NO_DOPPLER_PATH'),
  'onboarding compare-metric validation should accept ordered fallback Doppler paths'
);
assert.ok(
  !issueCodes.has('COMPARE_METRIC_NO_TJS_PATH'),
  'onboarding compare-metric validation should accept ordered fallback Transformers.js paths'
);
assert.ok(
  !issueCodes.has('RUNTIME_PROFILE_STRING_KERNEL_PATH'),
  'runtime profiles must not reintroduce string kernel-path registry IDs'
);
assert.ok(
  !issueCodes.has('RUNTIME_PROFILE_RUNTIME_MISSING'),
  'non-profile runtime policy assets should not be classified as runtime profiles'
);

console.log('onboarding-tooling-contract.test: ok');
