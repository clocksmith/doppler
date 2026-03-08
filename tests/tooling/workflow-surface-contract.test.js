import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const kernelTests = readFileSync(new URL('../../.github/workflows/kernel-tests.yml', import.meta.url), 'utf8');
const trainingCalibrateSmoke = readFileSync(new URL('../../.github/workflows/training-calibrate-smoke.yml', import.meta.url), 'utf8');
const onboardingTooling = readFileSync(new URL('../../.github/workflows/onboarding-tooling.yml', import.meta.url), 'utf8');
const inferenceGuard = readFileSync(new URL('../../.github/workflows/inference-guard.yml', import.meta.url), 'utf8');

for (const requiredPath of [
  "'tools/run-node-tests.mjs'",
  "'tools/run-node-test-file.mjs'",
  "'tools/run-node-coverage.mjs'",
  "'tools/node-test-runtime-setup.mjs'",
  "'tools/doppler-cli.js'",
  "'package.json'",
]) {
  assert.match(kernelTests, new RegExp(requiredPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
}

for (const requiredPath of [
  "'tools/run-node-tests.mjs'",
  "'tools/run-node-test-file.mjs'",
  "'tools/node-test-runtime-setup.mjs'",
  "'package.json'",
]) {
  assert.match(trainingCalibrateSmoke, new RegExp(requiredPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
}

for (const requiredPath of [
  "'tools/run-node-tests.mjs'",
  "'tools/run-node-test-file.mjs'",
  "'tools/node-test-runtime-setup.mjs'",
  "'package.json'",
]) {
  assert.match(readFileSync(new URL('../../.github/workflows/training-verify-smoke.yml', import.meta.url), 'utf8'), new RegExp(requiredPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
}

assert.match(onboardingTooling, /'docs\/rdrr-format\.md'/);
assert.doesNotMatch(onboardingTooling, /'docs\/formats\.md'/);

assert.match(inferenceGuard, /'src\/gpu\/kernels\/check-finiteness\.js'/);
assert.doesNotMatch(inferenceGuard, /check_finiteness\.wgsl/);

console.log('workflow-surface-contract.test: ok');
