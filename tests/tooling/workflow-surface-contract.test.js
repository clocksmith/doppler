import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';

const workflowDirectory = new URL('../../.github/workflows/', import.meta.url);
const workflowFiles = readdirSync(workflowDirectory)
  .filter((fileName) => fileName.endsWith('.yml'))
  .sort();

assert.deepEqual(workflowFiles, [
  'check-green.yml',
  'manual-runtime-validation.yml',
]);

const automaticCi = readFileSync(new URL('check-green.yml', workflowDirectory), 'utf8');
assert.match(automaticCi, /^name: Default Green Chain$/m);
assert.match(automaticCi, /^  check-green:$/m);
assert.match(automaticCi, /pull_request:/);
assert.match(automaticCi, /push:/);
assert.match(automaticCi, /npm run ci:check/);
assert.match(automaticCi, /npm ci --omit=optional/);
assert.match(automaticCi, /actions\/checkout@v7/);
assert.match(automaticCi, /actions\/setup-node@v6/);
assert.doesNotMatch(
  automaticCi,
  /playwright|models\/local|test:gpu|verify:model|DOPPLER_SMOKE|continue-on-error/i
);

const packageJson = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
assert.equal(
  packageJson.scripts?.['ci:check'],
  'npm run test:ci && npm run kernels:check && npm run check:green'
);
const ciTestScript = packageJson.scripts?.['test:ci'] ?? '';
assert.doesNotMatch(ciTestScript, /--suite|test:gpu|tests\/kernels|playwright|models\/local/i);
for (const testFile of [
  'tests/config/ci-workflow-local-model-contract.test.js',
  'tests/config/tracked-benchmark-evidence-contract.test.js',
  'tests/converter/quantizer.test.js',
  'tests/inference/attention-softmax-stability.test.js',
  'tests/inference/bdpa-steamroller.test.js',
  'tests/inference/generate-token-ids-behavioral-parity.test.js',
  'tests/integration/command-api.test.js',
  'tests/integration/doppler-provider-contract.test.js',
  'tests/tooling/workflow-surface-contract.test.js',
]) {
  assert.match(ciTestScript, new RegExp(testFile.replaceAll('.', '\\.')));
}

const manualRuntime = readFileSync(new URL('manual-runtime-validation.yml', workflowDirectory), 'utf8');
assert.match(manualRuntime, /workflow_dispatch:/);
assert.doesNotMatch(manualRuntime, /pull_request:|\n  push:/);
for (const lane of [
  'node-kernels',
  'browser-kernels',
  'opfs-text',
  'opfs-embedding',
]) {
  assert.match(manualRuntime, new RegExp(`- ${lane}`));
}

console.log('workflow-surface-contract.test: ok');
