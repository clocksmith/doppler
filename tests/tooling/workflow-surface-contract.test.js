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
assert.match(automaticCi, /actions\/checkout@v7/);
assert.match(automaticCi, /actions\/setup-node@v6/);
assert.doesNotMatch(automaticCi, /playwright|models\/local|test:gpu/);

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
