import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

function run(args) {
  return spawnSync(process.execPath, args, {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

{
  const result = run([
    'tools/convert-safetensors-node.js',
    'input-dir',
    'extra-input',
    '--config',
    'config.json',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unexpected positional arguments/);
}

{
  const result = run([
    'tools/convert-safetensors-node.js',
    'input-dir',
    '--worker-policy',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Missing value for --worker-policy/);
}

{
  const result = run([
    'tools/run-registry-verify.js',
    'demo-model',
    '--surface',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Missing value for --surface/);
}

{
  const result = run([
    'tools/onboarding-tooling.js',
    'check',
    '--stauts',
    'experimental',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unknown flag: --stauts/);
}

{
  const result = run([
    'tools/onboarding-tooling.js',
    'check',
    'extra-positional',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unexpected positional argument/);
}

{
  const result = run([
    'tools/run-node-tests.mjs',
    '--suite',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Missing value for --suite/);
}

{
  const result = run([
    'tools/run-node-coverage.mjs',
    '--policy',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Missing value for --policy/);
}

{
  const result = run([
    'tools/distill-studio-quality-gate.mjs',
    '--report',
    'report.json',
    '--min-steps',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Missing value for --min-steps/);
}

{
  const result = run([
    'tools/run-registry-verify.js',
    'demo-model',
    'extra-positional',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unexpected positional arguments/);
}

{
  const result = run([
    'tools/run-registry-verify.js',
    'demo-model',
    '--surface',
    'invalid-surface',
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /--surface must be one of/);
}

console.log('tools-cli-contract.test: ok');
