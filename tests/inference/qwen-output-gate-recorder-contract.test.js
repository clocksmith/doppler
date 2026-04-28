import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const runSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/run.js', import.meta.url), 'utf8');
const recordSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/record.js', import.meta.url), 'utf8');

assert.match(
  runSource,
  /const gateActivation = 'sigmoid';/,
  'run.js must route Qwen attention output gates through sigmoid'
);
assert.match(
  recordSource,
  /const gateActivation = 'sigmoid';/,
  'record.js must mirror run.js for Qwen attention output gates'
);
assert.doesNotMatch(
  recordSource,
  /rawGateType === 'swish'[\s\S]*\? 'silu'/,
  'record.js must not reinterpret Qwen outputGateType=swish as SiLU'
);

console.log('qwen-output-gate-recorder-contract.test: ok');
