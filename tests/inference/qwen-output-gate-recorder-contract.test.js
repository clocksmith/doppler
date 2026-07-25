import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const planSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/plan.js', import.meta.url), 'utf8');
const immediateSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/executor-immediate.js', import.meta.url), 'utf8');
const recordedSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/executor-recorded.js', import.meta.url), 'utf8');
const interpreterSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/interpreter.js', import.meta.url), 'utf8');

assert.match(
  planSource,
  /semantics:\s*normalized\.outputGate\.enabled\s*\?\s*'sigmoid'\s*:\s*'none'/,
  'the semantic attention plan must route enabled output gates through sigmoid'
);
assert.match(
  immediateSource,
  /interpretAttentionWithRecorder\(/,
  'the immediate adapter must delegate output-gate execution to the canonical interpreter'
);
assert.doesNotMatch(
  immediateSource,
  /gateActivation|rawGateType|outputGateType/,
  'the immediate adapter must not duplicate output-gate semantics'
);
assert.match(
  recordedSource,
  /interpretAttentionWithRecorder\(/,
  'the recorded adapter must delegate output-gate execution to the canonical interpreter'
);
assert.doesNotMatch(
  recordedSource,
  /gateActivation|rawGateType|outputGateType/,
  'the recorded adapter must not duplicate output-gate semantics'
);
assert.match(
  interpreterSource,
  /const gateActivation = attentionPlan\.outputGate\.semantics;/,
  'the canonical interpreter must consume output-gate semantics from the shared plan'
);
for (const [label, source] of [
  ['plan', planSource],
  ['immediate executor', immediateSource],
  ['recorded executor', recordedSource],
  ['interpreter', interpreterSource],
]) {
  assert.doesNotMatch(
    source,
    /rawGateType === 'swish'[\s\S]*\? 'silu'/,
    `${label} must not reinterpret Qwen outputGateType=swish as SiLU`
  );
}

console.log('qwen-output-gate-recorder-contract.test: ok');
