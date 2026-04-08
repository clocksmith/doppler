import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const moeSource = readFileSync(new URL('../../src/inference/pipelines/text/moe-gpu.js', import.meta.url), 'utf8');
const moeFfnSource = readFileSync(new URL('../../src/inference/pipelines/text/ffn/moe.js', import.meta.url), 'utf8');

assert.match(
  moeSource,
  /assertImplicitDtypeTransitionAllowed\(\{[\s\S]*op: 'moe_gate_up_bias'/,
  'MoE gate/up bias dtype repacks must fail fast when execution-v1 requires explicit cast steps'
);

assert.match(
  moeFfnSource,
  /executionPolicies: context\.executionPolicies \?\? null/,
  'MoE FFN dispatch must thread execution-v1 policies into the GPU expert path'
);

console.log('moe-dtype-contract-source.test: ok');
