import assert from 'node:assert/strict';
import { borrowDecodeFfnIntermediate } from '../../src/inference/pipelines/text/ffn/decode-intermediate.js';
import { releaseOrTrack } from '../../src/inference/pipelines/text/ops.js';
const buffer = { size: 4096 };
const owner = { getFFNIntermediateBuffer: () => buffer, ownsBuffer: value => value === buffer };
const context = { phase: 'decode', decodeBuffers: owner };
assert.equal(borrowDecodeFfnIntermediate(context, 1, 1024, 'f32'), buffer);
assert.equal(borrowDecodeFfnIntermediate(context, 1, 1025, 'f32'), null);
assert.equal(borrowDecodeFfnIntermediate(context, 2, 1024, 'f32'), null);
assert.equal(borrowDecodeFfnIntermediate(context, 1, 1024, 'f16'), null);
for (const change of [{ phase: 'prefill' }, { activationPrecision: 'two-f32' }, { decodeBuffers: null }, { trainingCapture: {} },
  { operatorDiagnostics: {} }, { debugProbes: [{}] }]) {
  assert.equal(borrowDecodeFfnIntermediate({ ...context, ...change }, 1, 1024, 'f32'), null);
}
let tracked = 0;
const recorder = { trackTemporaryBuffer() { tracked++; } };
releaseOrTrack(recorder, buffer, owner);
releaseOrTrack(recorder, {}, owner);
assert.equal(tracked, 1, 'recorder cleanup must retain the borrowed scratch but track ordinary outputs');
console.log('decode-ffn-intermediate.test: passed');
