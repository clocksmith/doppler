import assert from 'node:assert/strict';
import { beginDiagnosticSequence, markDiagnosticSequenceStep, endDiagnosticSequence } from '../../src/inference/pipelines/text/generator/diagnostic-sequence.js';

// A forced-prefix diagnostic sequence keeps one emitter across the prefill call and the
// decode steps that follow it, labelling every record with the call's phase, absolute
// position (decode) and sequence step; per-call diagnostics inside it are refused.

const state = { isGenerating: false, currentSeqLen: 0, manifest: { modelId: 'm' }, resolvedKernelPath: { id: 'k' },
  operatorDiagnostics: null, diagnosticSequence: null };
assert.equal(markDiagnosticSequenceStep(state, 'decode', {}), false);
beginDiagnosticSequence(state, { captureConfig: { targetLayers: [3], targetLevel: 'full' } });
assert.throws(() => beginDiagnosticSequence(state), /already open/);
const emitter = state.operatorDiagnostics.emitter;
assert.equal(state.operatorDiagnostics.captureConfig.targetLevel, 'full');

state.currentSeqLen = 0;
assert.equal(markDiagnosticSequenceStep(state, 'prefill', {}), true);
emitter.emitRecord('layer.out', { layerIdx: 3, phase: null, tokenIndex: null });
state.currentSeqLen = 28;
for (let step = 0; step < 3; step++) {
  markDiagnosticSequenceStep(state, 'decode', {});
  emitter.emitRecord('layer.out', { layerIdx: 3, phase: null, tokenIndex: null });
  state.currentSeqLen++;
}
assert.throws(() => markDiagnosticSequenceStep(state, 'decode', { diagnostics: { enabled: true } }), /inside an open diagnostic sequence/);

const result = endDiagnosticSequence(state);
assert.equal(result.steps, 4);
assert.deepEqual(result.timeline.map((r) => [r.phase, r.tokenIndex, r.sequenceStep]),
  [['prefill', null, 0], ['decode', 28, 1], ['decode', 29, 2], ['decode', 30, 3]]);
assert.equal(state.operatorDiagnostics, null);
assert.equal(state.diagnosticSequence, null);
assert.throws(() => endDiagnosticSequence(state), /No diagnostic sequence is open/);
assert.throws(() => beginDiagnosticSequence({ ...state, isGenerating: true }), /during generation/);
console.log('diagnostic-sequence.test: ok');
