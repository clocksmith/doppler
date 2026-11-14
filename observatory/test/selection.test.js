import test from 'node:test';
import assert from 'node:assert/strict';
import { createSelection } from '../public/selection.js';
import { describeProvenance, PROVENANCE } from '../public/viz/provenance.js';
import { inspectorRows } from '../public/viz/inspect.js';
import { captureRequest } from '../public/viz/gap.js';
import { answerIndex, layerChanges, sequencePosition } from '../public/shared-run.js';
import { differenceStrength, magnitudeShare, probabilityBands, signToken } from '../public/viz/scale.js';

test('one selection: a change reaches every subscriber, and no change reaches none', () => {
  const selection = createSelection({ runId: 'run-a' });
  const seen = [];
  const stop = selection.subscribe((state, changed) => seen.push([state.position, changed]));
  selection.select({ position: 4 });
  selection.select({ position: 4 });
  assert.deepEqual(seen, [[4, ['position']]]);
  stop();
  selection.select({ position: 5 });
  assert.equal(seen.length, 1);
});

test('choosing another run drops what belonged to the old one, and keeps the layer', () => {
  const selection = createSelection({ runId: 'run-a', position: 4, layer: 7, op: 'gate', compareRunId: 'run-b' });
  const next = selection.select({ runId: 'run-b' });
  assert.deepEqual([next.position, next.op, next.layer], [null, null, 7]);
  assert.equal(next.compareRunId, null, 'A run is not compared with itself');
  assert.equal(selection.select({ runId: 'run-c', position: 2 }).position, 2);
});

test('the selection refuses fields and values it does not hold', () => {
  const selection = createSelection();
  assert.throws(() => selection.select({ token: 3 }), /not part of the selection/);
  assert.throws(() => selection.select({ position: -1 }), /whole number/);
  assert.throws(() => selection.select({ runId: 7 }), /non-empty string/);
});

test('every mark needs a provenance kind', () => {
  assert.throws(() => describeProvenance(), /Every mark needs a provenance kind; got none/);
  assert.throws(() => describeProvenance('guessed'), /"guessed"/);
  assert.equal(describeProvenance('projected').label, 'Projected');
  assert.deepEqual(Object.keys(PROVENANCE), ['measured', 'derived', 'projected', 'reconstructed', 'estimated', 'unrun', 'missing']);
});

test('shared encodings: sign, magnitude, probability with its remainder, difference', () => {
  assert.deepEqual([signToken(2), signToken(-2), signToken(0)], ['--doppler-blue', '--doppler-red', null]);
  assert.deepEqual([magnitudeShare(-3, 6), magnitudeShare(9, 6), magnitudeShare(1, 0)], [0.5, 1, 0]);
  const bands = probabilityBands([{ tokenId: 1, probability: 0.6 }, { tokenId: 2, probability: 0.1 }]);
  assert.equal(bands.length, 3);
  assert.ok(bands[2].other && Math.abs(bands[2].share - 0.3) < 1e-12, 'The unlisted probability is its own band');
  assert.throws(() => probabilityBands([{ probability: 1.2 }]), /between 0 and 1/);
  assert.deepEqual([differenceStrength(0), differenceStrength(1e-9), differenceStrength(1)], [0, 0.06, 1]);
  assert.throws(() => differenceStrength(NaN));
});

test('the inspector needs a title, a value and a provenance, and says what was not recorded', () => {
  assert.throws(() => inspectorRows({ title: 'x', value: 1 }), /provenance kind/);
  assert.throws(() => inspectorRows({ value: 1, provenance: 'measured' }), /title/);
  assert.throws(() => inspectorRows({ title: 'x', provenance: 'measured' }), /needs a value/);
  assert.deepEqual(inspectorRows({ title: 'x', value: null, provenance: 'missing', device: 'apple metal-3' }),
    [['Value', 'not recorded'], ['Source', 'not recorded'], ['Precision', 'not recorded'], ['Device', 'apple metal-3']]);
});

test('a capture request names only what was asked for, and refuses what Doppler would not accept', () => {
  assert.deepEqual(captureRequest({ roles: ['kv-key', 'kv-key'], layers: [3, 1, 3], tokens: { from: 2, to: 5 } }), { roles: ['kv-key'], layers: [1, 3], tokens: { from: 2, to: 5 } });
  assert.deepEqual(captureRequest({ roles: [] }), { roles: [] });
  assert.throws(() => captureRequest({ layers: [-1] }), /whole numbers/);
  assert.throws(() => captureRequest({ tokens: { from: 4, to: 2 } }), /from <= to/);
});

test('positions count the input and then the answer', () => {
  const run = { inputTokenIds: [1, 2, 3], tokens: [{}, {}] };
  assert.equal(sequencePosition(run, 1), 4);
  assert.deepEqual([answerIndex(run, 4), answerIndex(run, 2), answerIndex(run, 5), answerIndex(run, null)], [1, null, null, null]);
});

test('layer changes: measured from the layer before, and a missing layer is null, not zero', () => {
  // Two positions x three layers x width 2.
  const values = new Float32Array([3, 4, 3, 4, 0, 5, /* position 1 */ 1, 0, 9, 9, 0, 2]);
  const data = Buffer.from(values.buffer).toString('base64');
  const run = { layerOutputs: { layers: 3, width: 2, positions: [0, 1], missing: [[1, 1]], encoding: 'f32-base64', data } };
  const first = layerChanges(run, 0);
  assert.deepEqual(first.map((row) => row.size), [5, 5, 5]);
  assert.deepEqual([first[0].change, first[1].change], [null, 0]);
  assert.ok(Math.abs(first[2].change - Math.sqrt(10) / 5) < 1e-6);
  assert.deepEqual(layerChanges(run, 1).map((row) => row.change), [null, null, null], 'Nothing is measured across a layer that was not recorded');
  assert.equal(layerChanges(run, 7), null);
  assert.equal(layerChanges({ layerOutputs: null }, 0), null);
});
