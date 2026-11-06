import assert from 'node:assert/strict';
import { compareLearnedLayerCapture } from '../helpers/learned-layer-observation.js';

const capture = { shape: [3, 4], fullTensorDigest: 'fixture', data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] };
const reference = { samples: [[1, 3], [9, 11]] };
const exact = compareLearnedLayerCapture(capture, reference, [0, 2], 2);
assert.equal(exact.maxAbsoluteError, 0);
assert.equal(exact.relativeL2, 0);
assert.deepEqual(exact.rows, [{ position: 0, observed: [1, 3] }, { position: 2, observed: [9, 11] }]);
const changed = structuredClone(capture);
changed.data[10] += 2;
assert.equal(compareLearnedLayerCapture(changed, reference, [0, 2], 2).maxAbsoluteError, 2);
assert.throws(() => compareLearnedLayerCapture({ ...capture, data: capture.data.slice(1) }, reference, [0, 2], 2));
assert.throws(() => compareLearnedLayerCapture(capture, reference, [0, 3], 2));
assert.throws(() => compareLearnedLayerCapture({ ...capture, error: 'readback failed' }, reference, [0, 2], 2));
const invalid = structuredClone(capture);
invalid.data[1] = NaN;
assert.throws(() => compareLearnedLayerCapture(invalid, reference, [0, 2], 2), /finite/);
console.log('learned layer observations: source coordinates and failed captures checked');
