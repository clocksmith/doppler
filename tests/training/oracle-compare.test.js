import assert from 'node:assert/strict';

import { compare, maxAbsDifference } from './browser/oracle-compare.js';

const exact = compare(Float32Array.from([1, 2, 3]), [1, 2, 3]);
assert.deepEqual(exact, { elementCount: 3, allFinite: true, actualFinite: true, expectedFinite: true, maxAbsError: 0, rmse: 0 });

const off = compare([1, 2, 4], [1, 2, 2]);
assert.equal(off.maxAbsError, 2);
assert.ok(Math.abs(off.rmse - Math.sqrt(4 / 3)) < 1e-12);
assert.equal(maxAbsDifference([0.5], [0.25]), 0.25);

// The defect the shared comparator closes: a longer actual array used to
// report zero error over the reference prefix.
assert.throws(() => compare([1, 2, 999], [1, 2]), /length mismatch: actual has 3 elements, expected has 2/);
assert.throws(() => compare([1], [1, 2]), /length mismatch/);
assert.throws(() => compare(null, [1]), /two numeric arrays/);
assert.throws(() => compare([1], 'x'), /two numeric arrays/);

const nan = compare([1, Number.NaN], [1, 2]);
assert.equal(nan.allFinite, false);
assert.ok(Number.isNaN(nan.maxAbsError), 'a non-finite actual value poisons the error metric rather than hiding');
const badReference = compare([1, 2], [1, Number.POSITIVE_INFINITY]);
assert.equal(badReference.expectedFinite, false);
assert.equal(badReference.actualFinite, true);
assert.equal(badReference.allFinite, false, 'a non-finite reference fails the comparison');
assert.equal(compare([], []).rmse, 0);

console.log('oracle-compare: shared metrics, length and finiteness checks verified');
