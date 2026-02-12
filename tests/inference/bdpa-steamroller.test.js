import assert from 'node:assert/strict';
import { steamrollerRadixArgsort, generateBDPAData } from '../../src/inference/pipelines/text/bdpa-steamroller.js';

const tokens = new Int32Array([5, 7, 5, 9]);
const sorted = steamrollerRadixArgsort(tokens);

// Sorted token ids should be [5, 5, 7, 9].
const sortedTokenIds = Array.from(sorted, (idx) => tokens[idx]);
assert.deepEqual(sortedTokenIds, [5, 5, 7, 9]);

const kvSize = 2;
const kBuffer = new Float32Array([
  2.0, 4.0,  // token 0 -> id 5
  10.0, 20.0, // token 1 -> id 7
  6.0, 8.0,  // token 2 -> id 5
  30.0, 40.0, // token 3 -> id 9
]);
const vBuffer = new Float32Array([
  1.0, 3.0,
  11.0, 21.0,
  5.0, 7.0,
  31.0, 41.0,
]);

const bdpa = generateBDPAData(sorted, kBuffer, vBuffer, tokens, kvSize);
assert.equal(bdpa.numBasisVectors, 3);
assert.equal(bdpa.iFlat.length, tokens.length * 3);
assert.equal(bdpa.pDeltaK.length, tokens.length * kvSize);
assert.equal(bdpa.pDeltaV.length, tokens.length * kvSize);

// Token id 5 appears twice with K vectors [2,4] and [6,8] => centroid [4,6].
const firstBasisK = Array.from(bdpa.tBasisK.slice(0, kvSize));
assert.deepEqual(firstBasisK, [4, 6]);

// iFlat stores original positions in slot+2.
const originalPositions = [];
for (let i = 0; i < tokens.length; i++) {
  originalPositions.push(bdpa.iFlat[i * 3 + 2]);
}
assert.deepEqual(originalPositions.sort((a, b) => a - b), [0, 1, 2, 3]);

console.log('bdpa-steamroller.test: ok');
