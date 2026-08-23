import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  buildBoundaryComparisonReceipt,
  compareFloat32Arrays,
  selectCandidateBoundaryValues,
} from '../../tools/compare-source-boundaries.js';

const exact = compareFloat32Arrays(new Float32Array([1, 2]), new Float32Array([1, 2]));
assert.equal(exact.exact, true);
assert.equal(exact.cosineSimilarity, 1);
assert.throws(
  () => compareFloat32Arrays(new Float32Array([1]), new Float32Array([1, 2])),
  /element-count mismatch/
);
assert.deepEqual(
  Array.from(selectCandidateBoundaryValues(
    new Float32Array([0, 0]),
    new Float32Array([1, 2, 3, 4]),
    'last-row'
  )),
  [3, 4]
);

const policyPath = 'src/config/forge/reference/glimmer-30b-decode-boundary-comparison.json';
const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
const expected = JSON.parse(await fs.readFile(policy.output, 'utf8'));
const observed = await buildBoundaryComparisonReceipt(policyPath);
assert.deepEqual(observed, expected);
assert.equal(observed.capture.comparisonCount, 21);
assert.equal(observed.finding.firstExactDivergence, 'generation.7.model.embedding.output');
assert.equal(observed.finding.firstToleranceDivergence, null);
assert.equal(observed.finding.classification, 'precision-lane-drift-within-diagnostic-tolerance');
assert.equal(observed.finding.tokenParityPassed, false);
assert.equal(observed.finding.promotionEligible, false);

console.log('source-boundary-comparison.test: ok');
