import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const {
  DEFAULT_FINITENESS_ABS_THRESHOLD,
  resolveFinitenessAbsThreshold,
  shouldTriggerFinitenessValue,
} = await import('../../src/gpu/kernels/check-finiteness.js');

assert.equal(resolveFinitenessAbsThreshold(undefined), DEFAULT_FINITENESS_ABS_THRESHOLD);
assert.equal(resolveFinitenessAbsThreshold(50000), 50000);
assert.equal(resolveFinitenessAbsThreshold(-1), DEFAULT_FINITENESS_ABS_THRESHOLD);

assert.equal(shouldTriggerFinitenessValue(100), false);
assert.equal(shouldTriggerFinitenessValue(65501, 65500), true);
assert.equal(shouldTriggerFinitenessValue(-70000, 65500), true);
assert.equal(shouldTriggerFinitenessValue(Number.POSITIVE_INFINITY, 65500), true);
assert.equal(shouldTriggerFinitenessValue(Number.NEGATIVE_INFINITY, 65500), true);
assert.equal(shouldTriggerFinitenessValue(Number.NaN, 65500), true);

console.log('finiteness-overflow-gate.test: ok');
