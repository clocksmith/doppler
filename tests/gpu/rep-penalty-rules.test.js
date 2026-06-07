import assert from 'node:assert/strict';

import { selectRepPenaltyVariant } from '../../src/gpu/kernels/rep-penalty.js';

assert.equal(selectRepPenaltyVariant(false), 'default');
assert.equal(selectRepPenaltyVariant(true), 'default_f16');

console.log('rep-penalty-rules.test: ok');
