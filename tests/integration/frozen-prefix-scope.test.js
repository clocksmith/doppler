import assert from 'node:assert/strict';
import { validatePrefixOnlyTarget } from '../helpers/frozen-prefix-scope.js';

const reference = { expected: [{ steps: [{}, {}] }, { steps: [{}] }] };
assert.equal(validatePrefixOnlyTarget({ index: 0, step: 1, prefixesOnly: true }, reference), true);
assert.equal(validatePrefixOnlyTarget({ index: 0, step: 0 }, reference), false);
for (const fields of [{ index: 2 }, { index: -1 }, { step: 2 }, { step: -1 }, { step: 0.5 },
  { physicalCache: true }, { opIds: ['embed.out'] }, { prefixesOnly: 'true' }]) {
  assert.throws(() => validatePrefixOnlyTarget({ index: 0, step: 0, prefixesOnly: true, ...fields }, reference));
}
console.log('Frozen prefix-only scope controls passed');
