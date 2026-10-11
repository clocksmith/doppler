import assert from 'node:assert/strict';

export function validatePrefixOnlyTarget(target, reference) {
  assert(target.prefixesOnly === undefined || typeof target.prefixesOnly === 'boolean', 'prefixesOnly must be boolean');
  if (target.prefixesOnly !== true) return false;
  assert(target.physicalCache !== true && target.opIds === undefined, 'Prefix-only replay cannot request observations');
  assert(Number.isSafeInteger(target.index) && target.index >= 0 && target.index < reference.expected.length,
    'Prefix request index is outside the frozen reference');
  assert(Number.isSafeInteger(target.step) && target.step >= 0
    && target.step < reference.expected[target.index].steps.length, 'Prefix step is outside the frozen reference');
  return true;
}
