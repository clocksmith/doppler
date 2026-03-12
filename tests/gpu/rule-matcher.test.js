import assert from 'node:assert/strict';

const { matchesRule, selectByRules } = await import('../../src/gpu/kernels/rule-matcher.js');

// Exact match
assert.equal(matchesRule({ isDecode: true }, { isDecode: true }), true);
assert.equal(matchesRule({ isDecode: true }, { isDecode: false }), false);

// Multiple keys
assert.equal(matchesRule({ isDecode: true, bDtype: 'q4k' }, { isDecode: true, bDtype: 'q4k' }), true);
assert.equal(matchesRule({ isDecode: true, bDtype: 'q4k' }, { isDecode: true, bDtype: 'f16' }), false);

// Null/undefined match (null match matches anything)
assert.equal(matchesRule(null, { isDecode: true }), true);
assert.equal(matchesRule(undefined, { isDecode: true }), true);

// Empty match matches anything
assert.equal(matchesRule({}, { isDecode: true }), true);

// Range operators: eq, neq
assert.equal(matchesRule({ M: { eq: 1 } }, { M: 1 }), true);
assert.equal(matchesRule({ M: { eq: 1 } }, { M: 2 }), false);
assert.equal(matchesRule({ M: { neq: 1 } }, { M: 2 }), true);
assert.equal(matchesRule({ M: { neq: 1 } }, { M: 1 }), false);

// Range operators: gt, gte, lt, lte
assert.equal(matchesRule({ N: { gt: 100 } }, { N: 101 }), true);
assert.equal(matchesRule({ N: { gt: 100 } }, { N: 100 }), false);
assert.equal(matchesRule({ N: { gte: 100 } }, { N: 100 }), true);
assert.equal(matchesRule({ N: { gte: 100 } }, { N: 99 }), false);
assert.equal(matchesRule({ N: { lt: 100 } }, { N: 99 }), true);
assert.equal(matchesRule({ N: { lt: 100 } }, { N: 100 }), false);
assert.equal(matchesRule({ N: { lte: 100 } }, { N: 100 }), true);
assert.equal(matchesRule({ N: { lte: 100 } }, { N: 101 }), false);

// Combined range
assert.equal(matchesRule({ N: { gt: 10, lt: 100 } }, { N: 50 }), true);
assert.equal(matchesRule({ N: { gt: 10, lt: 100 } }, { N: 5 }), false);
assert.equal(matchesRule({ N: { gt: 10, lt: 100 } }, { N: 200 }), false);

// 'in' operator
assert.equal(matchesRule({ dtype: { in: ['f16', 'f32'] } }, { dtype: 'f16' }), true);
assert.equal(matchesRule({ dtype: { in: ['f16', 'f32'] } }, { dtype: 'q4k' }), false);

// String operators: contains
assert.equal(matchesRule({ name: { contains: 'attn' } }, { name: 'self_attn.q_proj' }), true);
assert.equal(matchesRule({ name: { contains: 'attn' } }, { name: 'ffn.up_proj' }), false);
assert.equal(matchesRule({ name: { contains: ['attn', 'ffn'] } }, { name: 'ffn.gate' }), true);

// String operators: startsWith
assert.equal(matchesRule({ name: { startsWith: 'model.' } }, { name: 'model.layers.0' }), true);
assert.equal(matchesRule({ name: { startsWith: 'model.' } }, { name: 'lm_head' }), false);
assert.equal(matchesRule({ name: { startsWith: ['model.', 'lm_head'] } }, { name: 'lm_head' }), true);

// String operators: endsWith
assert.equal(matchesRule({ name: { endsWith: '.weight' } }, { name: 'norm.weight' }), true);
assert.equal(matchesRule({ name: { endsWith: '.weight' } }, { name: 'norm.bias' }), false);

// String operators on non-string actual value
assert.equal(matchesRule({ name: { contains: 'x' } }, { name: 42 }), false);
assert.equal(matchesRule({ name: { startsWith: 'x' } }, { name: null }), false);

// Missing context key
assert.equal(matchesRule({ missing: true }, { other: true }), false);

// === selectByRules ===

{
  const rules = [
    { match: { isDecode: true, bDtype: 'q4k' }, value: 'q4k_decode' },
    { match: { isDecode: true }, value: 'generic_decode' },
    { match: {}, value: 'fallback' },
  ];
  assert.equal(selectByRules(rules, { isDecode: true, bDtype: 'q4k' }), 'q4k_decode');
  assert.equal(selectByRules(rules, { isDecode: true, bDtype: 'f16' }), 'generic_decode');
  assert.equal(selectByRules(rules, { isDecode: false }), 'fallback');
}

// selectByRules throws when no rule matches
{
  const rules = [
    { match: { isDecode: true }, value: 'decode_only' },
  ];
  assert.throws(
    () => selectByRules(rules, { isDecode: false }),
    /no rule matched context/
  );
}

// selectByRules with empty rules
assert.throws(
  () => selectByRules([], { isDecode: true }),
  /no rule matched context/
);

console.log('rule-matcher.test: ok');
