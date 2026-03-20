import assert from 'node:assert/strict';

const {
  chooseNullish,
  chooseDefined,
  chooseDefinedWithSource,
  mergeShallowObject,
  mergeLayeredShallowObjects,
  replaceSubtree,
  mergeKernelPathPolicy,
  mergeExecutionPatchLists,
} = await import('../../src/config/merge-helpers.js');

// === chooseNullish ===

assert.equal(chooseNullish('override', 'fallback'), 'override');
assert.equal(chooseNullish(null, 'fallback'), 'fallback');
assert.equal(chooseNullish(undefined, 'fallback'), 'fallback');
assert.equal(chooseNullish(0, 'fallback'), 0);
assert.equal(chooseNullish('', 'fallback'), '');
assert.equal(chooseNullish(false, 'fallback'), false);

// === chooseDefined ===

assert.equal(chooseDefined('override', 'fallback'), 'override');
assert.equal(chooseDefined(undefined, 'fallback'), 'fallback');
// null is defined — should use override
assert.equal(chooseDefined(null, 'fallback'), null);
assert.equal(chooseDefined(0, 'fallback'), 0);
assert.equal(chooseDefined(false, 'fallback'), false);

// === chooseDefinedWithSource ===

{
  const sources = new Map();
  const result = chooseDefinedWithSource('test.path', 'runtime-val', 'manifest-val', sources);
  assert.equal(result, 'runtime-val');
  assert.equal(sources.get('test.path'), 'runtime');
}

{
  const sources = new Map();
  const result = chooseDefinedWithSource('test.path', undefined, 'manifest-val', sources);
  assert.equal(result, 'manifest-val');
  assert.equal(sources.get('test.path'), 'manifest');
}

// null override is defined and should win
{
  const sources = new Map();
  const result = chooseDefinedWithSource('test.path', null, 'manifest-val', sources);
  assert.equal(result, null);
  assert.equal(sources.get('test.path'), 'runtime');
}

// Works without sources map
{
  const result = chooseDefinedWithSource('test.path', 'val', 'fallback', null);
  assert.equal(result, 'val');
}

// === mergeShallowObject ===

{
  const base = { a: 1, b: 2 };
  const override = { b: 3, c: 4 };
  const result = mergeShallowObject(base, override);
  assert.deepEqual(result, { a: 1, b: 3, c: 4 });
}

// undefined override returns base
{
  const base = { a: 1 };
  assert.deepEqual(mergeShallowObject(base, undefined), base);
}

// null override throws
assert.throws(
  () => mergeShallowObject({ a: 1 }, null),
  /shallow object overrides must be plain objects/
);

// array override throws
assert.throws(
  () => mergeShallowObject({ a: 1 }, [1, 2]),
  /shallow object overrides must be plain objects/
);

// string override throws
assert.throws(
  () => mergeShallowObject({ a: 1 }, 'bad'),
  /shallow object overrides must be plain objects/
);

// === mergeLayeredShallowObjects ===

{
  const result = mergeLayeredShallowObjects(
    { a: 1, b: 2 },
    { b: 3 },
    { c: 4 }
  );
  assert.deepEqual(result, { a: 1, b: 3, c: 4 });
}

{
  const result = mergeLayeredShallowObjects({ a: 1 });
  assert.deepEqual(result, { a: 1 });
}

{
  const result = mergeLayeredShallowObjects({ a: 1 }, undefined, { b: 2 });
  assert.deepEqual(result, { a: 1, b: 2 });
}

// === replaceSubtree ===

assert.equal(replaceSubtree('override', 'fallback'), 'override');
assert.equal(replaceSubtree(null, 'fallback'), 'fallback');
assert.equal(replaceSubtree(undefined, 'fallback'), 'fallback');
assert.equal(replaceSubtree(0, 'fallback'), 0);
assert.equal(replaceSubtree(false, 'fallback'), false);

// === mergeKernelPathPolicy ===

// Defaults when both undefined
{
  const result = mergeKernelPathPolicy(undefined, undefined);
  assert.equal(result.mode, 'locked');
  assert.deepEqual(result.sourceScope, ['model', 'manifest']);
  assert.deepEqual(result.allowSources, ['model', 'manifest']);
  assert.equal(result.onIncompatible, 'error');
}

// Override replaces base
{
  const base = { mode: 'locked', sourceScope: ['model'] };
  const override = { mode: 'capability-aware', sourceScope: ['model', 'config'] };
  const result = mergeKernelPathPolicy(base, override);
  assert.equal(result.mode, 'capability-aware');
  assert.deepEqual(result.sourceScope, ['model', 'config']);
}

// Base values used when override doesn't set them
{
  const base = { mode: 'capability-aware', onIncompatible: 'remap' };
  const result = mergeKernelPathPolicy(base, {});
  assert.equal(result.mode, 'capability-aware');
  assert.equal(result.onIncompatible, 'remap');
}

// Invalid mode throws
assert.throws(
  () => mergeKernelPathPolicy({ mode: 'invalid' }, undefined),
  /must be "locked" or "capability-aware"/
);

// Legacy "runtime" source throws
assert.throws(
  () => mergeKernelPathPolicy({ sourceScope: ['runtime'] }, undefined),
  /does not accept legacy "runtime"/
);

// Legacy "execution_v0" source throws
assert.throws(
  () => mergeKernelPathPolicy({ sourceScope: ['execution_v0'] }, undefined),
  /does not accept "execution-v0"/
);

// "execution-v0" source throws (v0 removed)
assert.throws(
  () => mergeKernelPathPolicy({ sourceScope: ['execution-v0'] }, undefined),
  /does not accept "execution-v0"/
);

// Invalid source throws
assert.throws(
  () => mergeKernelPathPolicy({ sourceScope: ['invalid'] }, undefined),
  /must be model\|manifest\|config/
);

// null policy throws
assert.throws(
  () => mergeKernelPathPolicy(null, undefined),
  /must not be null/
);

// Array policy throws
assert.throws(
  () => mergeKernelPathPolicy([], undefined),
  /must be an object/
);

// Empty sourceScope throws
assert.throws(
  () => mergeKernelPathPolicy({ sourceScope: [] }, undefined),
  /must be a non-empty array/
);

// Invalid onIncompatible throws
assert.throws(
  () => mergeKernelPathPolicy({ onIncompatible: 'ignore' }, undefined),
  /must be "error" or "remap"/
);

// Source aliases must match when both provided
assert.throws(
  () => mergeKernelPathPolicy({
    sourceScope: ['model'],
    allowSources: ['model', 'config'],
  }, undefined),
  /must match exactly/
);

// Deduplicated sources
{
  const result = mergeKernelPathPolicy({ sourceScope: ['model', 'model', 'config'] }, undefined);
  assert.deepEqual(result.sourceScope, ['model', 'config']);
}

// config is valid
{
  const result = mergeKernelPathPolicy({ sourceScope: ['config'] }, undefined);
  assert.deepEqual(result.sourceScope, ['config']);
}

// === mergeExecutionPatchLists ===

// Both undefined returns empty defaults
{
  const result = mergeExecutionPatchLists(undefined, undefined);
  assert.deepEqual(result, { set: [], remove: [], add: [] });
}

// Override replaces base
{
  const base = { set: [1], remove: [2], add: [3] };
  const override = { set: [10] };
  const result = mergeExecutionPatchLists(base, override);
  assert.deepEqual(result.set, [10]);
  assert.deepEqual(result.remove, [2]);
  assert.deepEqual(result.add, [3]);
}

// Base used when override missing
{
  const base = { set: [1] };
  const result = mergeExecutionPatchLists(base, undefined);
  assert.deepEqual(result.set, [1]);
  assert.deepEqual(result.remove, []);
  assert.deepEqual(result.add, []);
}

console.log('merge-helpers.test: ok');
