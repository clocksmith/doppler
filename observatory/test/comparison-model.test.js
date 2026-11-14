import test from 'node:test';
import assert from 'node:assert/strict';
import { forestRows, operationBreakdown, comparableCoverage } from '../public/views/comparison-model.js';

const run = (id, parent = null) => ({ id, parent, createdAt: '2026-11-12' });
test('forest retains repeated forks and nested descendants in parent order', () => {
  const a = run('a');
  const b = run('b', { runId: 'a', index: 2 });
  const c = run('c', { runId: 'a', index: 2 });
  const d = run('d', { runId: 'b', index: 0 });
  assert.deepEqual(forestRows(d, [a, b, c, d]).rows.map(({ run, depth }) => [run.id, depth]), [['a', 0], ['b', 1], ['d', 2], ['c', 1]]);
});
test('forest does not recurse forever on imported cyclic lineage', () => {
  const a = run('a', { runId: 'b', index: 0 });
  const b = run('b', { runId: 'a', index: 0 });
  const result = forestRows(a, [a, b]);
  assert.equal(result.cyclic, true);
  assert.equal(result.rows.length, 2);
});
test('a branch with a missing parent remains a visible root', () => {
  const a = run('a', { runId: 'missing', index: 0 });
  assert.equal(forestRows(a, [a]).root, a);
});
test('operation shares preserve zero, missing totals and overlapping scopes', () => {
  assert.deepEqual(operationBreakdown({ totalMs: 0, kernels: { zero: 0 } }).operations[0], { name: 'zero', time: 0, share: null });
  const result = operationBreakdown({ totalMs: 5, kernels: { matmul: 4, attention: 3, invalid: -1 } });
  assert.equal(result.exceedsTotal, true);
  assert.equal(result.remainder, 0);
  assert.equal(result.operations.length, 2);
  assert.equal(operationBreakdown({ totalMs: 5, kernels: { matmul: 3 } }).remainder, 2);
});
test('missing comparison cells and empty intersections cannot imply full coverage', () => {
  assert.deepEqual(comparableCoverage({ positions: [0, 1], layerCount: 2, relative: [[0, null], [0.1, 0]] }), { measured: 3, expected: 4, complete: false });
  assert.equal(comparableCoverage({ positions: [], layerCount: 2, relative: [] }).complete, false);
});
