import test from 'node:test';
import assert from 'node:assert/strict';
import { diverging, robustScale, tileSide } from '../public/views/activity.js';

test('tiles are square and hold every value', () => {
  assert.equal(tileSide(1024), 32);
  assert.equal(tileSide(640), 26);
  assert.ok(tileSide(640) ** 2 >= 640);
});

test('diverging colours: white at zero, blue below, red above, clipped beyond ±1', () => {
  assert.deepEqual(diverging(0), [255, 255, 255]);
  assert.deepEqual(diverging(-1), [37, 99, 235]);
  assert.deepEqual(diverging(1), [209, 59, 77]);
  assert.deepEqual(diverging(5), diverging(1));
  assert.deepEqual(diverging(Number.NaN), [255, 255, 255]);
});

test('robust scale ignores a single outlier', () => {
  const values = [...Array.from({ length: 999 }, (_, i) => (i % 2 ? 1 : -1)), 1000];
  assert.equal(robustScale(values), 1);
  assert.equal(robustScale([]), null);
});
