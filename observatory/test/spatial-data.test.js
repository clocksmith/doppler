import test from 'node:test';
import assert from 'node:assert/strict';
import { candidateAtlas, coordinateJourney, rotatePoint } from '../public/views/spatial-data.js';

test('coordinate journey selects original channels by variance and preserves gaps', () => {
  const result = coordinateJourney([[0, 4, 0, 10], null, [2, 4, 8, 10]]);
  assert.deepEqual(result.axes, [2, 0, 1]);
  assert.deepEqual(result.points[0].values, [-4, -1, 0]);
  assert.equal(result.points[1], null);
  assert.equal(result.varianceShare, 1);
  assert.equal(result.extent, 4);
});

test('flat and missing geometry does not invent movement', () => {
  assert.equal(coordinateJourney([null, [NaN]]), null);
  const flat = coordinateJourney([[3], [3]]);
  assert.equal(flat.extent, 0);
  assert.equal(flat.varianceShare, null);
  assert.deepEqual(rotatePoint([0], 3), [0, 0]);
  assert.deepEqual(rotatePoint([1, 2, 3], 0, false), [1, 2]);
  assert.equal(coordinateJourney([[1], [1, 2]]), null);
});

test('candidate atlas never turns an unlisted probability into zero', () => {
  const run = { tokens: [
    { candidates: [{ tokenId: 7, text: 'a', probability: 0.6 }, { tokenId: 8, text: 'b', probability: 0 }] },
    { candidates: [{ tokenId: 8, text: 'b', probability: 0.8 }] },
    { candidates: null },
  ] };
  const atlas = candidateAtlas(run);
  assert.deepEqual(atlas.rows.map((row) => row.tokenId), [8, 7]);
  assert.deepEqual(atlas.rows[0].probabilities, [0, 0.8, null]);
  assert.deepEqual(atlas.rows[1].probabilities, [0.6, null, null]);
  assert.equal(candidateAtlas(run, 1).omitted, 1);
});

test('invalid and duplicate probabilities cannot contaminate the atlas', () => {
  const atlas = candidateAtlas({ tokens: [{ candidates: [
    { tokenId: 0, probability: 0.2 }, { tokenId: 0, probability: 0.9 },
    { tokenId: 1, probability: -1 }, { tokenId: 2, probability: Infinity },
  ] }] });
  assert.equal(atlas.rows.length, 1);
  assert.equal(atlas.rows[0].probabilities[0], 0.2);
});

test('large atlases bound both decisions and materialized candidate rows', () => {
  const run = { tokens: Array.from({ length: 5000 }, (_, i) => ({ candidates: [{ tokenId: i, text: String(i), probability: 0.5 }] })) };
  const atlas = candidateAtlas(run, 24, { center: 4000 });
  assert.equal(atlas.start, 3968);
  assert.equal(atlas.count, 64);
  assert.equal(atlas.omittedColumns, 4936);
  assert.equal(atlas.rows.length, 24);
  assert.equal(atlas.rows[0].tokenId, 3968);
  assert.equal(atlas.rows[0].probabilities.length, 64);
  assert.equal(atlas.rows[0].probabilities[0], 0.5);
  assert.equal(candidateAtlas(run, 24, { center: 4999 }).start, 4936);
});
