import assert from 'node:assert/strict';
import test from 'node:test';
import { assertRunModel, validateProjection } from '../public/projection-contract.js';

test('projection and branching require the exact recorded model artifact', () => {
  const run = { model: { manifestHash: 'sha256:one' } };
  assert.doesNotThrow(() => assertRunModel(run, 'sha256:one'));
  // The runtime reports the bare hash and a run records it with its algorithm prefix: the same identity.
  assert.doesNotThrow(() => assertRunModel({ model: { manifestHash: 'sha256:ABC123' } }, 'abc123'));
  assert.throws(() => assertRunModel({ model: { manifestHash: 'sha256:abc123' } }, 'abc124'), /manifest identity/);
  for (const hash of [null, undefined, 'sha256:two']) assert.throws(() => assertRunModel(run, hash), /manifest identity/);
  assert.throws(() => assertRunModel({ model: {} }, undefined), /manifest identity/);
});

test('projection rejects empty, malformed and out-of-range candidates before rendering', () => {
  const good = { projection: { kind: 'projected' }, rows: [{ candidates: [{ tokenId: 1, text: 'x', probability: 0.5 }] }] };
  assert.equal(validateProjection(good, 1), good);
  assert.throws(() => validateProjection(good, 2));
  for (const candidates of [[], null, [{ tokenId: 1, text: 'x', probability: NaN }], [{ tokenId: 1, text: 'x', probability: 2 }]]) {
    assert.throws(() => validateProjection({ ...good, rows: [{ candidates }] }, 1));
  }
});
