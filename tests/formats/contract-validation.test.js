import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { hashSortedJson } from '../../src/formats/sorted-json.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { requireObject, requireExactKeys, requireString, requireDigest } from '../../src/formats/contract-validation.js';

// Numeric property keys and ASCII case ordering are part of existing artifact
// identities. The alternate canonical encoding must not silently replace them.
const value = { b: undefined, a: { z: 1, A: 2 }, 10: 'ten', 2: 'two' };
const encoding = '{"2":"two","10":"ten","a":{"A":2,"z":1}}';
assert.equal(hashSortedJson(value), `sha256:${createHash('sha256').update(encoding).digest('hex')}`);
assert.notEqual(hashSortedJson(value), computeCanonicalSha256(value));
assert.equal(hashSortedJson({ b: 2, a: 1 }), hashSortedJson({ a: 1, b: 2 }));

const errors = [];
assert.equal(requireObject([], 'record', errors), false);
assert.equal(requireObject(Object.create(null), 'record', errors), true);
requireExactKeys({ allowed: 1, extra: 2 }, new Set(['allowed']), 'record', errors);
requireExactKeys(null, new Set(), 'record', errors);
requireString('  ', 'name', errors);
requireString(' valid ', 'name', errors);
requireDigest('sha256:bad', 'digest', errors);
requireDigest(`sha256:${'a'.repeat(64)}`, 'digest', errors);
assert.deepEqual(errors, [
  'record must be an object.',
  'record.extra is not allowed.',
  'name must be a non-empty string.',
  'digest must be a SHA-256 digest.',
]);
console.log('contract-validation.test: ok');
