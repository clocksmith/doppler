import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { isPlainObject } from '../../src/formats/plain-object.js';
import { validateExactKeys } from '../../tools/lib/json-object-validation.js';
import { canonicalizeJson, computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import {
  canonicalizeJson as canonicalizeEvidenceJson,
  computeCanonicalJsonSha256,
} from '../../tools/lib/canonical-json.js';

for (const value of [null, undefined, false, 0, '', [], () => {}]) {
  assert.equal(isPlainObject(value), false);
  const errors = [];
  assert.equal(validateExactKeys(value, ['required'], 'record', errors), false);
  assert.deepEqual(errors, ['record must be an object']);
}
// Preserve the existing object-record predicate, including non-plain prototypes.
for (const value of [{}, Object.create(null), new Date(), new Map(), new class {}()]) {
  assert.equal(isPlainObject(value), true);
}
{
  const value = Object.assign(Object.create({ inherited: 1 }), { extraB: 2, extraA: 1 });
  const errors = [];
  assert.equal(validateExactKeys(value, ['required', 'inherited'], 'record', errors), true);
  assert.deepEqual(errors, [
    'record.extraB is not supported', 'record.extraA is not supported',
    'record.required is required', 'record.inherited is required',
  ]);
  const acceptedErrors = [];
  assert.equal(validateExactKeys({ required: undefined }, ['required'], 'record', acceptedErrors), true);
  assert.deepEqual(acceptedErrors, []); // Value validation belongs to the field checker.
}

// These established byte contracts are intentionally different. Consolidating
// their algorithms would rewrite signed Pack/evidence identities.
const value = { B: 1, a: 2 };
assert.equal(canonicalizeJson(value), '{"a":2,"B":1}');
assert.equal(computeCanonicalSha256(value), 'sha256:f4fb4f23c18557f189581b45e08078afb6b588addef57f8b1b4c80cef4a8716d');
assert.equal(canonicalizeEvidenceJson(value), '{"B":1,"a":2}');
assert.equal(computeCanonicalJsonSha256(value), 'sha256:812e5e7fb7bb816dc477e91a136430192eadcf83ff303881298146e106ae0161');
const nested = { z: [{ B: 1, a: 2 }, null], A: false };
assert.equal(computeCanonicalSha256(nested), 'sha256:7bc70502ebab234fabaf679c9d3882aca7d8b55523d13bc713a349ad95b2f550');
assert.equal(computeCanonicalJsonSha256(nested), 'sha256:2272b1ea5e54112e24f431ce2b1d53d724d4ebfddef18af29182c4b91a2f506f');
assert.equal(canonicalizeJson({ missing: undefined, values: [undefined] }), '{"values":[null]}');
assert.equal(canonicalizeEvidenceJson({ missing: undefined, values: [undefined] }), '{"missing":undefined,"values":[]}');
assert.throws(() => canonicalizeJson({ number: Infinity }), /non-finite/);

// Inventory only the checker/recorder boundary. Frozen experimental tools may
// retain historical implementations where their exact source is receipt-bound.
const files = execFileSync('git', ['ls-files', 'tools/check-*.js', 'tools/record-*.js'], { encoding: 'utf8' })
  .trim().split('\n');
for (const file of files) {
  assert.doesNotMatch(readFileSync(file, 'utf8'), /function isPlainObject\(value\)/, file);
}
console.log('evidence-json-contracts.test: ok');
