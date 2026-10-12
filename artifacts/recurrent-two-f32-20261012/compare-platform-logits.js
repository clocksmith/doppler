// Agreement is a portability observation, never an independent accuracy target.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const [firstPath, secondPath, output] = process.argv.slice(2);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const aBytes = await readFile(firstPath), bBytes = await readFile(secondPath);
const a = JSON.parse(aBytes), b = JSON.parse(bBytes);
for (const field of ['archiveSha256', 'referenceSha256', 'manifestIdentity', 'recurrentStatePrecision']) {
  assert.equal(a[field], b[field]);
}
for (const record of [a, b]) {
  assert.equal(record.executionCompleted, true); assert.equal(record.fullPrefixes, true);
  assert.deepEqual(record.declaredShaderMismatches, []); assert.equal(record.results.length, 57);
}
const rows = a.results.map((row, index) => {
  const other = b.results[index];
  assert.equal(row.index, other.index); assert.equal(row.step, other.step);
  assert.deepEqual(row.inputIds, other.inputIds); assert.equal(row.logits, other.logits);
  return { index: row.index, step: row.step, sha256: hash(Buffer.from(row.logits, 'base64')) };
});
assert.equal(new Set(rows.map(row => `${row.index}/${row.step}`)).size, 55);
await writeFile(output, JSON.stringify({ qualified: false, archiveSha256: a.archiveSha256,
  scope: 'Exact 55-prefix and two resident-reuse logits equality; not independent numerical acceptance',
  firstSha256: hash(aBytes), secondSha256: hash(bBytes), rows }, null, 2) + '\n');
