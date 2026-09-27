import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { decodeQ4KBlockReference } from '../../tools/lib/q4k-projection-reference.js';
const fixture = JSON.parse(await fs.readFile(new URL('../fixtures/qwen-q4k-independent-blocks.json', import.meta.url)));
for (const row of fixture.rows) {
  const bytes = new Uint8Array(Buffer.from(row.packedHex, 'hex'));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), row.packedSha256);
  assert.deepEqual(Array.from(decodeQ4KBlockReference(bytes).values), row.decoded, row.tensor);
}
assert.throws(() => decodeQ4KBlockReference(new Uint8Array(143)), /complete/);
