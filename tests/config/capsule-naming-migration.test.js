import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';

const archive = 'tests/fixtures/pre-capsule';
const manifest = JSON.parse(await fs.readFile(`${archive}/manifest.json`, 'utf8'));
// This is an explicit test projection, never a runtime compatibility adapter.
function renameIdentityFields(value) {
  if (Array.isArray(value)) return value.map(renameIdentityFields);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, child]) => [
    key === 'weightPackId' ? 'weightCapsuleId' : key === 'weightPackHash' ? 'weightCapsuleHash' : key,
    renameIdentityFields(child),
  ]));
}

for (const entry of manifest.files) {
  const bytes = await fs.readFile(`${archive}/${entry.source}`);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256, entry.source);
  if (!entry.source.startsWith('src/config/conversion/')) continue;
  const current = JSON.parse(await fs.readFile(entry.source, 'utf8'));
  const expected = renameIdentityFields(JSON.parse(bytes));
  if (entry.source === 'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af16-int4ple.json') {
    // The referenced manifest now uses Capsule identity fields. Bind the new
    // digest to its actual bytes while comparing every computational field.
    const base = await fs.readFile('models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json');
    expected.manifest.weightsRef.manifestDigest = `sha256:${createHash('sha256').update(base).digest('hex')}`;
  }
  assert.deepEqual(current, expected,
    `${entry.source}: naming must not change model computation or numerical policy`);
}
console.log('capsule-naming-migration.test: frozen evidence and unchanged conversion semantics passed');
