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
  assert.deepEqual(current, renameIdentityFields(JSON.parse(bytes)),
    `${entry.source}: naming must not change model computation or numerical policy`);
}
console.log('capsule-naming-migration.test: frozen evidence and unchanged conversion semantics passed');
