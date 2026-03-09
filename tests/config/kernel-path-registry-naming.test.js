import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const registryPath = path.join(repoRoot, 'src/config/presets/kernel-paths/registry.json');

const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
const entries = Array.isArray(registry?.entries) ? registry.entries : [];
const byId = new Map(entries.map((entry) => [entry.id, entry]));

function expectLegacyAlias(id, targetId) {
  const entry = byId.get(id);
  assert.ok(entry, `missing registry entry ${id}`);
  assert.equal(entry.status, 'legacy', `${id} must be marked legacy`);
  assert.equal(entry.aliasOf, targetId, `${id} must alias ${targetId}`);
}

function expectCanonicalNoSubgroups(id) {
  const entry = byId.get(id);
  assert.ok(entry, `missing registry entry ${id}`);
  assert.equal(entry.status, 'canonical', `${id} must be canonical`);
  assert.match(String(entry.notes ?? ''), /no subgroup requirement|subgroup-free/i, `${id} notes must mention no-subgroups behavior`);
  assert.match(String(entry.notes ?? ''), /shader-f16/i, `${id} notes must mention shader-f16 requirement`);
}

expectCanonicalNoSubgroups('gemma2-q4k-dequant-f32a-nosubgroups');
expectCanonicalNoSubgroups('gemma3-q4k-dequant-f32a-nosubgroups');
expectLegacyAlias('gemma2-q4k-dequant-f32a', 'gemma2-q4k-dequant-f32a-nosubgroups');
expectLegacyAlias('gemma3-q4k-dequant-f32a', 'gemma3-q4k-dequant-f32a-nosubgroups');

console.log('kernel-path-registry-naming.test: ok');
