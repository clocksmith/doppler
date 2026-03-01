import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readPreset(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(absolutePath, 'utf8'));
}

{
  const gemma2 = await readPreset('src/config/presets/models/gemma2.json');
  const q4k = gemma2?.inference?.kernelPaths?.q4k;
  assert.ok(q4k, 'gemma2 q4k kernelPaths must exist');
  assert.equal(q4k.f16, 'gemma2-q4k-dequant-f32a');
  assert.equal(q4k.f32, 'gemma2-q4k-dequant-f32a');
}

{
  const gemma3 = await readPreset('src/config/presets/models/gemma3.json');
  const q4k = gemma3?.inference?.kernelPaths?.q4k;
  assert.ok(q4k, 'gemma3 q4k kernelPaths must exist');
  assert.equal(q4k.default, 'gemma3-q4k-dequant-f32a-online');
  assert.equal(q4k.f16, 'gemma3-q4k-dequant-f16a-online');
  assert.equal(q4k.f16a, 'gemma3-q4k-dequant-f16a-online');
  assert.equal(q4k.f32, 'gemma3-q4k-dequant-f32a-online');
}

console.log('model-q4k-defaults-portable.test: ok');
