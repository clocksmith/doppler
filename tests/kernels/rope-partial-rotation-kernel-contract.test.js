import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '../..');

for (const relPath of ['src/gpu/kernels/rope.wgsl', 'src/gpu/kernels/rope_f16.wgsl']) {
  const source = readFileSync(join(rootDir, relPath), 'utf8');
  const helper = source.match(/fn get_second_rotary_idx[\s\S]*?\n\}/)?.[0] ?? '';

  assert.match(helper, /pair_span_dim \/ 2u/);
  assert.doesNotMatch(helper, /rotary_dim \/ 2u/);
  assert.doesNotMatch(helper, /head_dim \/ 2u/);
}

console.log('rope-partial-rotation-kernel-contract.test: ok');
