import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

for (const relativePath of [
  'src/inference/pipelines/text/logits/index.js',
  'src/inference/pipelines/text/logits/gpu.js',
]) {
  const source = await fs.readFile(path.join(repoRoot, relativePath), 'utf8');
  assert.match(
    source,
    /finalLogitSoftcapping\)\s*&&\s*config\.finalLogitSoftcapping\s*>\s*0/,
    `${relativePath} must force the logits tail onto f32 for softcapped f16 outputs.`
  );
  assert.match(
    source,
    /matmul_gemv_subgroup_f16a\.wgsl'.*matmul_gemv_subgroup\.wgsl/s,
    `${relativePath} must swap LM-head decode onto the stable f32 kernel variant.`
  );
}

console.log('logits-softcap-f32-tail-contract.test: ok');
