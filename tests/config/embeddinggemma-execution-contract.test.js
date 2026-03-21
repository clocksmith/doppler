import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const kernelPathFiles = [
  'src/config/kernel-paths/embeddinggemma-f16-f32a.json',
  'src/config/kernel-paths/embeddinggemma-f32-f32a.json',
  'src/config/kernel-paths/embeddinggemma-q4k-dequant-f32a.json',
];

for (const relativePath of kernelPathFiles) {
  const kernelPath = readJson(relativePath);
  assert.deepEqual(
    kernelPath.postLayer,
    [{ op: 'final_norm', kernel: 'rmsnorm.wgsl', entry: 'main' }],
    `${relativePath} must expose only final_norm in postLayer`
  );
  assert.deepEqual(kernelPath.sampling, [], `${relativePath} must not advertise sampling ops`);
}

const conversionConfig = readJson(
  'src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json'
);

assert.deepEqual(
  conversionConfig.execution.postLayer,
  [['final_norm', 'rmsnorm']],
  'EmbeddingGemma conversion config must stamp only final_norm in postLayer'
);
assert.ok(
  !Object.hasOwn(conversionConfig.execution.kernels, 'lm_head_tiled'),
  'EmbeddingGemma conversion config must not stamp lm_head_tiled kernel metadata'
);
assert.ok(
  !Object.hasOwn(conversionConfig.execution.kernels, 'sample'),
  'EmbeddingGemma conversion config must not stamp sample kernel metadata'
);

console.log('embeddinggemma-execution-contract.test: ok');
