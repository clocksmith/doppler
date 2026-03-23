import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const conversionConfig = readJson(
  'src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json'
);

assert.equal(
  conversionConfig.modelType,
  'embedding',
  'EmbeddingGemma conversion config must stamp modelType="embedding"'
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
assert.ok(
  !Object.hasOwn(conversionConfig.execution.kernels, 'lm_head_gemv'),
  'EmbeddingGemma conversion config must not stamp lm_head_gemv kernel metadata'
);

console.log('embeddinggemma-execution-contract.test: ok');
