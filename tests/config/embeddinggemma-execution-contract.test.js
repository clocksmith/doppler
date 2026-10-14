import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const conversionConfig = readJson(
  'src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json'
);
const localManifest = readJson(
  'models/local/google-embeddinggemma-300m-q4k-ehf16-af32/manifest.json'
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

assert.doesNotThrow(
  () => validateRequiredInferenceFields(conversionConfig.inference, conversionConfig.output.modelBaseId),
  'EmbeddingGemma conversion config must satisfy required inference-field validation'
);
assert.doesNotThrow(
  () => validateRequiredInferenceFields(localManifest.inference, localManifest.modelId),
  'Checked-in local EmbeddingGemma manifest must satisfy required inference-field validation'
);

console.log('embeddinggemma-execution-contract.test: ok');
