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

assert.equal(conversionConfig.inference?.attention?.valueNorm, false);
assert.equal(conversionConfig.inference?.ffn?.useDoubleWideMlp, false);
assert.equal(conversionConfig.inference?.rope?.ropeLocalPartialRotaryFactor, null);
assert.equal(conversionConfig.inference?.rope?.ropeFrequencyBaseDim, null);
assert.equal(conversionConfig.inference?.rope?.ropeLocalFrequencyBaseDim, null);
assert.deepEqual(conversionConfig.session, {
  compute: {
    defaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
  },
  kvcache: {
    kvDtype: 'f32',
    layout: 'contiguous',
    pageSize: 256,
    tiering: { mode: 'off' },
  },
  decodeLoop: {
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  },
});
assert.deepEqual(localManifest.inference?.session, conversionConfig.session);

assert.doesNotThrow(
  () => validateRequiredInferenceFields(conversionConfig.inference, conversionConfig.output.modelBaseId),
  'EmbeddingGemma conversion config must satisfy required inference-field validation'
);
assert.doesNotThrow(
  () => validateRequiredInferenceFields(localManifest.inference, localManifest.modelId),
  'Checked-in local EmbeddingGemma manifest must satisfy required inference-field validation'
);

console.log('embeddinggemma-execution-contract.test: ok');
