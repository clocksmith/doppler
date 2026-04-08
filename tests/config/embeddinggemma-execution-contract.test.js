import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const expectedEmbeddingPostprocessor = {
  poolingMode: 'mean',
  includePrompt: true,
  projections: [
    {
      weightTensor: 'embedding_postprocessor.projections.0.weight',
      biasTensor: null,
      inputSize: 768,
      outputSize: 3072,
      activation: 'identity',
    },
    {
      weightTensor: 'embedding_postprocessor.projections.1.weight',
      biasTensor: null,
      inputSize: 3072,
      outputSize: 768,
      activation: 'identity',
    },
  ],
  normalize: 'l2',
};

const expectedSession = {
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
    tiering: {
      mode: 'off',
    },
  },
  decodeLoop: {
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    readbackMode: 'sequential',
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  },
};

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
assert.deepEqual(conversionConfig.inference?.output?.embeddingPostprocessor, expectedEmbeddingPostprocessor);
assert.deepEqual(localManifest.inference?.output?.embeddingPostprocessor, expectedEmbeddingPostprocessor);
assert.deepEqual(conversionConfig.session, expectedSession);
assert.deepEqual(localManifest.inference?.session, expectedSession);

assert.doesNotThrow(
  () => validateRequiredInferenceFields(conversionConfig.inference, conversionConfig.output.modelBaseId),
  'EmbeddingGemma conversion config must satisfy required inference-field validation'
);
assert.doesNotThrow(
  () => validateRequiredInferenceFields(localManifest.inference, localManifest.modelId),
  'Checked-in local EmbeddingGemma manifest must satisfy required inference-field validation'
);

console.log('embeddinggemma-execution-contract.test: ok');
