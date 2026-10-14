import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const modelId = 'qwen-3-embedding-0-6b-q4k-ehf16-af32';
const config = readJson(`src/config/conversion/qwen3/${modelId}.json`);

if (config.modelType === 'embedding') {
  // Embedding artifacts expose hidden states; they must not bind logits or sampling.
  const postLayerOps = (config.execution?.postLayer ?? []).map((step) => step[0]);
  for (const op of ['lm_head', 'lm_head_prefill', 'sample']) {
    assert.ok(!postLayerOps.includes(op), `${modelId} must not run ${op} after the final norm`);
  }
  for (const kernelKey of ['lm_head_q4', 'lm_head_gemv', 'sample']) {
    assert.ok(!Object.hasOwn(config.execution?.kernels ?? {}, kernelKey), `${modelId} must not declare ${kernelKey}`);
  }
}

assert.doesNotThrow(
  () => validateRequiredInferenceFields(config.inference, modelId),
  `${modelId} conversion config must satisfy required inference-field validation`
);

console.log('qwen3-embedding-onboarding-contract.test: ok');
