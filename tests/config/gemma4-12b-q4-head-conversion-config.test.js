import assert from 'node:assert/strict';
import fs from 'node:fs';

import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';
import { selectRuleValue } from '../../src/rules/rule-registry.js';

const BASE_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af16';
const Q4_HEAD_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-hq4k-af16';
const W4A16_QAT_MODEL_ID = 'gemma-4-12b-it-text-w4a16-ct-ehf16-af16';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function postLayerStep(execution, op) {
  return execution?.postLayer?.find((step) => Array.isArray(step) && step[0] === op) ?? null;
}

const baseConfig = readJson(`src/config/conversion/gemma4/${BASE_MODEL_ID}.json`);
const q4HeadConfig = readJson(`src/config/conversion/gemma4/${Q4_HEAD_MODEL_ID}.json`);
const w4a16QatConfig = readJson(`src/config/conversion/gemma4/${W4A16_QAT_MODEL_ID}.json`);

assert.notEqual(
  q4HeadConfig.manifest?.artifactIdentity?.weightCapsuleId,
  baseConfig.manifest?.artifactIdentity?.weightCapsuleId,
  'Q4-head lane must produce a new weight capsule because conversion appends synthetic tied lm_head bytes'
);

assert.equal(q4HeadConfig.quantization?.lmHead, 'q4k');
const q4HeadLmHeadKernel = q4HeadConfig.execution?.kernels?.[postLayerStep(q4HeadConfig.execution, 'lm_head')?.[1]]?.kernel;
assert.match(
  q4HeadLmHeadKernel ?? '',
  /q4/,
  'Q4-head lane must decode through a packed Q4 tied-head kernel'
);

assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    hasGate: true,
    hasUp: true,
    hasDown: true,
    hasFusedWeights: false,
    inputIsSupported: true,
    hasLoRA: false,
    dtypeMatches: true,
    dtypeSupported: true,
    weightDtype: 'q4k',
    activationDtype: 'f16',
    hiddenSizeAligned32: true,
    batchSize: 4,
    useDoubleWideMlp: false,
  }),
  true,
  'Q4K f16 activation decode should use the existing fused gate/up kernel rule'
);

assert.doesNotThrow(
  () => validateRequiredInferenceFields(q4HeadConfig.inference, Q4_HEAD_MODEL_ID),
  'Q4-head conversion config must satisfy required inference fields'
);

// A non-Q4K weight lane must not dispatch packed Q4 kernels.
const qatExecutionText = JSON.stringify(w4a16QatConfig.execution);
assert.equal(qatExecutionText.includes('q4_decode_gemv'), false);
assert.equal(qatExecutionText.includes('q4_widetile'), false);
assert.equal(qatExecutionText.includes('lm_head_q4'), false);
assert.doesNotThrow(
  () => validateRequiredInferenceFields(w4a16QatConfig.inference, W4A16_QAT_MODEL_ID),
  'W4A16 QAT conversion config must satisfy required inference fields'
);

console.log('gemma4-12b-q4-head-conversion-config.test: ok');
