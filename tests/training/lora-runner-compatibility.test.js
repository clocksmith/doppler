import assert from 'node:assert/strict';

import {
  LORA_RUNNER_BASE_MODEL_REGISTRY,
  LORA_RUNNER_DATASET_FORMAT_REGISTRY,
  LORA_RUNNER_SUPPORT_CONTRACT,
  assertLoraRunnerCompatibility,
  getLoraRunnerCompatibility,
  runLoraPipeline,
} from '../../src/experimental/training/lora-pipeline.js';

const toyWorkload = {
  kind: 'lora',
  baseModelId: 'training-toy',
  pipeline: {
    datasetFormat: 'toy-linear-classification-jsonl',
    taskType: 'classification',
  },
};

const columboWorkload = {
  kind: 'lora',
  baseModelId: 'gemma4-e2b-it',
  pipeline: {
    datasetFormat: 'text-pairs',
    taskType: 'text-generation',
  },
};

assert.deepEqual(LORA_RUNNER_SUPPORT_CONTRACT, {
  supportedBaseModelId: 'training-toy',
  supportedDatasetFormat: 'toy-linear-classification-jsonl',
  registeredBaseModelIds: [
    'training-toy',
    'gemma-3-270m-it-f16-af32',
    'gemma-3-270m-it-q4k-ehf16-af32',
    'gemma4-e2b-it',
    'gemma-4-e2b-it-q4k-ehf16-af32',
    'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    'qwen-3-5-0-8b-q4k-ehaf16',
    'qwen-3-5-2b-q4k-ehaf16',
    'qwen-3-5-9b-hf-bf16',
    'qwen-3-6-27b-q4k-ehaf16',
    'qwen-3-6-27b-q4k-eaf16',
  ],
  registeredDatasetFormats: [
    'toy-linear-classification-jsonl',
    'text-pairs',
  ],
  implementedRunnerKeys: [
    'training-toy::toy-linear-classification-jsonl::classification',
    'gemma-3-270m-it-f16-af32::text-pairs::text-generation',
    'gemma-3-270m-it-q4k-ehf16-af32::text-pairs::text-generation',
    'gemma4-e2b-it::text-pairs::text-generation',
    'gemma-4-e2b-it-q4k-ehf16-af32::text-pairs::text-generation',
    'gemma-4-e2b-it-q4k-ehf16-af32-int4ple::text-pairs::text-generation',
    'qwen-3-5-0-8b-q4k-ehaf16::text-pairs::text-generation',
    'qwen-3-5-2b-q4k-ehaf16::text-pairs::text-generation',
    'qwen-3-5-9b-hf-bf16::text-pairs::text-generation',
    'qwen-3-6-27b-q4k-ehaf16::text-pairs::text-generation',
    'qwen-3-6-27b-q4k-eaf16::text-pairs::text-generation',
  ],
});
assert.equal(LORA_RUNNER_BASE_MODEL_REGISTRY['gemma4-e2b-it'].family, 'gemma4');
assert.equal(LORA_RUNNER_BASE_MODEL_REGISTRY['gemma-3-270m-it-f16-af32'].family, 'gemma3');
assert.equal(LORA_RUNNER_BASE_MODEL_REGISTRY['gemma-3-270m-it-q4k-ehf16-af32'].family, 'gemma3');
assert.equal(LORA_RUNNER_BASE_MODEL_REGISTRY['qwen-3-5-0-8b-q4k-ehaf16'].family, 'qwen3');
assert.equal(LORA_RUNNER_BASE_MODEL_REGISTRY['qwen-3-5-0-8b-q4k-ehaf16'].requiresExternalTrainer, true);
assert.deepEqual(
  LORA_RUNNER_BASE_MODEL_REGISTRY['qwen-3-5-0-8b-q4k-ehaf16'].nativeLoraTargets,
  [{ module: 'down_proj', layer: 'last' }]
);
assert.equal(LORA_RUNNER_BASE_MODEL_REGISTRY['qwen-3-5-9b-hf-bf16'].requiresExternalTrainer, true);
assert.equal(LORA_RUNNER_DATASET_FORMAT_REGISTRY['text-pairs'].datasetKind, 'causal-lm-text-pairs');

assert.equal(getLoraRunnerCompatibility(toyWorkload).supported, true);

const compatibility = getLoraRunnerCompatibility(columboWorkload);
assert.equal(compatibility.supported, true);
assert.equal(compatibility.observed.baseModelId, 'gemma4-e2b-it');
assert.equal(compatibility.observed.datasetFormat, 'text-pairs');
assert.equal(compatibility.observed.taskType, 'text-generation');
assert.equal(compatibility.observed.registeredBaseModel, true);
assert.equal(compatibility.observed.registeredDatasetFormat, true);
assert.equal(compatibility.observed.baseModelRunnerKind, 'causal-lm-text-generation');
assert.equal(compatibility.observed.requiresExternalTrainer, true);
assert.equal(compatibility.observed.datasetKind, 'causal-lm-text-pairs');
assert.deepEqual(compatibility.blockedReasons, []);

assert.equal(assertLoraRunnerCompatibility(columboWorkload).supported, true);

await assert.rejects(
  () => runLoraPipeline({
    loadedWorkload: {
      workload: columboWorkload,
      absolutePath: 'columbo-workload.json',
      workloadSha256: 'sha256:test',
    },
  }),
  /preflightCausalLmLoraWorkload requires workload\.datasetPath/
);

const unknownCompatibility = getLoraRunnerCompatibility({
  kind: 'lora',
  baseModelId: 'unknown-base',
  pipeline: {
    datasetFormat: 'unknown-format',
    taskType: 'text-generation',
  },
});
assert.deepEqual(unknownCompatibility.blockedReasons, [
  'base-model-not-registered-for-current-lora-runner',
  'dataset-format-not-supported-by-current-lora-runner',
]);

console.log('lora-runner-compatibility.test: ok');
