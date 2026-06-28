import assert from 'node:assert/strict';

import { loadTrainingWorkloadPack } from '../../src/experimental/training/workloads.js';

const distill = await loadTrainingWorkloadPack(
  'src/experimental/training/workload-packs/distill-translategemma-tiny.json'
);
assert.equal(distill.workload.kind, 'distill');
assert.equal(distill.workload.evalDatasets[0].decodePolicy?.maxTokens, 24);
assert.equal(distill.workload.pipeline.stagePlan.length, 2);

const lora = await loadTrainingWorkloadPack(
  'src/experimental/training/workload-packs/lora-toy-tiny.json'
);
assert.equal(lora.workload.kind, 'lora');
assert.equal(lora.workload.pipeline.datasetFormat, 'toy_linear_classification_jsonl');
assert.equal(lora.workload.pipeline.adapter.targetModules[0], 'q_proj');

const codeLora = await loadTrainingWorkloadPack('lora-doppler-code-agent-tiny');
assert.equal(codeLora.workload.kind, 'lora');
assert.equal(codeLora.workload.pipeline.datasetFormat, 'text-pairs');
assert.equal(codeLora.workload.pipeline.taskType, 'text_generation');

const codeLoraF16 = await loadTrainingWorkloadPack('lora-doppler-code-agent-gemma270m-f16-tiny');
assert.equal(codeLoraF16.workload.kind, 'lora');
assert.equal(codeLoraF16.workload.baseModelId, 'gemma-3-270m-it-f16-af32');
assert.equal(codeLoraF16.workload.pipeline.datasetFormat, 'text-pairs');
assert.equal(codeLoraF16.workload.pipeline.taskType, 'text_generation');

const sftDistill = await loadTrainingWorkloadPack('distill-glm52-doppler-code-agent-sft-tiny');
assert.equal(sftDistill.workload.kind, 'distill');
assert.equal(sftDistill.workload.pipeline.stagePlan[0].objective, 'sft');
assert.equal(sftDistill.workload.pipeline.sftLora.datasetFormat, 'text-pairs');
assert.equal(sftDistill.workload.pipeline.sftLora.adapter.targetModules.includes('q_proj'), true);

const sftDistillF16 = await loadTrainingWorkloadPack('distill-glm52-doppler-code-agent-gemma270m-f16-sft-tiny');
assert.equal(sftDistillF16.workload.kind, 'distill');
assert.equal(sftDistillF16.workload.baseModelId, 'gemma-3-270m-it-f16-af32');
assert.equal(sftDistillF16.workload.pipeline.stagePlan[0].objective, 'sft');
assert.equal(sftDistillF16.workload.pipeline.sftLora.datasetFormat, 'text-pairs');
assert.equal(sftDistillF16.workload.pipeline.sftLora.adapter.targetModules.includes('v_proj'), true);

for (const [workloadId, datasetId] of [
  ['lora-doppler-js-json-gemma270m-f16-tiny', 'doppler-js-json-sft-tiny'],
  ['lora-doppler-wgsl-gemma270m-f16-tiny', 'doppler-wgsl-sft-tiny'],
  ['lora-doppler-review-gemma270m-f16-tiny', 'doppler-review-sft-tiny'],
  ['lora-doppler-agent-harness-gemma270m-f16-tiny', 'doppler-agent-harness-sft-tiny'],
]) {
  const workload = await loadTrainingWorkloadPack(workloadId);
  assert.equal(workload.workload.kind, 'lora');
  assert.equal(workload.workload.baseModelId, 'gemma-3-270m-it-f16-af32');
  assert.equal(workload.workload.datasetId, datasetId);
  assert.equal(workload.workload.pipeline.datasetFormat, 'text-pairs');
}

for (const [workloadId, datasetId] of [
  ['distill-glm52-doppler-js-json-gemma270m-f16-sft-tiny', 'doppler-js-json-sft-tiny'],
  ['distill-glm52-doppler-wgsl-gemma270m-f16-sft-tiny', 'doppler-wgsl-sft-tiny'],
  ['distill-glm52-doppler-review-gemma270m-f16-sft-tiny', 'doppler-review-sft-tiny'],
  ['distill-glm52-doppler-agent-harness-gemma270m-f16-sft-tiny', 'doppler-agent-harness-sft-tiny'],
]) {
  const workload = await loadTrainingWorkloadPack(workloadId);
  assert.equal(workload.workload.kind, 'distill');
  assert.equal(workload.workload.teacherModelId, 'zai-org/GLM-5.2');
  assert.equal(workload.workload.datasetId, datasetId);
  assert.equal(workload.workload.pipeline.sftLora.datasetFormat, 'text-pairs');
}

console.log('training-workload-loader-v2.test: ok');
