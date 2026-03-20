import assert from 'node:assert/strict';

import { loadTrainingWorkloadPack } from '../../src/training/workloads.js';

const distill = await loadTrainingWorkloadPack(
  'src/training/workload-packs/distill-translategemma-tiny.json'
);
assert.equal(distill.workload.kind, 'distill');
assert.equal(distill.workload.evalDatasets[0].decodePolicy?.maxTokens, 24);
assert.equal(distill.workload.pipeline.stagePlan.length, 2);

const lora = await loadTrainingWorkloadPack(
  'src/training/workload-packs/lora-toy-tiny.json'
);
assert.equal(lora.workload.kind, 'lora');
assert.equal(lora.workload.pipeline.datasetFormat, 'toy_linear_classification_jsonl');
assert.equal(lora.workload.pipeline.adapter.targetModules[0], 'q_proj');

console.log('training-workload-loader-v2.test: ok');
