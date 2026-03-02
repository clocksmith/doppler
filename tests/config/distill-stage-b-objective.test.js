import assert from 'node:assert/strict';

import { createDistillTripletObjective } from '../../src/training/objectives/distill_triplet.js';

const objective = createDistillTripletObjective({
  crossEntropyLoss: async () => ({ label: 'loss' }),
});

const config = {
  training: {
    distill: {
      enabled: true,
      stage: 'stage_b',
      stageAArtifact: '/tmp/distill_stage_a_manifest.json',
      tripletMargin: 0.2,
    },
  },
};

await objective.prepareBatch({
  batch: {
    input: { label: 'input' },
    targets: { label: 'targets' },
  },
  model: {},
  config,
  options: {
    stageAArtifactContext: {
      metricsSummary: {
        kdMean: 0.3,
        stepCount: 2,
      },
    },
  },
  lossScale: 1,
});

const lossResult = await objective.computeLoss({
  batch: {
    targets: { label: 'targets' },
    distill: {
      tripletHint: 0.3,
      tripletMargin: 0.2,
    },
  },
  config,
  tape: {},
  forwardState: { logits: {} },
  options: {
    stepIndex: 0,
    stageAArtifactContext: {
      metricsSummary: {
        kdMean: 0.3,
        stepCount: 2,
      },
    },
  },
});

assert.ok(lossResult && typeof lossResult === 'object');
assert.ok(lossResult.components && typeof lossResult.components === 'object');
assert.equal(lossResult.components.distill_stage, 'stage_b');
assert.equal(lossResult.components.loss_triplet, 0.5);

console.log('distill-stage-b-objective.test: ok');
