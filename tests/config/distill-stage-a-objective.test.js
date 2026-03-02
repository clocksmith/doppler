import assert from 'node:assert/strict';

import { createDistillKdObjective } from '../../src/training/objectives/distill_kd.js';

const objective = createDistillKdObjective({
  crossEntropyLoss: async () => ({ label: 'loss' }),
});

const config = {
  training: {
    distill: {
      enabled: true,
      stage: 'stage_a',
      temperature: 2,
      alphaKd: 3,
      alphaCe: 0,
    },
  },
};

const lossResult = await objective.computeLoss({
  batch: {
    targets: { label: 'targets' },
    distill: {
      teacherLossHint: 0.5,
      temperature: 2,
      alphaKd: 3,
    },
  },
  config,
  tape: {},
  forwardState: { logits: {} },
  options: { stepIndex: 0 },
});

assert.ok(lossResult && typeof lossResult === 'object');
assert.ok(lossResult.components && typeof lossResult.components === 'object');
assert.equal(lossResult.components.distill_stage, 'stage_a');
assert.equal(lossResult.components.loss_kd, 0.75);

console.log('distill-stage-a-objective.test: ok');
