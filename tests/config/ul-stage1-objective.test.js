import assert from 'node:assert/strict';

import { createUlStage1JointObjective } from '../../src/training/objectives/ul_stage1_joint.js';

const objective = createUlStage1JointObjective({
  crossEntropyLoss: async () => ({ label: 'loss' }),
});

const config = {
  training: {
    ul: {
      enabled: true,
      lambda0: 0,
      priorAlignment: { enabled: true, weight: 2 },
      decoderSigmoidWeight: { enabled: true, slope: 1, midpoint: 0 },
      lossWeights: {
        ce: 1,
        prior: 3,
        decoder: 5,
        recon: 4,
      },
    },
  },
};

const batch = {
  targets: { label: 'targets' },
  ul: {
    lambda: 0,
    clean: { mean: 0, std: 0 },
    noise: { mean: 0, std: 0.2 },
    noisy: { mean: 1, std: 0 },
    stepIndex: 0,
    shape: [1, 2],
    values: {
      clean: [0, 0],
      noise: [1, -1],
      noisy: [1, 1],
    },
  },
};

const lossResult = await objective.computeLoss({
  model: { forward: async () => ({}) },
  batch,
  config,
  tape: {},
  forwardState: { logits: {} },
  options: { stepIndex: 0 },
});

assert.ok(lossResult && typeof lossResult === 'object');
assert.ok(lossResult.components && typeof lossResult.components === 'object');
assert.equal(lossResult.components.coeff_prior, 6);
assert.equal(lossResult.components.coeff_recon, 4);
assert.equal(lossResult.components.coeff_decoder, 2.5);
assert.equal(lossResult.components.loss_prior, 6);
assert.equal(lossResult.components.loss_recon, 4);
assert.equal(lossResult.components.loss_decoder, 3);
assert.equal(lossResult.components.loss_total, 13);

console.log('ul-stage1-objective.test: ok');
