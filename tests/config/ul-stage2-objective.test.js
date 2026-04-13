import assert from 'node:assert/strict';

import { createUlStage2BaseObjective } from '../../src/experimental/training/objectives/ul_stage2_base.js';

const objective = createUlStage2BaseObjective({
  crossEntropyLoss: async () => ({ label: 'loss' }),
});

const config = {
  training: {
    ul: {
      enabled: true,
      stage: 'stage2_base',
      stage1Artifact: '/tmp/stage1.json',
      lambda0: 5,
      decoderSigmoidWeight: {
        enabled: true,
        slope: 1,
        midpoint: 5,
      },
      lossWeights: {
        ce: 1,
        prior: 2,
        decoder: 3,
        recon: 4,
      },
    },
  },
};

const stage1Entry = {
  lambda: 5,
  latent_shape: [2],
  latent_clean_mean: 0,
  latent_clean_std: 0.2,
  latent_noisy_mean: 0.3,
  latent_noisy_std: 0.5,
  latent_noise_std: 0.1,
};

const lossResult = await objective.computeLoss({
  batch: {
    targets: {},
    input: { shape: [2] },
    ulStage2: {
      stage1Entry,
      latentMean: 0.1,
      latentStd: 0.4,
    },
  },
  config,
  tape: {},
  forwardState: { logits: {} },
  options: {
    stage1ArtifactContext: {
      latentDataset: {
        entries: [stage1Entry, stage1Entry],
        summary: {
          lambdaMean: 5,
        },
      },
    },
  },
});

assert.ok(lossResult && typeof lossResult === 'object');
assert.ok(lossResult.components && typeof lossResult.components === 'object');
assert.ok(lossResult.components.loss_total > 0);
assert.ok(lossResult.components.loss_prior > 0);
assert.ok(lossResult.components.loss_decoder > 0);
assert.ok(lossResult.components.loss_recon > 0);
assert.equal(lossResult.components.stage1_latent_count, 2);

console.log('ul-stage2-objective.test: ok');
