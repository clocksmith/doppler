import assert from 'node:assert/strict';

globalThis.GPUBufferUsage ??= {
  STORAGE: 1 << 0,
  COPY_DST: 1 << 1,
  COPY_SRC: 1 << 2,
  UNIFORM: 1 << 3,
};
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };

const { createDistillTripletObjective } = await import('../../src/experimental/training/objectives/distill_triplet.js');

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
      allowHintFallback: true,
    },
  },
};

await assert.rejects(
  () => objective.prepareBatch({
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
  }),
  /Distill triplet objective requires per-step positive\/negative triplet inputs/
);

await objective.prepareBatch({
  batch: {
    input: { label: 'input' },
    targets: { label: 'targets' },
    distill: {
      tripletPositivePrompts: ['positive sample'],
      tripletNegativePrompts: ['negative sample'],
    },
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
      tripletLossValues: [0.4, 0.2],
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
