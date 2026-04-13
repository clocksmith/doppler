import assert from 'node:assert/strict';

globalThis.GPUBufferUsage ??= {
  STORAGE: 1 << 0,
  COPY_DST: 1 << 1,
  COPY_SRC: 1 << 2,
  UNIFORM: 1 << 3,
};
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };

const { createDistillKdObjective } = await import('../../src/experimental/training/objectives/distill_kd.js');

const objective = createDistillKdObjective({
  crossEntropyLoss: async () => ({ label: 'loss' }),
});

const config = {
  training: {
    distill: {
      enabled: true,
      stage: 'stage_a',
      temperature: 1,
      alphaKd: 1,
      alphaCe: 0.5,
      allowHintFallback: true,
    },
  },
};

const lossResult = await objective.computeLoss({
  batch: {
    targets: { label: 'targets' },
    distill: {
      teacherTopProbs: [[1, 0]],
      studentTopProbs: [[0.5, 0.5]],
      teacherTargetIndices: [0],
      temperature: 1,
      alphaKd: 1,
      alphaCe: 0.5,
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
assert.ok(Math.abs(lossResult.components.loss_kd - Math.log(2)) < 1e-6);
assert.ok(Math.abs(lossResult.components.distill_loss_ce_aux - (0.5 * Math.log(2))) < 1e-6);

console.log('distill-stage-a-objective.test: ok');
