import assert from 'node:assert/strict';

const { TrainingRunner } = await import('../../src/training/runner.js');

const runner = new TrainingRunner(
  {
    training: {
      lossScaling: {
        enabled: false,
        mode: 'off',
        initialScale: 1,
        growthFactor: 2,
        backoffFactor: 0.5,
        growthInterval: 1,
        overflowCheck: false,
      },
    },
  },
  {
    optimizer: { stepCount: 0 },
    crossEntropyLoss() {
      throw new Error('crossEntropyLoss should not run for empty datasets');
    },
    clipGradients() {
      throw new Error('clipGradients should not run for empty datasets');
    },
  }
);

runner.lastCheckpoint = {
  key: 'stale',
  step: 12,
};
runner.lastArtifact = {
  path: 'stale-artifact.json',
};

const metrics = await runner.run({}, [], {
  epochs: 1,
  batchSize: 1,
  shuffle: false,
});

assert.deepEqual(metrics, []);
assert.equal(runner.lastCheckpoint, null);
assert.equal(runner.lastArtifact, null);

console.log('runner-state-cleanup.test: ok');
