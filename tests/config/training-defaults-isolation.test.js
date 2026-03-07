import assert from 'node:assert/strict';

import {
  createTrainingConfig,
  getTrainingConfig,
  resetTrainingConfig,
  setTrainingConfig,
} from '../../src/config/training-defaults.js';

{
  const config = createTrainingConfig();
  config.training.lora.targetModules.push('unit_test_target');
  config.training.telemetry.alerts.thresholds.maxStepTimeMs = 99;
  config.training.ul.noiseSchedule.steps = 999;

  const freshConfig = createTrainingConfig();
  assert.equal(freshConfig.training.telemetry.alerts.thresholds.maxStepTimeMs, null);
  assert.equal(freshConfig.training.ul.noiseSchedule.steps, 64);
  assert.equal(freshConfig.training.lora.targetModules.includes('unit_test_target'), false);
}

{
  const current = setTrainingConfig({
    training: {
      optimizer: {
        scheduler: {
          totalSteps: 1234,
        },
      },
    },
  });
  current.training.optimizer.scheduler.totalSteps = 4321;

  const resetConfig = resetTrainingConfig();
  assert.equal(getTrainingConfig(), resetConfig);
  assert.equal(resetConfig.training.optimizer.scheduler.totalSteps, 10000);
}

console.log('training-defaults-isolation.test: ok');
