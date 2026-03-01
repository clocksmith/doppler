import assert from 'node:assert/strict';
import { createTrainingConfig } from '../../src/config/training-defaults.js';
import { resolveUlScheduledLambda } from '../../src/training/ul_schedule.js';

{
  const config = createTrainingConfig({
    training: {
      ul: {
        enabled: true,
        stage: 'stage1_joint',
        lambda0: 5,
        noiseSchedule: {
          type: 'log_snr_linear',
          minLogSNR: -2,
          maxLogSNR: 6,
          steps: 5,
        },
      },
    },
  });
  assert.equal(resolveUlScheduledLambda(config.training.ul, 0), 6);
  assert.equal(resolveUlScheduledLambda(config.training.ul, 4), -2);
  assert.equal(resolveUlScheduledLambda(config.training.ul, 2), 2);
}

{
  const config = createTrainingConfig({
    training: {
      ul: {
        enabled: true,
        stage: 'stage1_joint',
        lambda0: 5,
        noiseSchedule: {
          type: 'log_snr_cosine',
          minLogSNR: -4,
          maxLogSNR: 4,
          steps: 9,
        },
      },
    },
  });
  const first = resolveUlScheduledLambda(config.training.ul, 0);
  const middle = resolveUlScheduledLambda(config.training.ul, 4);
  const last = resolveUlScheduledLambda(config.training.ul, 8);
  assert.equal(first, 4);
  assert.equal(last, -4);
  assert(first > middle && middle > last);
}

console.log('ul-noise-schedule.test: ok');
