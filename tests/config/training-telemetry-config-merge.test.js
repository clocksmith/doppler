import assert from 'node:assert/strict';

import { createTrainingConfig } from '../../src/config/training-defaults.js';

const config = createTrainingConfig({
  training: {
    telemetry: {
      alerts: {
        enabled: true,
        thresholds: {
          maxStepTimeMs: 12,
        },
      },
    },
  },
});

assert.equal(config.training.telemetry.alerts.enabled, true);
assert.equal(config.training.telemetry.alerts.failOnAlert, false);
assert.equal(config.training.telemetry.alerts.thresholds.maxStepTimeMs, 12);
assert.equal(config.training.telemetry.alerts.thresholds.maxGradientNorm, null);
assert.equal(config.training.telemetry.alerts.thresholds.maxNaNCount, null);

console.log('training-telemetry-config-merge.test: ok');
