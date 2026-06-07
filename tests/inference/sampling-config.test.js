import assert from 'node:assert/strict';

import { resolveSamplingConfig } from '../../src/inference/pipelines/text/sampling-config.js';

const runtimeConfig = {
  inference: {
    sampling: {
      temperature: 0.7,
      topP: 0.95,
      topK: 40,
      repetitionPenalty: 1.0,
      repetitionPenaltyWindow: 100,
      greedyThreshold: 0.01,
    },
  },
};

assert.deepEqual(resolveSamplingConfig({}, runtimeConfig), {
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
  repetitionPenalty: 1.0,
  repetitionPenaltyWindow: 100,
  greedyThreshold: 0.01,
});

assert.deepEqual(resolveSamplingConfig({
  temperature: 0,
  topP: 1,
  topK: 1,
  repetitionPenalty: 1.2,
}, runtimeConfig), {
  temperature: 0,
  topP: 1,
  topK: 1,
  repetitionPenalty: 1.2,
  repetitionPenaltyWindow: 100,
  greedyThreshold: 0.01,
});

assert.throws(
  () => resolveSamplingConfig({}, { inference: { sampling: { ...runtimeConfig.inference.sampling, topK: undefined } } }),
  /runtimeConfig\.inference\.sampling\.topK is required/
);

assert.throws(
  () => resolveSamplingConfig({ topP: 2 }, runtimeConfig),
  /topP is outside the configured range/
);

assert.throws(
  () => resolveSamplingConfig({ topK: 1.5 }, runtimeConfig),
  /topK must be an integer/
);

console.log('sampling-config.test: ok');
