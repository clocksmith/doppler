import assert from 'node:assert/strict';
import { resolveSamplingConfig } from '../../src/inference/pipelines/text/sampling-config.js';
import { applyPresencePenalty } from '../../src/inference/pipelines/text/sampling.js';

const runtimeConfig = {
  inference: {
    sampling: {
      temperature: 0.7,
      topP: 0.95,
      topK: 40,
      repetitionPenalty: 1.0,
      presencePenalty: 0.0,
      repetitionPenaltyWindow: 100,
      greedyThreshold: 0.01,
      suppressSpecialTokens: false,
      suppressSpecialLikeTokens: false,
      suppressTokenIds: [],
    },
  },
};

// 1. Config resolution default
const resolvedDefault = resolveSamplingConfig({}, runtimeConfig);
assert.equal(resolvedDefault.presencePenalty, 0.0);

// 2. Explicit presencePenalty override
const resolvedExplicit = resolveSamplingConfig({ presencePenalty: 0.5 }, runtimeConfig);
assert.equal(resolvedExplicit.presencePenalty, 0.5);

// 3. Validation: negative numbers must throw
assert.throws(
  () => resolveSamplingConfig({ presencePenalty: -0.1 }, runtimeConfig),
  /outside the configured range/
);

// 4. Behavior: presence penalty subtracts penalty once per distinct seen token
const logits = new Float32Array([10.0, 5.0, 2.0, -1.0, 0.0]);
const previousTokens = [0, 2, 0]; // Token 0 repeated, Token 2 once, Tokens 1, 3, 4 unseen
applyPresencePenalty(logits, previousTokens, 1.5);

assert.equal(logits[0], 8.5);  // 10.0 - 1.5 = 8.5 (penalized once despite occurring twice)
assert.equal(logits[1], 5.0);  // 5.0 (unseen, untouched)
assert.equal(logits[2], 0.5);  // 2.0 - 1.5 = 0.5 (penalized once)
assert.equal(logits[3], -1.0); // -1.0 (unseen, untouched)
assert.equal(logits[4], 0.0);  // 0.0 (unseen, untouched)

// 5. Zero penalty is a no-op
const unchangedLogits = new Float32Array([5.0, 5.0]);
applyPresencePenalty(unchangedLogits, [0, 1], 0);
assert.equal(unchangedLogits[0], 5.0);
assert.equal(unchangedLogits[1], 5.0);

console.log('presence-penalty.test: ok');
