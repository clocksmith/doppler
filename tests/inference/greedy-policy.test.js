import assert from 'node:assert/strict';
import { isGreedySampling } from '../../src/rules/greedy-sampling.js';
import { shouldUseGreedyLmHeadArgmaxFusion } from '../../src/inference/pipelines/text/generator/greedy-head-policy.js';

for (const [temperature, topK, threshold, expected] of [
  [0, 0, 0, true], [0, 40, 0, true], [0.8, 1, 0, true],
  [0.001, 40, 0.01, true], [0.01, 40, 0.01, false],
  [0.8, 0, 0, false], [0.8, 40, 0.01, false],
]) assert.equal(isGreedySampling(temperature, topK, threshold), expected);
for (const args of [[-1, 1, 0], [NaN, 1, 0], [0, 1.5, 0], [0, -1, 0], [0, 1, -1], [0, 1, Infinity]]) {
  assert.throws(() => isGreedySampling(...args));
}

const state = {
  runtimeConfig: { inference: { session: { useGreedyLmHeadArgmaxFusion: true } }, shared: { debug: { probes: [] } } },
  modelConfig: { logitOutputScale: 1 }, operatorDiagnostics: null,
};
const opts = { temperature: 0, topK: 1, presencePenalty: 0, suppressTokenIds: [] };
const defaults = { greedyThreshold: 0 };
const allowed = (s = state, o = opts, penalty = 1) => shouldUseGreedyLmHeadArgmaxFusion(s, o, defaults, penalty);
assert.equal(allowed(), true, 'zero threshold must not disable an explicitly greedy fused head');
assert.equal(allowed(state, { ...opts, temperature: 0.8 }), true, 'topK1 remains deterministic');
assert.equal(allowed(state, { ...opts, temperature: 0.8, topK: 40 }), false);
assert.equal(allowed(state, opts, 1.1), false);
assert.equal(allowed(state, { ...opts, presencePenalty: 0.1 }), false);
assert.equal(allowed(state, { ...opts, suppressTokenIds: [3] }), false);
assert.equal(allowed(state, { ...opts, logitMaskFn() {} }), false);
assert.equal(allowed({ ...state, modelConfig: { logitOutputScale: 2 } }), false);
assert.equal(allowed({ ...state, operatorDiagnostics: {} }), false);
assert.equal(allowed({ ...state, runtimeConfig: { ...state.runtimeConfig, shared: { debug: { probes: [{}] } } } }), false);
assert.equal(allowed({ ...state, runtimeConfig: { inference: { session: { useGreedyLmHeadArgmaxFusion: false } } } }), false);
console.log('greedy-policy.test: passed');
