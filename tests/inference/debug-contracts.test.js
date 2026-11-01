import assert from 'node:assert/strict';
import {
  createDefaultCaptureConfig,
  createEscalationPolicy,
  resolveCapturePolicy,
  validateCaptureConfig,
} from '../../src/debug/capture-policy.js';
import * as shared from '../../src/debug/config.js';
import * as pipelineOwner from '../../src/debug/pipeline-config.js';
import * as pipeline from '../../src/inference/pipelines/text/debug-utils/config.js';
import { getLogitsHealth } from '../../src/inference/pipelines/text/debug-utils/health-metrics.js';

const defaults = createDefaultCaptureConfig();
assert.equal(validateCaptureConfig(defaults), true);
assert.equal(validateCaptureConfig({ ...defaults, escalation: createEscalationPolicy() }), true);
for (const invalid of [null, undefined, [], false]) {
  assert.throws(() => validateCaptureConfig(invalid), /config must be an object/);
}
for (const field of Object.keys(defaults)) {
  const missing = { ...defaults };
  delete missing[field];
  assert.throws(() => validateCaptureConfig(missing), new RegExp(field));
}
const invalidFields = {
  enabled: [0, 'true', null],
  defaultLevel: ['', false, 'constructor', 'toString', '__proto__'],
  targetLevel: ['', null, 'constructor'],
  targetOpIds: ['op', [null], [''], ['  '], new Array(1)],
  targetOperatorClasses: ['attention', [1], ['']],
  targetLayers: ['0', ['0'], [-1], [0.5], [NaN], [Infinity]],
  sampleCount: [-1, 0.5, '16', NaN, Infinity],
  escalation: [false, [], {}, { ...createEscalationPolicy(), windowBefore: -1 },
    { ...createEscalationPolicy(), windowAfter: '1' },
    { ...createEscalationPolicy(), baseLevel: 'constructor' },
    { ...createEscalationPolicy(), escalatedLevel: '' },
    { ...createEscalationPolicy(), resolveForIndex: null }],
};
for (const [field, values] of Object.entries(invalidFields)) {
  for (const value of values) {
    assert.throws(() => validateCaptureConfig({ ...defaults, [field]: value }), new RegExp(field));
    assert.throws(() => resolveCapturePolicy('layer.0.attn', { ...defaults, [field]: value }), new RegExp(field));
  }
}
assert.equal(validateCaptureConfig({ ...defaults, sampleCount: 0 }), true);
assert.equal(resolveCapturePolicy('any', null), 'none');
for (const targeting of [
  { targetOpIds: ['layer.0.attn'] },
  { targetOperatorClasses: ['attention'] },
  { targetLayers: [0] },
]) {
  assert.equal(resolveCapturePolicy('layer.0.attn', { ...defaults, ...targeting, enabled: true }, 'attention'), defaults.targetLevel);
  assert.throws(() => resolveCapturePolicy('layer.0.attn', { ...defaults, ...targeting, enabled: true, targetLevel: undefined }, 'attention'), /targetLevel/);
}

try {
  for (const name of ['setDebugCategories', 'getDebugConfig', 'incrementDecodeStep', 'resetDecodeStep', 'getDecodeStep']) {
    assert.equal(pipeline[name], (pipelineOwner[name] ?? shared[name]), `${name} must have one owner`);
  }
  pipeline.resetDebugConfig();
  pipeline.setDebugCategories({ layer: true }, { layers: [2], maxDecodeSteps: 1 });
  shared.setTrace('attn', { layers: [2], maxDecodeSteps: 1 });
  assert.equal(pipelineOwner.getDebugConfig().categories.layer, true);
  assert.equal(pipeline.isEnabled('layer', 2), true);
  assert.equal(pipeline.isEnabled('layer', 1), false);
  assert.equal(pipeline.isEnabled('attn', 2), false, 'trace and pipeline categories retain distinct contracts');
  assert.equal(pipeline.incrementDecodeStep(), 1);
  assert.equal(shared.getDecodeStep(), 1);
  assert.equal(shared.isTraceEnabled('attn', 2), true);
  assert.equal(shared.incrementDecodeStep(), 2);
  assert.equal(pipeline.getDecodeStep(), 2);
  assert.equal(pipeline.isEnabled('layer', 2), false);
  assert.equal(shared.isTraceEnabled('attn', 2), false);
  pipeline.resetDecodeStep();
  assert.equal(shared.getDecodeStep(), 0);
  assert.equal(pipeline.isEnabled('layer', 2), true);
  shared.incrementDecodeStep();
  pipeline.resetDebugConfig();
  assert.equal(shared.getDecodeStep(), 0);
  assert.equal(pipeline.isEnabled('layer', 2), false);
  assert.equal(pipeline.shouldDebugLayerOutput(0, null), false);
  assert.equal(pipeline.shouldDebugLayerOutput(0, []), true);
} finally {
  shared.setTrace(false);
  pipelineOwner.resetDebugConfig();
}

for (const [values, expected] of [
  [[], { nanCount: 0, infCount: 0, nonZeroCount: 0, maxAbs: 0 }],
  [[NaN, Infinity, -Infinity], { nanCount: 1, infCount: 2, nonZeroCount: 0, maxAbs: 0 }],
  [[0, -0, -3, 2, NaN, Infinity], { nanCount: 1, infCount: 1, nonZeroCount: 2, maxAbs: 3 }],
]) {
  assert.deepEqual(getLogitsHealth(new Float32Array(values)), expected);
}
console.log('debug-contracts.test: ok');
