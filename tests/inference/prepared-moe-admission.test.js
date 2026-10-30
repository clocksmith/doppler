import assert from 'node:assert/strict';
import { assertPreparedDecodeWorkspaceCompatible, attachPreparedDecodeWorkspace } from '../../src/inference/pipelines/text/execution-plan.js';

const context = { decodeBuffers: {}, activationDtype: 'f32', config: { modelType: 'transformer',
  useMoE: true, expertFormat: 'qwen3_5' }, runtimeMoERoutingConfig: { activeExpertSelection: 'topk-route' } };
assert.doesNotThrow(() => assertPreparedDecodeWorkspaceCompatible(context));
assert.doesNotThrow(() => assertPreparedDecodeWorkspaceCompatible({ ...context, config: { useMoE: false } }));
for (const policy of ['all', 'topk-readback']) {
  assert.throws(() => assertPreparedDecodeWorkspaceCompatible({ ...context,
    runtimeMoERoutingConfig: { activeExpertSelection: policy } }), /supported GPU-route/);
}
for (const change of [{ activationDtype: 'f16' }, { activationPrecision: 'two-f32' }, { decodeBuffers: null },
  { lora: {} }, { operatorDiagnostics: {} }, { trainingCapture: {} }, { debugProbes: [{}] }]) {
  assert.throws(() => assertPreparedDecodeWorkspaceCompatible({ ...context, ...change }), /Prepared decode workspace requires/);
}
for (const expertFormat of ['gemma4', 'gpt-oss', 'mixtral']) {
  assert.throws(() => assertPreparedDecodeWorkspaceCompatible({ ...context,
    config: { ...context.config, expertFormat } }), /supported GPU-route|execution profile|No matching/);
}
// Single-token and logits decode attach the same session workspace as batch decode, or fail closed.
const workspace = {};
const attachContext = { ...context, decodeBuffers: { getWorkspace: device => (device === 'device' ? workspace : null) } };
assert.equal(attachPreparedDecodeWorkspace(false, null, attachContext, 'device'), null);
assert.throws(() => attachPreparedDecodeWorkspace(undefined, {}, attachContext, 'device'), /boolean usePreparedDecodeWorkspace/);
assert.throws(() => attachPreparedDecodeWorkspace(true, null, attachContext, 'device'), /requires a recorded decode step/);
assert.throws(() => attachPreparedDecodeWorkspace(true, {}, { ...attachContext, activationDtype: 'f16' }, 'device'),
  /Prepared decode workspace requires/);
const recorder = {};
assert.equal(attachPreparedDecodeWorkspace(true, recorder, attachContext, 'device'), workspace);
assert.equal(recorder.decodeWorkspace, workspace);
console.log('prepared-moe-admission.test: ok');
