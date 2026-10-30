import assert from 'node:assert/strict';
import { assertPreparedDecodeWorkspaceCompatible } from '../../src/inference/pipelines/text/execution-plan.js';

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
console.log('prepared-moe-admission.test: ok');
