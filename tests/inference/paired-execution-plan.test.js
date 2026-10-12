import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { compileExecutionPlanState, resolveExecutionSessionPlan } from '../../src/inference/pipelines/text/execution-plan.js';
const execution = JSON.parse(readFileSync(new URL('../../artifacts/paired-intermediates-20261012/execution.json', import.meta.url)));
const runtimeConfig = createDopplerConfig().runtime;
Object.assign(runtimeConfig.inference.session, { intermediatePrecision: 'two-f32', linearRecurrentStatePrecision: 'two-f32' });
runtimeConfig.inference.compute.activationDtype = 'f32';
runtimeConfig.inference.generation.maxTokens = 8;
runtimeConfig.inference.session.decodeLoop = { batchSize: 1, stopCheckMode: 'per-token', readbackInterval: 1, readbackMode: 'sequential', ringTokens: 1, ringStop: 1, ringStaging: 1, disableCommandBatching: false }; 
runtimeConfig.inference.session.kvcache.kvDtype = 'f16';
runtimeConfig.inference.session.kvcache.layout = 'contiguous';
const manifest = { architecture: { numLayers: 2 }, inference: { schema: 'doppler.execution/v1',
  layerPattern: { type: 'custom', layerTypes: ['linear_attention', 'full_attention'] },
  session: { ...runtimeConfig.inference.session, compute: { defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' } } }, execution } };
const compile = (target = manifest) => {
  const result = compileExecutionV1({ manifestInference: target.inference, modelId: 'paired-fixture', numLayers: 2,
    weightDtype: 'q4k', runtimeSession: target.inference.session, kernelPathPolicy: runtimeConfig.inference.kernelPathPolicy });
  return compileExecutionPlanState({ runtimeConfig, manifest: target, resolvedKernelPath: result.runtimeInferencePatch.kernelPath, kernelPathSource: 'execution-v1' });
};
const plan = compile();
assert.equal(plan.primaryPlan.activationPrecision, 'two-f32');
assert.equal(resolveExecutionSessionPlan(plan).activationElementStrideBytes, 8);
assert.equal(resolveExecutionSessionPlan(plan).activationComponents, 2);
for (const alter of [
  m => { m.inference.execution.kernels.projection_q4.constants.SCALAR_OUTPUT = true; },
  m => { m.inference.execution.kernels.head.constants.SCALAR_OUTPUT = false; },
  m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== 'cast_two_f32_to_f16'); },
  m => { m.inference.execution.decode = m.inference.execution.decode.filter(s => s[0] !== 'linear_b_proj'); },
  m => { m.inference.execution.kernels.norm.kernel = 'rmsnorm.wgsl'; },
  m => { m.inference.layerPattern.layerTypes = ['linear_attention']; },
]) { const changed = structuredClone(manifest); alter(changed); assert.throws(() => compile(changed), /PairedExecution|Execution/); }
console.log('paired execution plan: complete graph admission, exact variants, helper closure, scalar logits and session metadata pass');
