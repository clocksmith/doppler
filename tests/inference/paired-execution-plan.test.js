import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { compileExecutionPlanState, resolveExecutionSessionPlan } from '../../src/inference/pipelines/text/execution-plan.js';
const execution = JSON.parse(readFileSync(new URL('../../artifacts/paired-intermediates-20261012/signed-closure-correction/execution.json', import.meta.url)));
for (const [variant, shader] of [
  ['paired_beta', 'qwen_linear_attention_prepare_paired'],
  ['paired_decay_scale', 'qwen_linear_attention_scale_paired'],
  ['paired_log_decay', 'qwen_linear_attention_log_decay_paired'],
  ['paired_decay', 'qwen_linear_attention_decay_paired'],
]) {
  execution.kernels[variant] = { kernel: `${shader}.wgsl`, entry: 'main', constants: {},
    digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${shader}.wgsl#main`]}` };
  execution.mechanismKernels.push(variant);
}

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
  m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== 'paired_beta'); },
  m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== 'gather_f16_vec4'); },
  m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== 'gather_split8_f16_vec4_f32_out'); },
  m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== 'logit_finalize_f32'); },
  m => { m.inference.execution.kernels.projection_q4.constants.SCALAR_OUTPUT = true; },
  m => { m.inference.execution.kernels.head.constants.SCALAR_OUTPUT = false; },
  m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== 'cast_two_f32_to_f16'); },
  m => { m.inference.execution.decode = m.inference.execution.decode.filter(s => s[0] !== 'linear_b_proj'); },
  m => { m.inference.execution.kernels.norm.kernel = 'rmsnorm.wgsl'; },
  m => { m.inference.layerPattern.layerTypes = ['linear_attention']; },
]) { const changed = structuredClone(manifest); alter(changed); assert.throws(() => compile(changed), /PairedExecution|Execution/); }
console.log('paired execution plan: complete graph admission, exact variants, helper closure, scalar logits and session metadata pass');
