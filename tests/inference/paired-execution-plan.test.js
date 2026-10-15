import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { compileExecutionPlanState, resolveExecutionSessionPlan } from '../../src/inference/pipelines/text/execution-plan.js';
import { resolvePairedAttentionPipeline, resolvePairedAttentionVariant } from '../../src/gpu/kernels/paired-attention.js';
import { resolveDeclaredMechanismReplacements } from '../../src/config/execution-registry-contract.js';
const execution = JSON.parse(readFileSync(new URL('../../artifacts/paired-intermediates-20261012/signed-closure-correction/execution.json', import.meta.url)));
for (const [variant, shader] of [
  ['paired_beta', 'qwen_linear_attention_prepare_paired'],
  ['paired_decay_scale', 'qwen_linear_attention_scale_paired'],
  ['paired_log_decay', 'qwen_linear_attention_log_decay_paired'],
  ['paired_decay', 'qwen_linear_attention_decay_paired'],
  ...['qk', 'state', 'norm', 'gate'].map(phase => [`paired_${phase}`, `gated_delta_${phase}_paired`]),
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
// normalization without linearNormGateActivation: a pre-field (silu) Qwen3.5 manifest.
const manifest = { architecture: { numLayers: 2 }, inference: { schema: 'doppler.execution/v1', normalization: {},
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
  ...['qk', 'state', 'norm', 'gate'].map(phase => m => { m.inference.execution.mechanismKernels = m.inference.execution.mechanismKernels.filter(k => k !== `paired_${phase}`); }),
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
  // A sigmoid manifest cannot be admitted with the silu gate and anchor declared.
  m => { m.inference.normalization.linearNormGateActivation = 'sigmoid'; },
  m => { delete m.inference.normalization; },
]) { const changed = structuredClone(manifest); alter(changed); assert.throws(() => compile(changed), /PairedExecution|Execution|NormGate|normalization must be an object/); }
// The declared attention step selects the paired program; a tiled paired program is admitted and
// dispatched, and attention programs without paired Q/output over F16 K/V fail closed.
assert.equal(resolvePairedAttentionVariant(plan.primaryPlan.kernelPath, 'decode', 1), 'paired');
assert.equal(resolvePairedAttentionVariant(null, 'decode', 1), 'paired');
const tiled = structuredClone(manifest);
tiled.inference.execution.kernels.attention = { ...tiled.inference.execution.kernels.attention,
  kernel: 'attention_paired_tiled.wgsl', constants: { WORKGROUP_SIZE: 256, DOT_LANES: 4 },
  digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['attention_paired_tiled.wgsl#main']}` };
const tiledPlan = compile(tiled);
assert.equal(tiledPlan.primaryPlan.activationPrecision, 'two-f32');
for (const phase of ['prefill', 'decode']) assert.equal(resolvePairedAttentionVariant(tiledPlan.primaryPlan.kernelPath, phase, 1), 'paired_tiled');
assert.deepEqual(resolvePairedAttentionPipeline('paired_tiled'), { operation: 'attention', variant: 'paired_tiled', constants: null });
assert.throws(() => resolvePairedAttentionPipeline('decode_subgroup'), /not a declared paired/);
const scalarAttention = structuredClone(manifest);
scalarAttention.inference.execution.kernels.attention.kernel = 'attention_head256_f16kv.wgsl';
assert.throws(() => compile(scalarAttention), /attention is not bound to the declared paired attention implementation/);
// The sliced paired RMSNorm program is admitted for every declared norm role.
const sliced = structuredClone(manifest);
sliced.inference.execution.kernels.norm = { ...sliced.inference.execution.kernels.norm, kernel: 'rmsnorm_paired_sliced.wgsl',
  constants: { WORKGROUP_SIZE: 256 }, digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['rmsnorm_paired_sliced.wgsl#main']}` };
assert.equal(compile(sliced).primaryPlan.activationPrecision, 'two-f32');
// A declared replacement program joins the closure and is selected in the mechanism's place.
assert.deepEqual(resolveDeclaredMechanismReplacements(manifest.inference.execution), {});
const replaced = structuredClone(manifest);
replaced.inference.execution.kernels.paired_state_sliced = { kernel: 'gated_delta_state_paired_sliced.wgsl', entry: 'main',
  constants: { WORKGROUP_SIZE: 256, K_SLICES: 16 }, digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['gated_delta_state_paired_sliced.wgsl#main']}` };
replaced.inference.execution.kernels.paired_qk_parallel = { kernel: 'gated_delta_qk_paired_parallel.wgsl', entry: 'main',
  constants: { WORKGROUP_SIZE: 256 }, digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['gated_delta_qk_paired_parallel.wgsl#main']}` };
replaced.inference.execution.mechanismKernels.push('paired_state_sliced', 'paired_qk_parallel');
assert.equal(compile(replaced).primaryPlan.activationPrecision, 'two-f32');
assert.deepEqual(resolveDeclaredMechanismReplacements(replaced.inference.execution), {
  'gated_delta_recurrent/paired_state': 'paired_state_sliced', 'gated_delta_recurrent/paired_qk': 'paired_qk_parallel' });
// Other specialization constants do not declare the replacement.
const respecialized = structuredClone(replaced);
respecialized.inference.execution.kernels.paired_state_sliced.constants = { WORKGROUP_SIZE: 128, K_SLICES: 16 };
assert.deepEqual(resolveDeclaredMechanismReplacements(respecialized.inference.execution), { 'gated_delta_recurrent/paired_qk': 'paired_qk_parallel' });
// The rolled sliced state program is an alternative replacement of the same mechanism; declaring it with
// the sliced program fails closed (two replacements of one mechanism).
const rolled = structuredClone(manifest);
rolled.inference.execution.kernels.paired_state_sliced_rolled = { kernel: 'gated_delta_state_paired_sliced_rolled.wgsl', entry: 'main',
  constants: { WORKGROUP_SIZE: 256, K_SLICES: 16 }, digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['gated_delta_state_paired_sliced_rolled.wgsl#main']}` };
rolled.inference.execution.mechanismKernels.push('paired_state_sliced_rolled');
assert.equal(compile(rolled).primaryPlan.activationPrecision, 'two-f32');
assert.deepEqual(resolveDeclaredMechanismReplacements(rolled.inference.execution), { 'gated_delta_recurrent/paired_state': 'paired_state_sliced_rolled' });
const both = structuredClone(rolled);
both.inference.execution.kernels.paired_state_sliced = replaced.inference.execution.kernels.paired_state_sliced;
both.inference.execution.mechanismKernels.push('paired_state_sliced');
assert.throws(() => resolveDeclaredMechanismReplacements(both.inference.execution), /declare both/);
// Compensated matmul programs are admitted for vector projections and the scalar-output head.
const compensated = structuredClone(manifest);
for (const key of ['projection_q4', 'projection_f16', 'head']) {
  compensated.inference.execution.kernels[key] = { ...compensated.inference.execution.kernels[key], kernel: 'matmul_paired_compensated.wgsl',
    digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['matmul_paired_compensated.wgsl#main']}` };
}
assert.equal(compile(compensated).primaryPlan.activationPrecision, 'two-f32');
console.log('paired execution plan: complete graph admission, exact variants, helper closure, scalar logits, tiled attention, sliced RMSNorm, compensated matmul and declared state (sliced or rolled) and Q/K replacement selection and session metadata pass');
