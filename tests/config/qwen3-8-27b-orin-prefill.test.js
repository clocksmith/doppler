import assert from 'node:assert/strict';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import profile from '../../src/config/runtime/profiles/qwen3-8-27b-orin-prefill.json' with { type: 'json' };
import manifest from '../fixtures/qwen3-8-27b-orin-execution.json' with { type: 'json' };
import { validateRuntimeProfileMetadata } from '../../src/config/schema/runtime/runtime-profile.schema.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution/execution-v1.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

validateRuntimeProfileMetadata(profile);
assert.equal(profile.model, manifest.modelId);
assert.equal(profile.stability, 'experimental');
assert.deepEqual(profile.compatibleIntents, ['calibrate']);
const original = structuredClone(manifest);
const { executionPatch, kernelPathPolicy } = profile.runtime.inference;
const projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
assert.deepEqual(executionPatch.set.filter(step => step.section === 'prefill').map(step => step.op), projections);
assert.ok(executionPatch.set.every(step => ['preLayer', 'prefill'].includes(step.section)));
for (const { kernel } of executionPatch.addKernels) {
  assert.equal(kernel.digest, `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${kernel.kernel}#${kernel.entry}`]}`);
}
const options = {
  manifestInference: manifest.inference,
  modelId: manifest.modelId,
  numLayers: manifest.architecture.numLayers,
  headDim: manifest.architecture.headDim,
  weightDtype: manifest.quantizationInfo.weights,
  capabilities: { hasF16: false, hasSubgroups: true },
  platform: { id: 'generic' },
  useGPU: true,
  kernelPathPolicy,
};
// Compare against the same packed embedding lane, changing only prefill projections.
const packedOnly = {
  addKernels: executionPatch.addKernels.filter(item => item.key === 'embed_packed_f16'),
  set: executionPatch.set.filter(step => step.section === 'preLayer'),
};
const baseline = compileExecutionV1({ ...options, executionPatch: packedOnly });
const candidate = compileExecutionV1({ ...options, executionPatch });
const stock = compileExecutionV1(options);
for (const op of projections) {
  assert.equal(candidate.resolvedSteps.prefill.find(step => step.op === op).kernel, 'fused_matmul_q4_widetile_f32.wgsl');
  assert.equal(baseline.resolvedSteps.prefill.find(step => step.op === op).kernel, 'fused_matmul_q4_batched_multicol_shared.wgsl');
  assert.equal(stock.resolvedSteps.prefill.find(step => step.op === op).kernel, 'fused_matmul_q4_batched_multicol_shared.wgsl');
}
assert.equal(candidate.resolvedSteps.prefill.find(step => step.op === 'embed').kernel, 'gather_split4_packed_f16_vec4_f32_out.wgsl');
assert.deepEqual(candidate.resolvedSteps.decode, baseline.resolvedSteps.decode);
assert.deepEqual(candidate.resolvedSteps.prefill.filter(step => !projections.includes(step.op)),
  baseline.resolvedSteps.prefill.filter(step => !projections.includes(step.op)));
assert.deepEqual(candidate.session, baseline.session);
assert.deepEqual(candidate.policies, baseline.policies);
assert.deepEqual(manifest, original);
const kernel = getKernelConfig('matmul', 'q4_fused_widetile_f32');
assert.ok(!kernel.requires.includes('shader-f16'));
assert.deepEqual(kernel.variantMetadata, { tileM: 4, colsPerWg: 256, dispatchGeometry: 'row-column-tile' });
assert.deepEqual(kernel.workgroupSize, [256, 1, 1]);
assert.equal(kernel.outputDtype, 'f32');
console.log('qwen3-8-27b-orin-prefill: packed embedding and seven F32 prefill kernels; decode, session, identity and global policy preserved');
