import assert from 'node:assert/strict';
import base from '../../src/config/runtime/profiles/qwen3-8-27b-orin-prefill.json' with { type: 'json' };
import candidate from '../../src/config/runtime/profiles/qwen3-8-27b-orin-decode.json' with { type: 'json' };
import manifest from '../fixtures/qwen3-8-27b-orin-execution.json' with { type: 'json' };
import { validateRuntimeProfileMetadata } from '../../src/config/schema/runtime/runtime-profile.schema.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution/execution-v1.js';
import { resolveFusedGateUpPipelineConstants } from '../../src/inference/pipelines/text/ffn/dense-plan.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

validateRuntimeProfileMetadata(candidate);
assert.equal(candidate.extends, base.id);
assert.equal(candidate.model, manifest.modelId);
assert.equal(candidate.stability, 'experimental');
assert.deepEqual(candidate.compatibleIntents, ['calibrate']);
assert.deepEqual(Object.keys(candidate.runtime), ['inference']);
assert.deepEqual(Object.keys(candidate.runtime.inference), ['session']);
assert.deepEqual(Object.keys(candidate.runtime.inference.session), ['fusedFfnQ4K']);
assert.deepEqual(Object.keys(candidate.runtime.inference.session.fusedFfnQ4K), ['decode']);
const original = structuredClone(manifest);
const merged = mergeRuntimeValues(base.runtime, candidate.runtime);
const compile = runtime => compileExecutionV1({
  manifestInference: manifest.inference, modelId: manifest.modelId,
  numLayers: manifest.architecture.numLayers, headDim: manifest.architecture.headDim,
  weightDtype: manifest.quantizationInfo.weights,
  capabilities: { hasF16: false, hasSubgroups: true }, platform: { id: 'generic' }, useGPU: true,
  kernelPathPolicy: runtime.inference.kernelPathPolicy,
  executionPatch: runtime.inference.executionPatch,
  runtimeSession: runtime.inference.session,
});
const before = compile(base.runtime), after = compile(merged);
assert.deepEqual(after.resolvedSteps, before.resolvedSteps);
assert.deepEqual(after.policies, before.policies);
const { fusedFfnQ4K, ...remaining } = after.session;
const { fusedFfnQ4K: previous, ...previousRemaining } = before.session;
assert.deepEqual(remaining, previousRemaining);
assert.equal(previous, undefined);
assert.equal(fusedFfnQ4K.decode.variant, 'q4k_subgroup8');
assert.deepEqual(resolveFusedGateUpPipelineConstants({ phase: 'decode', runtimeSession: after.session }), {
  WORKGROUP_SIZE: 256, COLS_PER_WG: 32, THREADS_PER_COL: 8, USE_FULL_BLOCK_FAST_PATH: true,
});
assert.equal(resolveFusedGateUpPipelineConstants({ phase: 'prefill', runtimeSession: after.session }), null);
assert.equal(manifest.architecture.hiddenSize % 256, 0);
const kernel = getKernelConfig('fused_ffn', 'q4k_subgroup8');
assert.ok(kernel.requires.includes('subgroups'));
assert.ok(!kernel.requires.includes('shader-f16'));
assert.equal(kernel.shaderFile, 'fused_ffn_q4k_subgroup8.wgsl');
assert.deepEqual(manifest, original);
console.log('qwen3-8-27b-orin-decode: only fused gate/up decode session changes; graph, prefill and identity preserved');
