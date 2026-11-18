import assert from 'node:assert/strict';
import prefill from '../../src/config/runtime/profiles/qwen3-8-27b-orin-prefill.json' with { type: 'json' };
import decode from '../../src/config/runtime/profiles/qwen3-8-27b-orin-decode.json' with { type: 'json' };
import candidate from '../../src/config/runtime/profiles/qwen3-8-27b-orin-projections.json' with { type: 'json' };
import manifest from '../fixtures/qwen3-8-27b-orin-execution.json' with { type: 'json' };
import { validateRuntimeProfileMetadata } from '../../src/config/schema/runtime/runtime-profile.schema.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution/execution-v1.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

validateRuntimeProfileMetadata(candidate);
assert.equal(candidate.intent, 'calibrate');
assert.deepEqual(Object.keys(candidate.runtime), ['inference']);
assert.equal(candidate.extends, decode.id);
assert.equal(candidate.model, manifest.modelId);
assert.equal(candidate.stability, 'experimental');
assert.deepEqual(candidate.compatibleIntents, ['calibrate']);
assert.deepEqual(Object.keys(candidate.runtime.inference), ['executionPatch']);
const baseline = mergeRuntimeValues(prefill.runtime, decode.runtime);
const merged = mergeRuntimeValues(baseline, candidate.runtime);
// Runtime arrays replace, so retain every inherited embedding/prefill patch.
assert.deepEqual(merged.inference.executionPatch.addKernels.slice(0, -1), baseline.inference.executionPatch.addKernels);
assert.deepEqual(merged.inference.executionPatch.set.slice(0, -5), baseline.inference.executionPatch.set);
assert.deepEqual(merged.inference.session, baseline.inference.session);
assert.deepEqual(merged.inference.kernelPathPolicy, baseline.inference.kernelPathPolicy);
const original = structuredClone(manifest);
const compile = runtime => compileExecutionV1({
  manifestInference: manifest.inference, modelId: manifest.modelId,
  numLayers: manifest.architecture.numLayers, headDim: manifest.architecture.headDim,
  weightDtype: manifest.quantizationInfo.weights,
  capabilities: { hasF16: false, hasSubgroups: true }, platform: { id: 'generic' }, useGPU: true,
  kernelPathPolicy: runtime.inference.kernelPathPolicy,
  executionPatch: runtime.inference.executionPatch, runtimeSession: runtime.inference.session,
});
const before = compile(baseline), after = compile(merged);
assert.deepEqual(after.session, before.session);
assert.deepEqual(after.policies, before.policies);
const ops = new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj']);
assert.deepEqual(after.resolvedSteps.prefill, before.resolvedSteps.prefill);
assert.equal(after.resolvedSteps.decode.length, before.resolvedSteps.decode.length);
let changedCount = 0;
for (const [index, step] of after.resolvedSteps.decode.entries()) {
  const previous = before.resolvedSteps.decode[index];
  if (step.phase === 'decode' && ops.has(step.op)) {
    changedCount++;
    assert.equal(step.kernel, 'fused_matmul_q4_subgroup.wgsl');
    assert.equal(step.entry, 'main');
    assert.deepEqual(step.constants, {
      WORKGROUP_SIZE: 256, COLS_PER_WG: 32, THREADS_PER_COL_GEMV: 8,
      USE_FULL_BLOCK_FAST_PATH: true,
    });
    assert.deepEqual(step.kernelRef, {
      id: 'fused_matmul_q4_subgroup.main', version: '1.0.0',
      digest: 'sha256:d0aa5474d8b82b918f6f214f85eb35d39513feb1ab4ff52e7e3cfb8eefc4049e',
    });
    const { kernel: ignoredKernel, entry: ignoredEntry, constants: ignoredConstants,
      kernelRef: ignoredRef, ...unchanged } = step;
    const { kernel: oldKernel, entry: oldEntry, constants: oldConstants,
      kernelRef: oldRef, ...oldUnchanged } = previous;
    assert.deepEqual(unchanged, oldUnchanged);
  } else {
    assert.deepEqual(step, previous);
  }
}
assert.equal(changedCount, 5);
const changes = merged.inference.executionPatch.set.slice(-5);
assert.deepEqual(changes.map(step => step.op), [...ops]);
assert.ok(changes.every(step => step.section === 'decode' && step.kernelKey === 'q4_projection_orin_sg8'));
const kernel = getKernelConfig('matmul', 'q4_fused_subgroup8');
assert.ok(kernel.requires.includes('subgroups'));
assert.ok(!kernel.requires.includes('shader-f16'));
assert.equal(manifest.architecture.hiddenSize % 256, 0);
assert.equal(manifest.architecture.intermediateSize % 256, 0);
assert.deepEqual(manifest, original);
console.log('qwen3-8-27b-orin-projections: explicit decode projections preserve inherited prefill, embeddings, fused FFN and identity');
