import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { bindLinearOutputProfile } from '../../tools/pooled-equal-weights/bind-linear-output-profile.js';
import { getLayerSteps } from '../../src/config/kernel-path-loader.js';
import { resolveQwenNormRouter, resolveQwenSharedSlotVariants } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
import { runMoEFFNGPU } from '../../src/inference/pipelines/text/ffn/moe.js';

const dopplerRoot = fileURLToPath(new URL('../../', import.meta.url));
test('binds opposite Q4/Q8 layer encodings from actual tensor metadata without changing the template', async () => {
  const profile = JSON.parse(await readFile(new URL('../../tools/pooled-equal-weights/profiles/qwen3-6-35b-a3b-q4_0-pooled-q8-common-diagnostic.json', import.meta.url)));
  const snapshot = structuredClone(profile);
  const manifest = { architecture: { numLayers: 7 }, inference: {}, tensors: {
    'blk.4.ssm_out.weight': { dtype: 'Q8_0', shape: [32, 32], shard: 0, offset: 0, size: 1088 },
    'blk.6.ssm_out.weight': { dtype: 'Q4_0', shape: [32, 32], shard: 0, offset: 1088, size: 576 },
  } };
  const bound = await bindLinearOutputProfile({ profile, manifest, dopplerRoot });
  assert.equal(bound.bindings.length, 4);
  assert.equal(bound.representationTensorsChecked, 2);
  assert.deepEqual(profile, snapshot);
  const repeated = await bindLinearOutputProfile({ profile: bound.profile, manifest, dopplerRoot });
  assert.equal(repeated.bindings.length, 0);
  assert.deepEqual(repeated.profile, bound.profile);
  manifest.tensors['blk.4.ssm_out.weight'].dtype = 'F32';
  await assert.rejects(bindLinearOutputProfile({ profile, manifest, dopplerRoot }), /Unsupported packed GGUF/);
});

test('does not turn an incompatible shader into a supported program', async () => {
  const profile = { runtime: { inference: { kernelPath: { id: 'bad', decode: { steps: [
    { op: 'linear_out_proj', kernel: 'matmul_f32.wgsl', constants: {} },
  ] } } } } };
  const manifest = { architecture: { numLayers: 1 }, tensors: { 'blk.0.ssm_out.weight': { dtype: 'Q8_0' } } };
  await assert.rejects(bindLinearOutputProfile({ profile, manifest, dopplerRoot }), /does not declare/);
});

test('retains optimized declarations and adds every missing model projection in both phases and at the root', async () => {
  const retained = { op: 'linear_a_proj', kernel: 'matmul_gemv_f32_subgroup.wgsl', entry: 'main' };
  const added = { op: 'attn_hc_inject', kernel: 'matmul_f32.wgsl', entry: 'main' };
  const root = { op: 'hc_mixer_down', kernel: 'matmul_f32.wgsl', entry: 'main' };
  const profile = { runtime: { inference: { kernelPath: { id: 'candidate',
    decode: { steps: [retained] }, prefill: { steps: [retained] }, postLayer: [] } } } };
  const requiredKernelPath = { id: 'actual-model', decode: { steps: [added] }, prefill: { steps: [added] }, postLayer: [root] };
  const manifest = { architecture: { numLayers: 1 }, inference: {}, tensors: {
    'blk.0.hc_attn_inject.weight': { dtype: 'F32', role: 'other', shape: [4, 32], shard: 0, offset: 0, size: 512 },
    'output_hc_down.weight': { dtype: 'F32', role: 'other', shape: [4, 32], shard: 0, offset: 512, size: 512 },
  } };
  const result = await bindLinearOutputProfile({ profile, manifest, dopplerRoot, requiredKernelPath });
  assert.equal(result.declarations.length, 3);
  assert.deepEqual(result.profile.runtime.inference.kernelPath.layerOverrides[0].prefill.steps[0], retained);
  assert.deepEqual(result.profile.runtime.inference.kernelPath.postLayer, [root]);
  assert.equal(profile.runtime.inference.kernelPath.postLayer.length, 0);
});

test('binds classic MoE fusions to the HC caller without inventing pending norm or residual inputs', async () => {
  const profile = JSON.parse(await readFile(new URL('../../tools/pooled-equal-weights/profiles/qwen3-6-35b-a3b-q4_0-pooled-q8-common-diagnostic.json', import.meta.url)));
  const snapshot = structuredClone(profile);
  const manifest = { architecture: { numLayers: 1 }, inference: { hyperConnections: { streamCount: 4 } }, tensors: {} };
  const result = await bindLinearOutputProfile({ profile, manifest, dopplerRoot });
  const kernelPath = result.profile.runtime.inference.kernelPath;
  assert.equal(result.semanticBindings.length, 2);
  assert.deepEqual(profile, snapshot);
  assert.equal(resolveQwenNormRouter({ kernelPath, phase: 'decode', layerIdx: 0 }), false);
  assert.equal(resolveQwenSharedSlotVariants({ kernelPath, phase: 'decode', layerIdx: 0 }).down, 'pair_combine_f32');
  const step = getLayerSteps(kernelPath, 0, 'decode').find(step => step.op === 'moe_shared_slot_down');
  assert.equal(step.digest, 'sha256:6b89485190d4b253374f705a5ff20d6111edb8d239597d0e66e44ca8cc72378d');
  // No GPU: reaching component validation proves the unchanged runtime guards
  // accept this HC caller without processFFNStandard's pending inputs.
  await assert.rejects(runMoEFFNGPU(0, null, 1, { kernelPath, phase: 'decode',
    config: { expertFormat: 'qwen3_5' }, runtimeMoERoutingConfig: { activeExpertSelection: 'topk-route' } }),
  /MoE components not initialized/);
  const repeated = await bindLinearOutputProfile({ profile: result.profile, manifest, dopplerRoot });
  assert.equal(repeated.semanticBindings.length, 0);
  assert.deepEqual(repeated.profile, result.profile);
  const classic = await bindLinearOutputProfile({ profile, manifest: { ...manifest, inference: {} }, dopplerRoot });
  assert.equal(classic.semanticBindings.length, 0);
  assert.deepEqual(getLayerSteps(classic.profile.runtime.inference.kernelPath, 0, 'decode'),
    getLayerSteps(snapshot.runtime.inference.kernelPath, 0, 'decode'));
});
