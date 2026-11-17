import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { bindLinearOutputProfile } from '../../tools/pooled-equal-weights/bind-linear-output-profile.js';

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
