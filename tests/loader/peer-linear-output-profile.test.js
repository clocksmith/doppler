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
