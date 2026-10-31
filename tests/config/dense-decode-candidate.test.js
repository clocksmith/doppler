import assert from 'node:assert/strict';
import { buildDenseDecodeCandidate } from '../../tools/experiments/dense-decode-candidate.js';
import { setDevice } from '../../src/gpu/device.js';
import { selectMatmulVariantAndFlags } from '../../src/gpu/kernels/matmul-selection.js';

const caps = { hasF16: true, hasSubgroups: true, hasSubgroupId: true };
const path = {
  activationDtype: 'f32',
  decode: { steps: [{ op: 'moe_router', kernel: 'matmul_f32.wgsl', entry: 'main', weights: 'router' }] },
  prefill: { steps: [{ op: 'moe_router', kernel: 'matmul_f32.wgsl', entry: 'main' }] },
  postLayer: [
    { op: 'lm_head', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main', weights: 'head', constants: { TILE_SIZE: 16 } },
    { op: 'lm_head_prefill', kernel: 'matmul_f16w_f32a.wgsl', entry: 'main' },
  ],
};
const router = { op: 'moe_router', section: 'layer', layerIdx: 0, M: 1, N: 256, K: 2048,
  activationDtype: 'f32', weightDtype: 'f32', outputDtype: 'f32', transposeB: true };
const head = { ...router, op: 'lm_head', section: 'postLayer', N: 248320, weightDtype: 'f16' };
const before = structuredClone(path);
const candidate = buildDenseDecodeCandidate(path, [router, head], caps);
assert.deepEqual(path, before, 'Candidate construction cannot mutate the admitted source graph');
assert.equal(candidate.qualified, false, 'A composed candidate is not a promoted result');
assert.deepEqual(candidate.kernelPath.prefill, path.prefill);
assert.deepEqual(candidate.kernelPath.postLayer[1], path.postLayer[1], 'Prefill head cannot become M=1-only');
assert.equal(candidate.kernelPath.postLayer[0].weights, 'head');
assert.equal(candidate.kernelPath.postLayer[0].constants, undefined, 'Tiled constants cannot leak into GEMV');
assert.match(candidate.kernelPath.postLayer[0].digest, /^sha256:[0-9a-f]{64}$/);
setDevice({ features: new Set(['subgroups', 'shader-f16']),
  limits: { maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64, maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 16384 }, queue: { submit() {} },
  createBindGroup() {}, lost: new Promise(() => {}) });
try {
  for (const [binding, expected] of [[router, 'gemv_f32_subgroup'], [head, 'gemv_subgroup_multicol']]) {
    const selection = selectMatmulVariantAndFlags('record', 1, binding.N, binding.K,
      binding.activationDtype, binding.weightDtype, binding.transposeB, binding.outputDtype,
      { role: binding.op, layerIdx: 0, kernelPath: candidate.kernelPath, phaseOverride: 'decode' });
    assert.equal(selection.variant, expected, 'The real wrapper must admit the intended storage-compatible program');
    assert.equal(selection.useGemv, true);
    assert.throws(() => selectMatmulVariantAndFlags('record', 2, binding.N, binding.K,
      binding.activationDtype, binding.weightDtype, binding.transposeB, binding.outputDtype,
      { role: binding.op, layerIdx: 0, kernelPath: candidate.kernelPath, phaseOverride: 'decode' }), /M=1/);
  }
} finally { setDevice(null); }
for (const unsupported of [
  { ...caps, hasSubgroups: false }, { ...caps, hasSubgroupId: false },
]) {
  const result = buildDenseDecodeCandidate(path, [router], unsupported);
  assert.deepEqual(result.kernelPath, path, 'Unsupported capability retains the declared baseline');
  assert(result.decisions[0].reason);
}
assert.deepEqual(buildDenseDecodeCandidate(path, [{ ...router, M: 2 }], caps).kernelPath, path);
assert.deepEqual(buildDenseDecodeCandidate(path, [{ ...router, M: 2 }], { ...caps, M: 1 }).kernelPath, path,
  'Capability metadata cannot override actual matrix geometry');
const invalidPrecision = structuredClone(path);
invalidPrecision.decode.steps[0].precision = { inputDtype: 'f16' };
assert.throws(() => buildDenseDecodeCandidate(invalidPrecision, [router], caps), /materialized binding/);
assert.deepEqual(buildDenseDecodeCandidate(path, [head], { ...caps, hasF16: false }).kernelPath, path);
assert.throws(() => buildDenseDecodeCandidate(path, [{ ...router, weightDtype: 'f16' }], caps), /materialized/);
assert.throws(() => buildDenseDecodeCandidate(path, [{ ...router, K: 0 }], caps), /M\/N\/K/);
assert.throws(() => buildDenseDecodeCandidate(path, [router, router], caps), /Duplicate/);
assert.throws(() => buildDenseDecodeCandidate(path, [router], {}), /boolean capability/);
assert.throws(() => buildDenseDecodeCandidate({ ...path, layerOverrides: [{}] }, [router], caps), /Expand layer overrides/);
console.log('dense-decode-candidate: storage, capabilities, shape, source immutability and real wrapper admission passed');
