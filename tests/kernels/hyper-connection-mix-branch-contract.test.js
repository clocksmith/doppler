import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolveHyperConnectionMixBranch } from '../../src/inference/pipelines/text/hyper-connection/projection.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

// CPU contract proof only: exact math-source reuse and the fused exchange's
// indexing/materialization. Physical arithmetic parity is a separate test.
const shader = name => readFile(new URL(`../../src/gpu/kernels/${name}.wgsl`, import.meta.url), 'utf8');
const [activation, branch, fused] = await Promise.all([
  shader('hyper_connection_activation'), shader('hyper_connection_branch'), shader('hyper_connection_mix_branch'),
]);
const refinedFunctions = source => source.slice(source.indexOf('fn exp_refined'), source.indexOf('fn silu('));
assert.ok(fused.includes(refinedFunctions(activation)), 'fusion must retain the original refined exponential, reciprocal, division and sigmoid');
assert.ok(fused.includes('rounded_mix[product_lane] * normed[product_base + stream * STREAM_WIDTH]'));
assert.ok(fused.includes('sum = sum + rounded_products[(lid.x + WORKGROUP_SIZE - 1u) % WORKGROUP_SIZE]'));
assert.ok(branch.includes('sum = sum + rounded_products[(lid.x + WORKGROUP_SIZE - 1u) % WORKGROUP_SIZE]'));
assert.equal((fused.match(/workgroupBarrier\(\);/g) ?? []).length, 3);

const step = { op: 'hc_mix_branch', kernel: 'hyper_connection_mix_branch.wgsl', entry: 'main',
  digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['hyper_connection_mix_branch.wgsl#main']}` };
const path = steps => ({ id: 'hc-contract', decode: { steps }, prefill: { steps: [] } });
const select = kernelPath => resolveHyperConnectionMixBranch({ kernelPath, phase: 'decode', layerIdx: 2 });
assert.equal(select(null), false);
assert.equal(select(path([])), false);
assert.equal(select(path([step])), true);
assert.throws(() => select(path([{ ...step, digest: `sha256:${'0'.repeat(64)}` }])), /compiled entry digest/);
assert.throws(() => select(path([{ ...step, digest: undefined }])), /compiled entry digest/);
assert.throws(() => select(path([{ ...step, entry: 'wrong_entry' }])), /one registered/);
assert.equal(resolveHyperConnectionMixBranch({ kernelPath: path([step]), phase: 'prefill', layerIdx: 2 }), false);
assert.throws(() => select(path([step, step])), /one registered/);
assert.throws(() => select(path([{ ...step, kernel: 'hyper_connection_branch.wgsl' }])), /one registered/);
assert.throws(() => select(path([{ ...step, constants: { STREAM_COUNT: 4 } }])), /one registered/);
assert.throws(() => select(path([{ ...step, constants: { WORKGROUP_SIZE: 128 } }])), /one registered/);
assert.equal(select(path([{ ...step, constants: { WORKGROUP_SIZE: 256 } }])), true);
const override = { ...path([step]), layerOverrides: [{ layers: [2], decode: { steps: [{ op: 'unchanged' }] } }] };
assert.equal(select(override), false, 'layer overrides own selection; no implicit base-path promotion');
assert.throws(() => resolveHyperConnectionMixBranch({ kernelPath: path([step]), phase: null, layerIdx: 2 }), /explicit phase/);
const config = getKernelConfig('hyper_connection_mix_branch', 'f32');
assert.equal(config.shaderFile, 'hyper_connection_mix_branch.wgsl');
assert.deepEqual(config.bindings.map(binding => binding.name), ['uniforms', 'logits', 'normed', 'output']);

// Model F32 sigmoid materialization, then the cyclic producer/consumer exchange
// for irregular token, stream and workgroup boundaries. This does not emulate
// GPU exp/fma accuracy; the physical test compares the two real implementations.
const f = Math.fround;
for (const width of [1, 65, 257]) {
  for (const streams of [2, 4, 5]) {
    const tokens = 3, count = width * tokens, group = 256;
    const mix = Float32Array.from({ length: count * streams }, (_, i) => f(1 / (1 + Math.exp((i % 37) - 18))));
    const norm = Float32Array.from({ length: mix.length }, (_, i) => f(Math.sin(i * 1.17) * (i % 7 ? 1 : 1000)));
    const expected = new Float32Array(count), actual = new Float32Array(count);
    for (let index = 0; index < count; index++) {
      const base = Math.floor(index / width) * streams * width + index % width;
      let sum = 0;
      for (let stream = 0; stream < streams; stream++) sum = f(sum + f(mix[base + stream * width] * norm[base + stream * width]));
      expected[index] = f(sum / streams);
    }
    for (let base = 0; base < count; base += group) {
      const sums = new Float32Array(group), roundedMix = new Float32Array(group), products = new Float32Array(group);
      for (let stream = 0; stream < streams; stream++) {
        for (let lane = 0; lane < group; lane++) {
          const index = base + lane, offset = Math.floor(index / width) * streams * width + index % width + stream * width;
          roundedMix[lane] = index < count ? mix[offset] : 0;
        }
        for (let lane = 0; lane < group; lane++) {
          const next = (lane + 1) % group, index = base + next;
          const offset = Math.floor(index / width) * streams * width + index % width + stream * width;
          products[lane] = index < count ? f(roundedMix[next] * norm[offset]) : 0;
        }
        for (let lane = 0; lane < group; lane++) sums[lane] = f(sums[lane] + products[(lane + group - 1) % group]);
      }
      for (let lane = 0; lane < group && base + lane < count; lane++) actual[base + lane] = f(sums[lane] / streams);
    }
    assert.deepEqual(new Uint32Array(actual.buffer), new Uint32Array(expected.buffer));
  }
}
console.log('HC fused mix-branch: explicit identity, refined math and cyclic F32 exchange contracts passed (CPU only)');
