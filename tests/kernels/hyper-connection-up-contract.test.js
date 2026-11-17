import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { resolveHyperConnectionUpMixBranch } from '../../src/inference/pipelines/text/hyper-connection/projection.js';

const shader = name => readFile(new URL(`../../src/gpu/kernels/${name}.wgsl`, import.meta.url), 'utf8');
const [original, fused, mix] = await Promise.all([
  shader('matmul_q4_0'), shader('hyper_connection_up_mix_branch_q4_0'), shader('hyper_connection_mix_branch'),
]);
assert.ok(fused.includes(original.slice(original.indexOf('fn q40_byte'), original.indexOf('var<workgroup> partial:'))),
  'dequantization must preserve the original Q4_0 byte addressing and expression');
assert.ok(fused.includes(mix.slice(mix.indexOf('fn exp_refined'), mix.indexOf('@compute'))), 'refined arithmetic cannot change');
const tree = source => source.slice(source.indexOf('for (var stride = THREADS_PER_COL_GEMV'), source.indexOf('\n    }', source.indexOf('for (var stride = THREADS_PER_COL_GEMV')) + 6);
assert.equal(tree(fused), tree(original), 'all32 lanes and the16/8/4/2/1 addition tree must remain identical');
assert.ok(fused.includes('sum = fma(input[k], q40_value(row, k, u.blocks), sum);'));
assert.ok(fused.includes('k += THREADS_PER_COL_GEMV'));
assert.ok(fused.includes('if (thread == 0u) { projected[stream] = partial[lane] * 1.0; }'));
assert.ok(fused.includes('if (thread == 1u) { mixed[stream] = sigmoid_refined(projected[stream]); }'));
assert.ok(fused.includes('if (thread == 2u) { products[stream] = mixed[stream] * normed[row]; }'));

const declaration = role => ({ op: `${role}_up_mix_branch`, kernel: 'hyper_connection_up_mix_branch_q4_0.wgsl', entry: 'main',
  digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['hyper_connection_up_mix_branch_q4_0.wgsl#main']}` });
const up = role => ({ op: `${role}_up`, kernel: 'matmul_q4_0.wgsl', entry: 'gemv', constants: {
  WORKGROUP_SIZE: 128, COLS_PER_WG: 4, THREADS_PER_COL_GEMV: 32, TILE_SIZE: 16, BLOCK_VALUES: 32, BLOCK_BYTES: 18, QUANT_FORMAT: 0,
} });
const path = steps => ({ id: 'up-contract', decode: { steps }, prefill: { steps: [] } });
const select = (kernelPath, role = 'attn_hc', phase = 'decode') => resolveHyperConnectionUpMixBranch({ kernelPath, role, phase, layerIdx: 0 });
assert.equal(select(path([])), false);
assert.equal(select(path([up('attn_hc'), declaration('attn_hc')])), true);
assert.equal(select(path([up('attn_hc'), declaration('attn_hc')]), 'mlp_hc'), false, 'roles do not implicitly opt each other in');
assert.equal(select(path([up('mlp_hc'), declaration('mlp_hc')]), 'mlp_hc'), true);
assert.throws(() => select(path([declaration('attn_hc')])), /original q4_0_gemv/);
assert.throws(() => select(path([{ ...up('attn_hc'), entry: 'cooperative' }, declaration('attn_hc')])), /original q4_0_gemv/);
assert.throws(() => select(path([{ ...up('attn_hc'), kernel: 'matmul_f32.wgsl', entry: 'main' }, declaration('attn_hc')])), /original q4_0_gemv/);
assert.throws(() => select(path([up('attn_hc'), { ...declaration('attn_hc'), digest: `sha256:${'0'.repeat(64)}` }])), /pinned Q4_0/);
assert.throws(() => select(path([up('attn_hc'), { ...declaration('attn_hc'), entry: 'wrong' }])), /pinned Q4_0/);
assert.throws(() => select(path([up('attn_hc'), { ...declaration('attn_hc'), constants: { STREAM_COUNT: 5 } }])), /pinned Q4_0/);
assert.throws(() => select(path([up('attn_hc'), declaration('attn_hc'), declaration('attn_hc')])), /pinned Q4_0/);
assert.throws(() => select({ ...path([]), prefill: { steps: [up('attn_hc'), declaration('attn_hc')] } }, 'attn_hc', 'prefill'), /in decode/);
assert.throws(() => select(path([]), 'hc_mixer'), /root mixer/);

// CPU scheduling reference: regroup four stream rows, without changing any
// row's lane accumulation or reduction. Physical FMA/exp parity is separate.
const f = Math.fround;
for (const rank of [32, 320, 352]) {
  const width = 7;
  const input = Float32Array.from({ length: rank }, (_, k) => f(Math.sin(k * 0.317)));
  const dot = row => {
    const partial = new Float32Array(32);
    for (let lane = 0; lane < 32; lane++) {
      for (let k = lane; k < rank; k += 32) {
        const scale = [0, 0.5, 1, -1][(row + Math.floor(k / 32)) % 4];
        const weight = f(scale * ((row * 13 + k * 7) % 16 - 8));
        partial[lane] = f(input[k] * weight + partial[lane]);
      }
    }
    for (let stride = 16; stride; stride /= 2) {
      for (let lane = 0; lane < stride; lane++) partial[lane] = f(partial[lane] + partial[lane + stride]);
    }
    return partial[0];
  };
  const originalRows = Float32Array.from({ length: width * 4 }, (_, row) => dot(row));
  const regrouped = new Float32Array(originalRows.length);
  for (let column = 0; column < width; column++) for (let stream = 0; stream < 4; stream++) {
    const row = stream * width + column;
    regrouped[row] = dot(row);
  }
  assert.deepEqual(new Uint32Array(regrouped.buffer), new Uint32Array(originalRows.buffer));
}
console.log('HC Q4_0 up fusion: original gemv32/tree/math, role identity and regrouped-row CPU contracts passed');
