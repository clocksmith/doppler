import assert from 'node:assert/strict';

// topk/fused_logit_rank (softmax_topk_logit_rank.wgsl) ranks the logits (ties to the
// lower index) and weights the picks by the softmax over the picked logits. Against a
// CPU reference and against topk/fused_ranked (softmax over all experts, top-k,
// renormalize): the same experts in the same order, weights within rounding.
// Unnormalized routing is refused.

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`softmax-topk-logit-rank-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { destroyDevice } = await import('../../src/gpu/device.js');
const { acquireBuffer, releaseBuffer, uploadData, readBuffer } = await import('../../src/memory/buffer-pool.js');
const { runSoftmaxTopK } = await import('../../src/gpu/kernels/softmax.js');

const owned = [];
const upload = (values) => {
  const buffer = acquireBuffer(values.byteLength);
  uploadData(buffer, values);
  owned.push(buffer);
  return buffer;
};
let seed = 0x2545f491;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296 - 0.5;
};

function reference(logits, numTokens, n, k) {
  const indices = [], weights = [];
  for (let token = 0; token < numTokens; token++) {
    const row = Array.from(logits.subarray(token * n, token * n + n), (value, i) => [value, i])
      .sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k);
    const e = row.map(([value]) => Math.fround(Math.exp(value - row[0][0])));
    const sum = e.reduce((acc, value) => Math.fround(acc + value), 0);
    for (let i = 0; i < k; i++) { indices.push(row[i][1]); weights.push(e[i] / sum); }
  }
  return { indices, weights };
}

try {
  for (const [numTokens, n, k, ties] of [[3, 256, 8, false], [2, 256, 8, true], [1, 64, 4, false], [2, 63, 6, true]]) {
    const logits = Float32Array.from({ length: numTokens * n }, () => random() * 8);
    if (ties) for (let i = 5; i < logits.length; i += 7) logits[i] = logits[i - 3];
    const input = upload(logits);
    const run = async (variant) => {
      const { indices, weights } = await runSoftmaxTopK(input, numTokens, n, k,
        { normalize: true, inputDtype: 'f32', weightsDtype: 'f32', variant });
      owned.push(indices, weights);
      return {
        indices: Array.from(new Uint32Array(await readBuffer(indices, numTokens * k * 4))),
        weights: Array.from(new Float32Array(await readBuffer(weights, numTokens * k * 4))),
      };
    };
    const ranked = await run('fused_logit_rank');
    const softmax = await run('fused_ranked');
    const expected = reference(logits, numTokens, n, k);
    assert.deepEqual(ranked.indices, expected.indices, `indices (tokens ${numTokens}, n ${n}, k ${k}, ties ${ties})`);
    assert.deepEqual(ranked.indices, softmax.indices, `fused_ranked picks other experts (n ${n}, k ${k})`);
    ranked.weights.forEach((weight, i) => {
      assert.ok(Math.abs(weight - expected.weights[i]) <= 2e-7 + 2e-6 * expected.weights[i], `weight ${i}: ${weight} vs ${expected.weights[i]}`);
      assert.ok(Math.abs(weight - softmax.weights[i]) <= 2e-7 + 2e-6 * softmax.weights[i], `weight ${i} vs fused_ranked`);
    });
  }
  await assert.rejects(runSoftmaxTopK(owned[0], 1, 256, 8, { normalize: false, inputDtype: 'f32', weightsDtype: 'f32',
    variant: 'fused_logit_rank' }), /renormalized F32 routing only/);
  console.log('softmax-topk-logit-rank-gpu.test: ok');
} finally {
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
