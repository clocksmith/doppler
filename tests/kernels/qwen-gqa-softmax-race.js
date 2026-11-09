import { getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createPipeline } from '../../src/gpu/kernels/pipeline-cache.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { createKernelUniformBuffer } from '../../src/gpu/kernels/uniform-utils.js';
import { releaseUniformBuffer } from '../../src/gpu/uniform-cache.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { runQwenGqaAttentionBackward } from '../../src/gpu/kernels/backward/qwen_gqa_attention_backward.js';
import { computeAttentionSoftmaxData, computeAttentionBackwardData } from '../../src/experimental/training/attention-backward.js';

function compare(actual, expected, tolerance, label) {
  if (actual.length !== expected.length) throw new Error(`${label}: shape mismatch`);
  let maxAbs = 0;
  for (let i = 0; i < expected.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || error > tolerance) throw new Error(`${label}: element=${i} actual=${actual[i]} expected=${expected[i]}`);
    maxAbs = Math.max(maxAbs, error);
  }
  return maxAbs;
}

export async function checkQwenGqaSoftmaxRace() {
  const device = getDevice(), reports = [], iterations = 8;
  const pipeline = await createPipeline('qwen_gqa_softmax_recompute', 'default');
  for (const seqLen of [33, 58, 129]) for (const causal of [true, false]) {
    const numHeads = 4, numKVHeads = 2, headDim = 8;
    const options = { seqLen, numHeads, numKVHeads, headDim, scale: Math.fround(1 / Math.sqrt(headDim)), causal };
    const query = Float32Array.from({ length: seqLen * numHeads * headDim }, (_, i) => Math.sin(i * 0.071) * 3);
    const key = Float32Array.from({ length: seqLen * numKVHeads * headDim }, (_, i) => Math.cos(i * 0.113 + 0.3) * 2);
    const value = Float32Array.from({ length: key.length }, (_, i) => Math.sin(i * 0.17) * 0.4);
    const gradOutput = Float32Array.from({ length: query.length }, (_, i) => Math.cos(i * 0.137) * 0.25);
    const reference = computeAttentionSoftmaxData(query, key, options);
    const expected = new Float32Array(reference.length);
    for (let token = 0; token < seqLen; token++) for (let head = 0; head < numHeads; head++) {
      expected.set(reference.subarray((head * seqLen + token) * seqLen, (head * seqLen + token + 1) * seqLen), (token * numHeads + head) * seqLen);
    }
    const owned = [];
    const tensor = (data, shape, label) => {
      const buffer = acquireBuffer(data.byteLength, undefined, label); owned.push(buffer);
      uploadData(buffer, data); return createTensor(buffer, 'f32', shape, label);
    };
    let uniform;
    try {
      const q = tensor(query, [seqLen, numHeads, headDim], 'softmax_race_q');
      const k = tensor(key, [seqLen, numKVHeads, headDim], 'softmax_race_k');
      const v = tensor(value, [seqLen, numKVHeads, headDim], 'softmax_race_v');
      const d = tensor(gradOutput, [seqLen, numHeads, headDim], 'softmax_race_gradient');
      const output = acquireBuffer(expected.byteLength, undefined, 'softmax_race_output'); owned.push(output);
      uniform = createKernelUniformBuffer('softmax_race_uniforms', getKernelConfig('qwen_gqa_softmax_recompute', 'default'), {
        seq_len: seqLen, num_heads: numHeads, num_kv_heads: numKVHeads, head_dim: headDim, scale: options.scale, causal: causal ? 1 : 0, _pad0: 0, _pad1: 0,
      });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: q.buffer } },
        { binding: 2, resource: { buffer: k.buffer } }, { binding: 3, resource: { buffer: output } },
      ] });
      let baseline, maxAbs = 0;
      for (let iteration = 0; iteration < iterations; iteration++) {
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(seqLen * numHeads); pass.end();
        device.queue.submit([encoder.finish()]);
        const bytes = await readBuffer(output, expected.byteLength), actual = new Float32Array(bytes);
        maxAbs = Math.max(maxAbs, compare(actual, expected, 2e-5, `Qwen softmax T=${seqLen} causal=${causal} iteration=${iteration}`));
        const bits = new Uint32Array(bytes);
        if (baseline && bits.some((x, i) => x !== baseline[i])) throw new Error(`Qwen softmax changed across identical dispatches T=${seqLen} causal=${causal}`);
        baseline ||= bits;
        for (let row = 0; row < seqLen * numHeads; row++) {
          let sum = 0; for (let col = 0; col < seqLen; col++) sum += actual[row * seqLen + col];
          if (Math.abs(sum - 1) > 2e-5) throw new Error(`Qwen softmax row ${row} sums to ${sum}`);
        }
      }
      const expectedGradients = computeAttentionBackwardData(query, key, value, reference, gradOutput, options);
      const gradients = await runQwenGqaAttentionBackward(q, k, v, d, options), gradientErrors = {};
      try {
        for (const [name, expectedName] of [['query', 'dQ'], ['key', 'dK'], ['value', 'dV']]) {
          const actual = new Float32Array(await readBuffer(gradients[name].buffer, expectedGradients[expectedName].byteLength));
          gradientErrors[name] = compare(actual, expectedGradients[expectedName], 1e-4, `Qwen GQA ${name} gradient T=${seqLen} causal=${causal}`);
        }
      } finally { Object.values(gradients).forEach(t => releaseBuffer(t.buffer)); }
      reports.push({ seqLen, numHeads, numKVHeads, headDim, causal, iterations, bitStable: true, maxAbs, gradientErrors });
    } finally {
      await device.queue.onSubmittedWorkDone();
      if (uniform) releaseUniformBuffer(uniform);
      owned.forEach(releaseBuffer);
    }
  }
  return reports;
}
