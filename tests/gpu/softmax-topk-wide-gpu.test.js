import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer, getBufferPool, destroyBufferPool } from '../../src/memory/buffer-pool.js';
import { runSoftmaxTopK, recordSoftmaxTopK } from '../../src/gpu/kernels/softmax.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  assert.ok(!process.argv.includes('--require-gpu'), gpu.reason);
  console.log(`wide top-k GPU test skipped: ${gpu.reason}`);
} else {
  const errors = [];
  getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
  try {
    for (const n of [63, 256, 257, 511, 512]) for (const normalize of [true, false]) {
      const k = 10, tokens = 3;
      // Non-tied permutation, high-index winners, and ties exercise both halves.
      const logits = Float32Array.from({ length: tokens * n }, (_, i) => {
        const token = Math.floor(i / n), index = i % n;
        return token === 0 ? ((index * 137) % 521) / 32 : token === 1 ? index / 32 : 1;
      });
      const owned = [];
      try {
        const input = acquireBuffer(logits.byteLength); owned.push(input); uploadData(input, logits);
        for (const recorded of [false, true]) {
          const recorder = recorded ? new CommandRecorder(getDevice(), 'wide_topk') : null;
          const options = { inputDtype: 'f32', weightsDtype: 'f32', normalize, variant: 'fused_ranked_512' };
          const output = recorder ? await recordSoftmaxTopK(recorder, input, tokens, n, k, options)
            : await runSoftmaxTopK(input, tokens, n, k, options);
          owned.push(output.indices, output.weights);
          if (recorder) await recorder.submitAndWait();
          const indices = new Uint32Array(await readBuffer(output.indices, tokens * k * 4));
          const weights = new Float32Array(await readBuffer(output.weights, tokens * k * 4));
          for (let token = 0; token < tokens; token++) {
            const row = [...logits.subarray(token * n, (token + 1) * n)];
            const order = row.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value || a.index - b.index);
            const denominator = (normalize ? order.slice(0, k) : order).reduce((sum, entry) => sum + Math.exp(entry.value - order[0].value), 0);
            for (let pick = 0; pick < k; pick++) {
              assert.equal(indices[token * k + pick], order[pick].index, `n=${n}, recorded=${recorded}`);
              const expected = Math.exp(order[pick].value - order[0].value) / denominator;
              assert.ok(Math.abs(weights[token * k + pick] - expected) <= 2e-7 + 2e-6 * expected);
            }
          }
        }
        await assert.rejects(runSoftmaxTopK(input, 1, 513, k,
          { inputDtype: 'f32', weightsDtype: 'f32', variant: 'fused_ranked_512' }), /at most 512/);
        if (n > 256) await assert.rejects(runSoftmaxTopK(input, tokens, n, k,
          { inputDtype: 'f32', weightsDtype: 'f32', variant: 'fused_ranked' }), /at most 256/);
      } finally {
        await getDevice().queue.onSubmittedWorkDone();
        for (const buffer of owned) releaseBuffer(buffer);
      }
    }
    assert.deepEqual(errors, []);
    console.log('physical wide top-k: 63/256/257/511/512 experts, top-10, normalized/raw, ties, immediate/recorded passed');
  } finally {
    await getBufferPool().settleDeferredDestruction();
    assert.equal(getBufferPool().getStats().activeBuffers, 0);
    destroyBufferPool(); destroyDevice(); await releaseNodeWebGPU();
  }
}
