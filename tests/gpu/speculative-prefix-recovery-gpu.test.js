import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';
import { createLinearAttentionRecovery } from '../../src/gpu/kernels/linear-attention-recovery.js';
import { recordSpeculativeAcceptance, createSpeculativeRecoveryUniform, recordSpeculativeCarry } from '../../src/gpu/kernels/speculative-accept.js';

// Physical recovery evidence, not full MTP qualification: retained projection inputs,
// GPU acceptance and recurrent-prefix recovery share one submission, with no full-logit readback.
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice(), errors = [], owned = [], cases = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = values => {
  const buffer = acquireBuffer(values.byteLength); owned.push(buffer); uploadData(buffer, values); return buffer;
};
const values = (n, rate, scale) => Float32Array.from({ length: n }, (_, i) => Math.sin((i + 1) * rate) * scale);
const words = async (buffer, n) => new Uint32Array(await readBuffer(buffer, n * 4));
const NK = 2, NV = 4, K = 128, V = 128, window = 4, T = 4;
const Q = NK * K, D = NV * V, C = 2 * Q + D;
const initial = {
  convWeight: values(C * window, 0.011, 0.5), convState: values(C * window, 0.021, 0.05),
  bias: values(NV, 0.19, 0.2), aLog: Float32Array.from({ length: NV }, (_, h) => -0.25 - h * 0.07),
  norm: Float32Array.from({ length: D }, (_, i) => 0.7 + (i % 13) * 0.04), state: values(NV * K * V, 0.007, 0.005),
};
const makeState = () => ({
  sourceLayout: 'gguf-v1', convDim: C, convKernelSize: window, valueDim: D, numVHeads: NV, numKHeads: NK,
  headKDim: K, headVDim: V, qSize: Q, kSize: Q, qRep: NV / NK, normMode: 'per_head', rmsNormEps: 1e-6,
  convWeightGPU: upload(initial.convWeight), dtBiasGPU: upload(initial.bias), aLogGPU: upload(initial.aLog),
  normWeightGPU: upload(initial.norm), convStateGPU: upload(initial.convState), recurrentStateGPU: upload(initial.state),
  recurrentStatePrecision: 'f32', recurrentStateComponents: 1,
});
const inputs = [values(T * C, 0.019, 0.2), values(T * D, 0.029, 0.8), values(T * NV, 0.11, 1.2), values(T * NV, 0.09, 2)]
  .map((data, i) => createTensor(upload(data), 'f32', [T, [C, D, NV, NV][i]]));
const options = register => ({
  numTokens: T, normGateActivation: 'silu', outputDtype: 'f32', qkL2NormEps: 1e-6,
  runtimeSession: { useLinearAttentionRegisterPrefill: register, useLinearAttentionRegisterDecode: register,
    useLinearAttentionFusedDecodeCore: false },
});
try {
  // Arbitrary objects cannot authorize overwriting a uniform frame with a GPU count.
  const badRecorder = createCommandRecorder('forged_acceptance');
  assert.throws(() => createSpeculativeRecoveryUniform(badRecorder, { buffer: inputs[0].buffer, maxTokens: T }, new ArrayBuffer(64), T), /recorded acceptance/);
  await badRecorder.abort();
  for (const register of [false, true]) {
    for (const [mismatch, limit, invalid] of [[0,4,false], [1,4,false], [2,4,false], [3,4,false], [3,1,false], [3,2,false], [3,4,true]]) {
      const count = invalid ? 0 : Math.min(mismatch + 1, limit);
      const reference = makeState(), recovered = makeState();
      if (count > 0) {
        const output = await runLinearAttentionCoreGPU(...inputs, reference, { ...options(register), numTokens: count });
        owned.push(output.buffer);
      }
      const recorder = createCommandRecorder('speculative_prefix_recovery');
      const recovery = createLinearAttentionRecovery(recorder, T);
      let submitted = false;
      try {
        const tentative = await runLinearAttentionCoreGPU(...inputs, recovered, { ...options(register), recorder });
        recorder.trackTemporaryBuffer(tentative.buffer);
        const drafts = upload(new Uint32Array([10, 11, 12]));
        const targetIDs = [10, 11, 12, 13];
        if (mismatch < 3) targetIDs[mismatch] = 99;
        if (invalid) targetIDs[3] = 200;
        const result = acquireBuffer(36); owned.push(result);
        const acceptance = await recordSpeculativeAcceptance(recorder, drafts, upload(new Uint32Array(targetIDs)), {
          draftCount: 3, outputLimit: limit, vocabSize: 200, outputBuffer: result,
        });
        assert.equal(recovery.layerCount, 1);
        await recovery.recordRecover(recorder, acceptance);
        const hiddenRows = Float32Array.from({ length: T * 4 }, (_, i) => i - 5.5);
        const carryHidden = upload(new Float32Array(4)), carryToken = upload(new Uint32Array(1));
        await recordSpeculativeCarry(recorder, acceptance, createTensor(upload(hiddenRows), 'f32', [T, 4]), 4, carryHidden, carryToken);
        recorder.submit({ cleanup: 'deferred' }); submitted = true;
        await device.queue.onSubmittedWorkDone();
        const compact = await words(result, 9);
        assert.deepEqual(Array.from(compact.slice(0, 3)), [count, invalid ? 0 : Math.min(mismatch, limit), invalid ? 1 : 0]);
        if (!invalid) {
          assert.deepEqual(Array.from(new Float32Array(await readBuffer(carryHidden, 16))), Array.from(hiddenRows.slice((count - 1) * 4, count * 4)));
          assert.equal((await words(carryToken, 1))[0], targetIDs[count - 1]);
        }
        assert.deepEqual(await words(recovered.convStateGPU, C * window), await words(reference.convStateGPU, C * window), 'conv prefix differs');
        assert.deepEqual(await words(recovered.recurrentStateGPU, NV * K * V), await words(reference.recurrentStateGPU, NV * K * V), 'recurrent prefix differs');
        assert.equal(recorder.getStats().submissionCount, 1);
        await recorder.completeDeferredCleanup();
        // A continuation must see exactly the recovered prefix, including the zero-update invalid case.
        const next = [];
        for (const layer of [reference, recovered]) {
          const continuation = await runLinearAttentionCoreGPU(...inputs, layer, { ...options(register), numTokens: 1 });
          owned.push(continuation.buffer); next.push(await words(continuation.buffer, D));
        }
        assert.deepEqual(next[0], next[1], 'continuation differs');
        cases.push({ register, mismatch, limit, invalid, recoveredTokens: count, exactStateAndContinuation: true });
      } finally {
        recovery.close();
        assert.throws(() => recovery.recordRestore(recorder), /live parent lease/);
        if (submitted) await recorder.completeDeferredCleanup(); else await recorder.abort();
      }
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'passed', scope: 'GPU prefix recovery component, not full MTP', cases }));
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
