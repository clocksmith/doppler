import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const capabilities = getKernelCapabilities();
assert.match(JSON.stringify(capabilities.adapterInfo).toLowerCase(), /apple|m3/);
const T = 257, D = 128, NK = 1, NV = 2, C = (2 * NK + NV) * D, V = NV * D, KERNEL = 4;
const owned = new Set();
const upload = values => {
  const buffer = acquireBuffer(values.byteLength);
  uploadData(buffer, values);
  owned.add(buffer);
  return buffer;
};
const values = (count, factor) => Float32Array.from({ length: count }, (_, i) => Math.sin(i * factor) * 0.1);
const qkv = values(T * C, 0.013), z = values(T * V, 0.017), a = values(T * NV, 0.011), b = values(T * NV, 0.019);
const makeState = () => ({ sourceLayout: 'gguf-v1', convDim: C, convKernelSize: KERNEL,
  valueDim: V, numVHeads: NV, numKHeads: NK, headKDim: D, headVDim: D, qSize: NK * D, kSize: NK * D,
  qRep: NV / NK, normMode: 'shared', rmsNormEps: 0.000001, normWeightOffset: false,
  convWeightGPU: upload(values(C * KERNEL, 0.023)), dtBiasGPU: upload(values(NV, 0.031)),
  aLogGPU: upload(new Float32Array(NV).fill(-1)), normWeightGPU: upload(new Float32Array(D).fill(1)),
  convStateGPU: upload(new Float32Array(C * KERNEL)),
  recurrentStateGPU: upload(new Float32Array(NV * D * D)), recurrentStatePrecision: 'f32', recurrentStateComponents: 1,
});
const runtimeSession = { useLinearAttentionRegisterPrefill: false, useLinearAttentionRegisterDecode: false,
  useLinearAttentionFusedDecodeCore: false };
const execute = async lengths => {
  const state = makeState();
  const outputs = new Float32Array(T * V);
  let offset = 0;
  for (const count of lengths) {
    const tensor = (array, width) => createTensor(upload(array.subarray(offset * width, (offset + count) * width)), 'f32', [count, width]);
    const output = await runLinearAttentionCoreGPU(tensor(qkv, C), tensor(z, V), tensor(a, NV), tensor(b, NV), state,
      { numTokens: count, normGateActivation: 'silu', outputDtype: 'f32', qkL2NormEps: 0.000001, runtimeSession });
    owned.add(output.buffer);
    outputs.set(new Float32Array(await readBuffer(output.buffer, count * V * 4)), offset * V);
    offset += count;
  }
  assert.equal(offset, T);
  return { output: outputs,
    conv: new Float32Array(await readBuffer(state.convStateGPU, C * KERNEL * 4)),
    recurrent: new Float32Array(await readBuffer(state.recurrentStateGPU, NV * D * D * 4)) };
};
try {
  const whole = await execute([257]);
  const chunked = await execute([256, 1]);
  const errors = {};
  for (const field of ['output', 'conv', 'recurrent']) {
    let maxAbs = 0;
    for (let i = 0; i < whole[field].length; i++) {
      assert.ok(Number.isFinite(chunked[field][i]));
      maxAbs = Math.max(maxAbs, Math.abs(whole[field][i] - chunked[field][i]));
    }
    assert.ok(maxAbs <= 0.00001, `${field}: ${maxAbs}`);
    errors[field] = maxAbs;
  }
  console.log(JSON.stringify({ timeUTC: new Date().toISOString(), providerReceipt: probe.providerReceipt,
    adapter: capabilities.adapterInfo, status: 'passed', errors,
    geometry: { tokens: T, headDim: D, keyHeads: NK, valueHeads: NV, convKernelSize: KERNEL },
    scope: 'Reduced-head synthetic native F32 linear-core continuation; not35B fullmodel, AMD qualification, watchdog attribution or performance' }));
} finally {
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
