import assert from 'node:assert/strict';

// GGUF register decode runs the GGUF register recurrent program at one token. It must
// leave the same bits as the unfused GGUF program: output, recurrent state and conv
// state, over consecutive decode tokens. NK=2 / NV=4 exercises the head % NK mapping.

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`gguf-linear-attention-register-decode-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { destroyDevice } = await import('../../src/gpu/device.js');
const { acquireBuffer, releaseBuffer, uploadData, readBuffer } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { runLinearAttentionCoreGPU } = await import('../../src/gpu/kernels/linear-attention-core.js');

const owned = [];
const upload = (values) => {
  const buffer = acquireBuffer(values.byteLength);
  uploadData(buffer, values);
  owned.push(buffer);
  return buffer;
};
const values = (n, rate, scale) => Float32Array.from({ length: n }, (_, i) => Math.sin((i + 1) * rate) * scale);
const words = async (buffer, count) => new Uint32Array(await readBuffer(buffer, count * 4));

const NK = 2, NV = 4, K = 128, V = 128, window = 4;
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

try {
  const layers = { unfused: makeState(), register: makeState() };
  for (let step = 0; step < 3; step++) {
    const qkv = values(C, 0.037 + step * 0.01, 0.2), z = values(D, 0.023 + step * 0.01, 0.8);
    const a = values(NV, 0.17 + step, 1.2), b = values(NV, 0.13 + step, 2);
    /** @type {Record<string, Uint32Array>} */
    const outputs = {};
    for (const [name, layer] of Object.entries(layers)) {
      const tensor = (data, shape) => createTensor(upload(data), 'f32', shape);
      const output = await runLinearAttentionCoreGPU(tensor(qkv, [1, C]), tensor(z, [1, D]), tensor(a, [1, NV]), tensor(b, [1, NV]), layer, {
        numTokens: 1, normGateActivation: 'silu', outputDtype: 'f32', qkL2NormEps: 1e-6,
        runtimeSession: { useLinearAttentionRegisterPrefill: false, useLinearAttentionFusedDecodeCore: false,
          useLinearAttentionRegisterDecode: name === 'register' },
      });
      owned.push(output.buffer);
      outputs[name] = await words(output.buffer, D);
    }
    assert.ok(outputs.unfused.some((word) => word !== 0), 'unfused output was never written');
    assert.deepEqual(outputs.register, outputs.unfused, `output differs at decode step ${step}`);
    assert.deepEqual(await words(layers.register.recurrentStateGPU, NV * K * V),
      await words(layers.unfused.recurrentStateGPU, NV * K * V), `recurrent state differs at decode step ${step}`);
    assert.deepEqual(await words(layers.register.convStateGPU, C * window),
      await words(layers.unfused.convStateGPU, C * window), `conv state differs at decode step ${step}`);
  }
  console.log('gguf-linear-attention-register-decode-gpu.test: ok');
} finally {
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
