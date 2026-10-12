import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, getBufferPool, readBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runDenseFFNGPU } from '../../src/inference/pipelines/text/ffn/dense.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice(), H = 128, I = 64, T = 2;
  const pair = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
  const input = Float32Array.from({ length: T * H * 2 }, (_, i) =>
    i % 2 ? ((i % 7) - 3) * 1e-9 : ((i % 19) - 9) / 32);
  const gate = Float32Array.from({ length: I * H }, (_, i) => ((i * 3 % 31) - 15) / 64);
  const up = Float32Array.from({ length: I * H }, (_, i) => ((i * 7 % 37) - 18) / 64);
  const down = Float32Array.from({ length: H * I }, (_, i) => ((i * 11 % 41) - 20) / 64);
  const owned = [];
  const upload = (data) => {
    const buffer = acquireBuffer(data.byteLength, undefined, 'paired_ffn_control');
    owned.push(buffer); device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  const tensor = createTensor(upload(input), 'f32', [T, H], 'paired_input', pair);
  const weights = { gate: createWeightBuffer(upload(gate), 'f32', 'row', [I, H]),
    up: createWeightBuffer(upload(up), 'f32', 'row', [I, H]),
    down: createWeightBuffer(upload(down), 'f32', 'row', [H, I]) };
  const projection = op => ({ op, kernel: 'matmul_paired.wgsl', entry: 'main',
    constants: { WEIGHT_F16: false, WEIGHT_Q4K: false, SCALAR_OUTPUT: false } });
  const steps = [projection('gate_proj'), projection('up_proj'),
    { op: 'activation', kernel: 'paired_silu.wgsl', entry: 'main' }, projection('down_proj')];
  const base = { config: { hiddenSize: H, intermediateSize: I, hiddenActivation: 'silu',
    swigluLimit: null, gatedActivation: true, useDoubleWideMlp: false }, phase: 'prefill',
    kernelPath: { prefill: { steps }, decode: { steps } } };
  const reference = [];
  for (let t = 0; t < T; t++) {
    const activation = [];
    for (let n = 0; n < I; n++) {
      let g = 0, u = 0;
      for (let k = 0; k < H; k++) {
        const x = input[(t * H + k) * 2] + input[(t * H + k) * 2 + 1];
        g += x * gate[n * H + k]; u += x * up[n * H + k];
      }
      activation.push(g / (1 + Math.exp(-g)) * u);
    }
    for (let n = 0; n < H; n++) reference.push(activation.reduce((sum, x, k) => sum + x * down[n * I + k], 0));
  }
  const results = [];
  try {
    const active = getBufferPool().getStats().activeBuffers;
    for (const recorded of [false, true]) {
      const recorder = recorded ? new CommandRecorder(device, 'paired_ffn_control') : null;
      let result;
      try {
        result = await runDenseFFNGPU(0, tensor, T, { ...base, recorder }, weights);
        if (recorder) await recorder.submitAndWait();
        assert.equal(result.precision, 'two-f32'); assert.deepEqual(result.shape, [T, H]);
        const components = new Float32Array(await readBuffer(result.buffer, T * H * 8));
        let maximum = 0;
        for (let i = 0; i < reference.length; i++) {
          const actual = components[i * 2] + components[i * 2 + 1];
          maximum = Math.max(maximum, Math.abs(actual - reference[i]));
        }
        assert(maximum <= 1e-11, `paired FFN maximum error ${maximum}`);
        results.push({ recorded, maximum, components: Array.from(components) });
      } finally { if (recorder) await recorder.abort(); if (result) releaseBuffer(result.buffer); }
      assert.equal(getBufferPool().getStats().activeBuffers, active);
    }
    console.log(JSON.stringify({ schema: 'doppler.paired-ffn-control/v1', passed: true,
      tolerance: 1e-11, shape: [T, H, I], input: Array.from(input), gate: Array.from(gate),
      up: Array.from(up), down: Array.from(down), reference, results, provider: probe.providerReceipt }));
  } finally { for (const buffer of owned) releaseBuffer(buffer); }
} else console.log(JSON.stringify({ passed: false, skipped: probe.reason }));
destroyDevice();
