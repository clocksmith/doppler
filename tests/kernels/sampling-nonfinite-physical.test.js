import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { runArgmax, runGPUSample } from '../../src/gpu/kernels/sample.js';
import { readBuffer } from '../../src/memory/buffer-pool.js';
import { sample } from '../../src/inference/token-sampling.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical sampling diagnostic requires WebGPU: ${probe.reason}`);
const device = getDevice(), rows = [];
try {
  for (const size of [5, 131075]) {
    for (const prefix of [[NaN, Infinity, -Infinity, 2, 2], [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity],
      [-3.4028234663852886e38, -3.4028234663852886e38, -Infinity, NaN, Infinity]]) {
      const values = new Float32Array(size).fill(-Infinity); values.set(prefix);
      const buffer = device.createBuffer({ size: values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(buffer, 0, values);
      try {
        const returnedBits = new Uint32Array(await readBuffer(buffer));
        assert.deepEqual(returnedBits, new Uint32Array(values.buffer), 'scores reach the shader unchanged');
        for (const temperature of [0, 1]) {
          const options = { logitsDtype: 'f32', outputIndex: 0, logitSoftcap: null, padTokenId: 3,
            temperature, topK: 4, topP: 1, randomSeed: 7, greedyThreshold: 0 };
          let expected, actual;
          try { expected = sample(values.slice(), { ...options, seed: 7 }); } catch { expected = 'no-finite-candidate'; }
          try { actual = temperature === 0 ? await runArgmax(buffer, size, options) : await runGPUSample(buffer, size, options); }
          catch (error) { if (!/finite candidate/.test(error.message)) throw error; actual = 'no-finite-candidate'; }
          rows.push({ size, inputPrefixBits: Array.from(returnedBits.slice(0, 5)), temperature, expected, actual });
        }
      } finally { buffer.destroy(); }
    }
  }
  console.log(JSON.stringify({ schema: 'doppler.sampling-nonfinite-diagnostic/v1', adapter: getKernelCapabilities().adapterInfo, rows }));
  assert(rows.every(row => row.expected === row.actual), 'GPU and CPU must agree on invalid candidates and minimum finite scores');
} finally { destroyDevice(); }
