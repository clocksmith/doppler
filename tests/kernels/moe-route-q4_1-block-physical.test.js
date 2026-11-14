import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { runRouteQ4Matmul, recordRouteQ4Matmul } from '../../src/gpu/kernels/moe-route.js';
import { packedFixture, decodePacked } from '../helpers/q4_0-fixture.js';

// Independent decoded-weight dot oracle; signed scale/minimum, invalid route, negative alpha,
// tails in output columns, blocks spanning several lanes, immediate and recorded paths.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Native Q4_1 block oracle needs physical WebGPU: ${probe.reason}`);
const device = getDevice();
const errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
try {
  for (const K of [32, 96, 640, 2560]) {
    const E = 3, I = 7, M = 4, topK = 2, N = I * 2, alpha = -0.75;
    const gate = packedFixture('q4_1', E * N, K);
    const decoded = decodePacked('q4_1', gate, E * N, K);
    const routes = new Uint32Array([2, 0, 1, E]);
    for (const inputMode of ['token', 'route']) {
      const rows = inputMode === 'token' ? M / topK : M;
      const values = Float32Array.from({ length: rows * K }, (_, i) => Math.sin(i * 0.13) * 0.1);
      const expected = new Float32Array(M * N);
      for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
        if (routes[m] >= E) continue;
        const token = inputMode === 'token' ? Math.floor(m / topK) : m;
        const weights = decoded;
        let sum = 0;
        for (let k = 0; k < K; k++) sum += values[token * K + k] * weights[(routes[m] * N + n) * K + k];
        expected[m * N + n] = sum * alpha;
      }
      const owned = [];
      const upload = data => {
        const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
        padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        const buffer = acquireBuffer(padded.byteLength);
        owned.push(buffer);
        uploadData(buffer, padded);
        return buffer;
      };
      try {
        const input = createTensor(upload(values), 'f32', [rows, K]);
        const indices = upload(routes);
        const weights = createWeightBuffer(upload(gate), 'q4_1', 'row', [E, N, K]);
        for (const recorded of [false, true]) {
          const recorder = recorded ? createCommandRecorder('q4_1-block-oracle') : null;
          let output;
          try {
            const options = { numRoutes: M, numExperts: E, topK, N, K, inputMode, alpha };
            output = recorder ? await recordRouteQ4Matmul(recorder, input, indices, weights, options)
              : await runRouteQ4Matmul(input, indices, weights, options);
            if (recorder) await recorder.submitAndWait();
            const actual = new Float32Array(await readBuffer(output.buffer, expected.byteLength));
            let maxAbs = 0;
            for (let i = 0; i < actual.length; i++) {
              maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
              assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) <= 2e-5 + 2e-5 * Math.abs(expected[i]),
                `K=${K}, ${inputMode}, recorded=${recorded}, i=${i}: ${actual[i]} != ${expected[i]}`);
            }
            console.log(JSON.stringify({ K, inputMode, recorded, values: actual.length, maxAbs }));
          } finally {
            if (output) releaseBuffer(output.buffer);
            if (recorder) await recorder.abort();
          }
        }
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) releaseBuffer(buffer);
      }
    }
  }
  assert.deepEqual(errors, []);
} finally { destroyDevice(); }
