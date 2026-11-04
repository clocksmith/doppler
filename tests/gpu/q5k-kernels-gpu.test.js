import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { runQ40Matmul, recordQ40Matmul } from '../../src/gpu/kernels/q4_0.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { ggufBlockBytes } from '../../src/formats/gguf/block-storage.js';
import { normalizeQuantTag, toWebGPUDtype } from '../../src/converter/quantization-info.js';
import { toMatmulDtype } from '../../src/gpu/kernels/matmul-selection.js';

// Independent scalar reference follows GGML dequantize_row_q5_K's 64-value order.
// Quantizer construction explicitly exercises all eight scale/min and high-bit groups.
function fixture(rows, cols) {
  const bytes = new Uint8Array(rows * cols / 256 * 176), view = new DataView(bytes.buffer);
  for (let b = 0; b < rows * cols / 256; b++) {
    const base = b * 176;
    view.setUint16(base, b % 3 === 0 ? 0 : 0x1400, true);
    view.setUint16(base + 2, b % 3 === 2 ? 0 : 0x0c00, true);
    for (let g = 0; g < 8; g++) {
      const scale = (g * 11 + b * 7) % 64, min = (g * 13 + b * 5) % 64;
      if (g < 4) { bytes[base + 4 + g] = scale; bytes[base + 8 + g] = min; }
      else {
        bytes[base + 8 + g] = (scale & 15) | ((min & 15) << 4);
        bytes[base + g] |= (scale >> 4) << 6;
        bytes[base + 4 + g] |= (min >> 4) << 6;
      }
    }
    for (let i = 0; i < 32; i++) bytes[base + 16 + i] = (b * 37 + i * 53) % 256;
    for (let i = 0; i < 128; i++) bytes[base + 48 + i] = (b * 19 + i * 29) % 256;
  }
  return bytes;
}
function decode(bytes) {
  const values = new Float32Array(bytes.length / 176 * 256), view = new DataView(bytes.buffer);
  for (let b = 0; b < bytes.length / 176; b++) {
    const base = b * 176, scales = bytes.subarray(base + 4, base + 16);
    const d = view.getUint16(base, true) === 0 ? 0 : 2 ** -10;
    const minimum = view.getUint16(base + 2, true) === 0 ? 0 : 2 ** -12;
    for (let groupPair = 0; groupPair < 4; groupPair++) {
      for (let half = 0; half < 2; half++) {
        const g = groupPair * 2 + half;
        const scale = g < 4 ? scales[g] & 63 : (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4);
        const min = g < 4 ? scales[g + 4] & 63 : (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
        for (let l = 0; l < 32; l++) {
          const packed = bytes[base + 48 + groupPair * 32 + l];
          const q = (half ? packed >> 4 : packed & 15) + ((bytes[base + 16 + l] & (1 << g)) ? 16 : 0);
          values[b * 256 + g * 32 + l] = d * scale * q - minimum * min;
        }
      }
    }
  }
  return values;
}
assert.equal(normalizeQuantTag('Q5_K'), 'q5k');
assert.equal(toWebGPUDtype('q5k'), 'Q5_K');
assert.equal(toMatmulDtype('q5k'), 'q5k');
assert.equal(ggufBlockBytes([7, 512], 'Q5_K'), 7 * 2 * 176);
assert.throws(() => ggufBlockBytes([7, 513], 'Q5_K'), /complete quantization blocks/);
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice(), owned = [], errors = [], cases = [];
device.addEventListener('uncapturederror', e => errors.push(e.error.message));
const upload = data => {
  const buffer = acquireBuffer(Math.ceil(data.byteLength / 4) * 4); owned.push(buffer);
  uploadData(buffer, data); return buffer;
};
try {
  for (const [M, N, K] of [[1, 7, 256], [1, 17, 6144], [3, 7, 512], [4, 19, 512], [5, 19, 512], [7, 19, 2048], [17, 19, 512], [65, 67, 512]]) {
    const packed = fixture(2 * N, K), decoded = decode(packed);
    const input = Float32Array.from({ length: M * K + 4 }, (_, i) => Math.sin(i * 0.13) * 0.1);
    const A = createTensor(upload(input), 'f32', [input.length]);
    const B = createWeightBuffer(upload(packed), 'q5k', 'row', [2, N, K]);
    const expected = new Float32Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
      let value = 0;
      for (let k = 0; k < K; k++) value += input[4 + m * K + k] * decoded[(n + 1) * K + k];
      expected[m * N + n] = value * 0.5;
    }
    for (const recorded of [false, true]) {
      const recorder = recorded ? createCommandRecorder('q5k-reference') : null;
      const options = { outputDtype: 'f32', alpha: 0.5, aOffset: 16, bOffset: K / 256 * 176 };
      const out = recorder ? await recordMatmul(recorder, A, B, M, N, K, options)
        : await runMatmul(A, B, M, N, K, options);
      owned.push(out.buffer); if (recorder) await recorder.submitAndWait();
      const actual = new Float32Array(await readBuffer(out.buffer, M * N * 4)); let maxAbs = 0;
      for (let i = 0; i < actual.length; i++) {
        assert.ok(Number.isFinite(actual[i]));
        maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
        assert.ok(Math.abs(actual[i] - expected[i]) < 0.0002, `${M}/${N}/${K} at ${i}: ${actual[i]} vs ${expected[i]}`);
      }
      cases.push({ M, N, K, recorded, maxAbs });
    }
    for (const recorded of [false, true]) {
      const results = [];
      for (const variant of ['q5k_gemv', 'q5k_gemv_blockwise', 'q5k_gemv_blockwise_columns']) {
        const recorder = recorded ? createCommandRecorder('q5k-blockwise-parity') : null;
        const guardedOutput = upload(new Float32Array(M * N + 8).fill(-12345));
        const options = { outputDtype: 'f32', alpha: 0.5, aOffset: 16, bOffset: K / 256 * 176,
          cOffset: 16, outputBuffer: guardedOutput, variant };
        const out = recorder ? await recordQ40Matmul(recorder, A, B, M, N, K, options)
          : await runQ40Matmul(A, B, M, N, K, options);
        if (recorder) await recorder.submitAndWait();
        const guarded = new Float32Array(await readBuffer(out.buffer, (M * N + 8) * 4));
        assert.ok(guarded.subarray(0, 4).every(value => value === -12345), 'Q5_K output prefix changed');
        assert.ok(guarded.subarray(4 + M * N).every(value => value === -12345), 'Q5_K output tail changed');
        results.push(new Uint32Array(guarded.buffer, 16, M * N).slice());
      }
      assert.deepEqual(results[1], results[0], `Blockwise Q5_K changed output bits at ${M}/${N}/${K}, recorded=${recorded}`);
      assert.deepEqual(results[2], results[0], `Four-row Q5_K changed output bits at ${M}/${N}/${K}, recorded=${recorded}`);
      cases.push({ M, N, K, recorded, blockwiseBitwiseParity: true, columnsBitwiseParity: true });
    }
    const cooperative = new Map();
    for (const recorded of [false, true]) for (const variant of ['q5k_cooperative', 'q5k_cooperative_columns', 'q5k_prefill_tile64']) {
      const recorder = recorded ? createCommandRecorder('q5k-cooperative-reference') : null;
      const guardedOutput = upload(new Float32Array(M * N + 8).fill(-12345));
      const options = { outputDtype: 'f32', alpha: 0.5, aOffset: 16, bOffset: K / 256 * 176,
        cOffset: 16, outputBuffer: guardedOutput, variant };
      const out = recorder ? await recordQ40Matmul(recorder, A, B, M, N, K, options)
        : await runQ40Matmul(A, B, M, N, K, options);
      if (recorder) await recorder.submitAndWait();
      const guarded = new Float32Array(await readBuffer(out.buffer, (M * N + 8) * 4));
      assert.ok(guarded.subarray(0, 4).every(value => value === -12345), 'Cooperative Q5_K output prefix changed');
      assert.ok(guarded.subarray(4 + M * N).every(value => value === -12345), 'Cooperative Q5_K output tail changed');
      const actual = guarded.slice(4, 4 + M * N); let maxAbs = 0;
      for (let i = 0; i < actual.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
        assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) < 0.0002,
          `${variant}/${M}/${N}/${K}/${recorded} at ${i}`);
      }
      if (recorded) assert.deepEqual(actual, cooperative.get(variant), 'Cooperative immediate and recorded arithmetic differ');
      else cooperative.set(variant, actual);
      cases.push({ M, N, K, recorded, variant, maxAbs, independentReference: true });
    }
    await assert.rejects(runQ40Matmul(A, B, M, N, K,
      { outputDtype: 'f32', variant: 'q5k_cooperative', aOffset: 4 }), /16-byte/);
    await assert.rejects(runMatmul(A, B, M, N, K, { outputDtype: 'f16' }), /F32/);
    await assert.rejects(runMatmul(A, B, M, N, K, { outputDtype: 'f32', outputBuffer: A.buffer }), /alias|undersized/);
  }
  if (process.env.DOPPLER_Q5K_FIXTURE_DIR) {
    const directory = process.env.DOPPLER_Q5K_FIXTURE_DIR;
    const fixture = JSON.parse(fs.readFileSync(path.join(directory, 'fixture.json'), 'utf8'));
    const data = {};
    for (const name of ['weights.bin', 'input.bin', 'expected.bin']) {
      const bytes = fs.readFileSync(path.join(directory, name));
      assert.equal(createHash('sha256').update(bytes).digest('hex'), fixture.files[name]);
      data[name] = new Uint8Array(bytes);
    }
    const M = 1, N = fixture.rows, K = fixture.cols;
    assert.equal(data['input.bin'].byteLength, K * 4);
    assert.equal(data['weights.bin'].byteLength, ggufBlockBytes([N, K], 'q5k'));
    const A = createTensor(upload(data['input.bin']), 'f32', [1, K]);
    const B = createWeightBuffer(upload(data['weights.bin']), 'q5k', 'row', [N, K]);
    const expected = new Float32Array(data['expected.bin'].buffer);
    assert.equal(expected.length, N);
    const out = await runMatmul(A, B, M, N, K, { outputDtype: 'f32' }); owned.push(out.buffer);
    const actual = new Float32Array(await readBuffer(out.buffer, N * 4)); let maxAbs = 0;
    for (let i = 0; i < N; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
      assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) < 0.0002);
    }
    const blockwise = await runQ40Matmul(A, B, M, N, K, { outputDtype: 'f32', variant: 'q5k_gemv_blockwise' });
    owned.push(blockwise.buffer);
    const blockwiseValues = new Float32Array(await readBuffer(blockwise.buffer, N * 4));
    assert.deepEqual(new Uint32Array(blockwiseValues.buffer), new Uint32Array(actual.buffer), 'Blockwise real source projection differs from baseline');
    cases.push({ sourceTensor: fixture.sourceTensor, reference: fixture.reference, M, N, K, maxAbs });
  }
  await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, scope: 'Q5_K immediate and recorded projections, all scale/min groups, offsets and partial output tiles', cases, gpuErrors: errors }));
} finally { for (const buffer of owned) releaseBuffer(buffer); destroyDevice(); }
