import { getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { float32ToFloat16, float16ToFloat32 } from '../../src/formats/float16.js';
import { runRMSNorm, recordRMSNorm, selectRMSNormKernel } from '../../src/gpu/kernels/rmsnorm.js';

export async function checkSmallF16RmsNorm() {
  const rows = 58, eps = 1e-6, iterations = 8, reports = [];
  for (const size of [64, 128, 256]) {
    const input = new Uint16Array(rows * size), expected = new Float32Array(input.length);
    for (let row = 0; row < rows; row++) {
      const amplitude = [0.25, 4, 32, 128, 512][row % 5];
      let squares = 0;
      for (let col = 0; col < size; col++) {
        const index = row * size + col;
        input[index] = float32ToFloat16(amplitude * Math.sin(index * 0.071 + row * 0.31));
        const value = float16ToFloat32(input[index]);
        squares += value * value;
      }
      const inverse = 1 / Math.sqrt(squares / size + eps);
      for (let col = 0; col < size; col++) {
        const index = row * size + col;
        expected[index] = float16ToFloat32(input[index]) * inverse;
      }
    }
    const options = { batchSize: rows, hiddenSize: size, rmsNormWeightOffset: false };
    if (selectRMSNormKernel(options, true) !== 'small_f16') throw new Error('Small F16 regression did not select the affected kernel');
    const inputBuffer = acquireBuffer(input.byteLength, undefined, 'small_f16_race_input');
    const weightBuffer = acquireBuffer(size * 4, undefined, 'small_f16_race_weight');
    uploadData(inputBuffer, input); uploadData(weightBuffer, new Float32Array(size).fill(1));
    const tensor = createTensor(inputBuffer, 'f16', [rows, size]);
    const weight = createWeightBuffer(weightBuffer, 'f32', 'row', [size]);
    let baseline = null, maxAbs = 0;
    try {
      for (const recorded of [false, true]) {
        for (let iteration = 0; iteration < iterations; iteration++) {
          const recorder = recorded ? new CommandRecorder(getDevice(), 'small_f16_race') : null;
          let output;
          try {
            output = recorder ? await recordRMSNorm(recorder, tensor, weight, eps, options)
              : await runRMSNorm(tensor, weight, eps, options);
            if (recorder) await recorder.submitAndWait();
            const actual = new Uint16Array(await readBuffer(output.buffer, input.byteLength));
            for (let i = 0; i < actual.length; i++) {
              const value = float16ToFloat32(actual[i]), error = Math.abs(value - expected[i]);
              if (!Number.isFinite(value) || error > 0.02) {
                throw new Error(`Small F16 RMSNorm size=${size} row=${Math.floor(i / size)} recorded=${recorded} iteration=${iteration}: actual=${value} expected=${expected[i]}`);
              }
              maxAbs = Math.max(maxAbs, error);
              if (baseline && actual[i] !== baseline[i]) throw new Error(`Small F16 RMSNorm changed at size=${size} element=${i} recorded=${recorded} iteration=${iteration}`);
            }
            baseline ||= actual;
          } finally {
            if (recorder) await recorder.abort();
            if (output) releaseBuffer(output.buffer);
          }
        }
      }
      reports.push({ size, rows, iterationsPerMode: iterations, modes: ['immediate', 'recorded'], maxAbs, bitStable: true });
    } finally {
      await getDevice().queue.onSubmittedWorkDone();
      releaseBuffer(inputBuffer); releaseBuffer(weightBuffer);
    }
  }
  return reports;
}
