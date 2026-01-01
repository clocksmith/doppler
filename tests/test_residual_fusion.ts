/**
 * Test script to verify Residual+RMSNorm fusion speedup
 *
 * Compares:
 * 1. Separate: ResidualAdd + RMSNorm (2 kernel dispatches)
 * 2. Fused: RMSNorm with residual parameter (1 kernel dispatch)
 */

import { initDevice } from '../src/gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../src/gpu/buffer-pool.js';
import { runRMSNorm } from '../src/gpu/kernels/rmsnorm.js';
import { runResidualAdd } from '../src/gpu/kernels/residual.js';

async function benchmarkResidualRMSNorm() {
  console.log('Initializing WebGPU...');
  const device = await initDevice();

  // Gemma 1B dimensions
  const hiddenSize = 1152;
  const batchSize = 1; // decode
  const eps = 1e-6;
  const runs = 100;

  // Create test buffers
  console.log(`Creating test buffers (hiddenSize=${hiddenSize}, batchSize=${batchSize})...`);
  const size = hiddenSize * batchSize * 4;

  const inputBuf = acquireBuffer(size, undefined, 'test_input');
  const residualBuf = acquireBuffer(size, undefined, 'test_residual');
  const weightBuf = acquireBuffer(hiddenSize * 4, undefined, 'test_weight');

  // Fill with test data
  const inputData = new Float32Array(hiddenSize * batchSize);
  const residualData = new Float32Array(hiddenSize * batchSize);
  const weightData = new Float32Array(hiddenSize);

  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.random() * 0.1;
    residualData[i] = Math.random() * 0.1;
  }
  for (let i = 0; i < weightData.length; i++) {
    weightData[i] = 1.0 + Math.random() * 0.1;
  }

  device.queue.writeBuffer(inputBuf, 0, inputData);
  device.queue.writeBuffer(residualBuf, 0, residualData);
  device.queue.writeBuffer(weightBuf, 0, weightData);
  await device.queue.onSubmittedWorkDone();

  console.log('\\nWarming up...');
  // Warmup
  for (let i = 0; i < 10; i++) {
    const temp = await runResidualAdd(inputBuf, residualBuf, hiddenSize * batchSize);
    const output = await runRMSNorm(temp, weightBuf, eps, { batchSize, hiddenSize });
    releaseBuffer(temp);
    releaseBuffer(output);
  }
  await device.queue.onSubmittedWorkDone();

  // Benchmark 1: Separate ResidualAdd + RMSNorm
  console.log(`\\nBenchmarking SEPARATE (ResidualAdd + RMSNorm, ${runs} runs)...`);
  const startSeparate = performance.now();

  for (let i = 0; i < runs; i++) {
    const temp = await runResidualAdd(inputBuf, residualBuf, hiddenSize * batchSize);
    const output = await runRMSNorm(temp, weightBuf, eps, { batchSize, hiddenSize });
    releaseBuffer(temp);
    releaseBuffer(output);
  }
  await device.queue.onSubmittedWorkDone();

  const endSeparate = performance.now();
  const timeSeparate = (endSeparate - startSeparate) / runs;

  console.log(`Separate: ${timeSeparate.toFixed(3)} ms/iter`);

  // Benchmark 2: Fused RMSNorm with residual
  console.log(`\\nBenchmarking FUSED (RMSNorm with residual, ${runs} runs)...`);
  const startFused = performance.now();

  for (let i = 0; i < runs; i++) {
    const output = await runRMSNorm(inputBuf, weightBuf, eps, {
      batchSize,
      hiddenSize,
      residual: residualBuf
    });
    releaseBuffer(output);
  }
  await device.queue.onSubmittedWorkDone();

  const endFused = performance.now();
  const timeFused = (endFused - startFused) / runs;

  console.log(`Fused: ${timeFused.toFixed(3)} ms/iter`);

  // Results
  const speedup = timeSeparate / timeFused;
  console.log(`\\n${'='.repeat(60)}`);
  console.log(`RESULTS (hiddenSize=${hiddenSize}, batchSize=${batchSize})`);
  console.log(`${'='.repeat(60)}`);
  console.log(`Separate: ${timeSeparate.toFixed(3)} ms`);
  console.log(`Fused:    ${timeFused.toFixed(3)} ms`);
  console.log(`Speedup:  ${speedup.toFixed(2)}x`);
  console.log(`Expected: 1.2-1.3x (per PHASE_1_PERFORMANCE.md)`);

  if (speedup >= 1.15) {
    console.log(`✅ PASS: Speedup ${speedup.toFixed(2)}x >= 1.15x`);
  } else {
    console.log(`❌ FAIL: Speedup ${speedup.toFixed(2)}x < 1.15x (expected 1.2-1.3x)`);
  }

  // Cleanup
  releaseBuffer(inputBuf);
  releaseBuffer(residualBuf);
  releaseBuffer(weightBuf);

  console.log(`\\nTest complete.`);
}

benchmarkResidualRMSNorm().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
