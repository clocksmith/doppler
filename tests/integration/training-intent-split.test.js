import assert from 'node:assert/strict';

const {
  runTrainingSuite,
  runTrainingBenchSuite,
} = await import('../../src/training/suite.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('training-intent-split.test: skipped (no WebGPU runtime)');
} else {

  const verifyResult = await runTrainingSuite({
    trainingSchemaVersion: 1,
    trainingTests: ['runner-smoke'],
  });

  assert.equal(verifyResult.suite, 'training');
  assert.ok(Number.isInteger(verifyResult.metrics.testsRun));
  assert.ok(Array.isArray(verifyResult.metrics.selectedTests));
  assert.ok(!Object.prototype.hasOwnProperty.call(verifyResult.metrics, 'workloadType'));

  const benchResult = await runTrainingBenchSuite({
    trainingSchemaVersion: 1,
    workloadType: 'training',
    trainingBenchSteps: 2,
    benchRun: {
      warmupRuns: 0,
      timedRuns: 1,
    },
  });

  assert.equal(benchResult.suite, 'bench');
  assert.equal(benchResult.metrics.workloadType, 'training');
  assert.ok(Array.isArray(benchResult.metrics.trainingMetricsReport));
  assert.ok(benchResult.metrics.trainingMetricsReport.length > 0);
  assert.ok(benchResult.metrics.latency && typeof benchResult.metrics.latency === 'object');
  assert.ok(benchResult.metrics.throughput && typeof benchResult.metrics.throughput === 'object');
}

console.log('training-intent-split.test: ok');
