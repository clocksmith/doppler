import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const {
  runTrainingSuite,
  runTrainingBenchSuite,
} = await import('../../src/training/suite.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { initDevice } = await import('../../src/gpu/device.js');

function isUnavailableNodeWebGPUError(value) {
  return /createShaderModule is not a function|Failed to load shader|fetch failed|requires a GPUBuffer/i.test(String(value || ''));
}

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  await initDevice();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('training-intent-split.test: skipped (no WebGPU runtime)');
} else {
  const tempDir = mkdtempSync(join(tmpdir(), 'doppler-training-intent-split-'));
  try {
    const verifyResult = await runTrainingSuite({
      trainingSchemaVersion: 1,
      trainingTests: ['runner-smoke'],
      ulArtifactDir: tempDir,
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
      ulArtifactDir: tempDir,
    });

    assert.equal(benchResult.suite, 'bench');
    assert.equal(benchResult.metrics.workloadType, 'training');
    assert.ok(Array.isArray(benchResult.metrics.trainingMetricsReport));
    assert.ok(benchResult.metrics.trainingMetricsReport.length > 0);
    assert.ok(benchResult.metrics.progress && typeof benchResult.metrics.progress === 'object');
    assert.ok(Number.isFinite(benchResult.metrics.progress.percentComplete));
    assert.ok(Array.isArray(benchResult.metrics.checkpointResumeTimeline));
    assert.ok(benchResult.metrics.checkpointResumeTimeline.length > 0);
    assert.ok(benchResult.metrics.latency && typeof benchResult.metrics.latency === 'object');
    assert.ok(benchResult.metrics.throughput && typeof benchResult.metrics.throughput === 'object');
  } catch (error) {
    if (isUnavailableNodeWebGPUError(error)) {
      console.log('training-intent-split.test: skipped (functional WebGPU runtime unavailable)');
    } else {
      throw error;
    }
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

console.log('training-intent-split.test: ok');
