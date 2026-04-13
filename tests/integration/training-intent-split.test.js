import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const {
  runTrainingSuite,
  runTrainingBenchSuite,
} = await import('../../src/experimental/training/suite.js');
const { probeNodeGPU } = await import('../helpers/gpu-probe.js');

function isUnavailableNodeWebGPUError(value) {
  return /createShaderModule (is not a function|failed)|Failed to load shader|fetch failed|requires a GPUBuffer|GPUBuffer is not defined/i.test(String(value || ''));
}

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`training-intent-split.test: skipped (${gpuProbe.reason})`);
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
