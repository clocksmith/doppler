import assert from 'node:assert/strict';

const { runTrainingBenchSuite } = await import('../../src/training/suite.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { installNodeFileFetchShim } = await import('../../src/tooling/node-file-fetch.js');
const { initDevice } = await import('../../src/gpu/device.js');

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  installNodeFileFetchShim();
  await initDevice();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('training-telemetry-alerts.test: skipped (no WebGPU runtime)');
} else {
  {
    const result = await runTrainingBenchSuite({
      trainingSchemaVersion: 1,
      workloadType: 'training',
      trainingBenchSteps: 1,
      benchRun: {
        warmupRuns: 0,
        timedRuns: 1,
      },
      trainingConfig: {
        telemetry: {
          alerts: {
            enabled: true,
            failOnAlert: false,
            thresholds: {
              maxStepTimeMs: 0,
            },
          },
        },
      },
    });

    const entries = result.metrics.trainingMetricsReport;
    assert.ok(Array.isArray(entries) && entries.length > 0);
    assert.ok(Array.isArray(entries[0].telemetry_alerts));
    assert.ok(entries[0].telemetry_alerts.includes('max_step_time_ms_exceeded'));
  }

  {
    await assert.rejects(
      () => runTrainingBenchSuite({
        trainingSchemaVersion: 1,
        workloadType: 'training',
        trainingBenchSteps: 1,
        benchRun: {
          warmupRuns: 0,
          timedRuns: 1,
        },
        trainingConfig: {
          telemetry: {
            alerts: {
              enabled: true,
              failOnAlert: true,
              thresholds: {
                maxStepTimeMs: 0,
              },
            },
          },
        },
      }),
      /training telemetry alert\(s\): max_step_time_ms_exceeded/
    );
  }
}

console.log('training-telemetry-alerts.test: ok');
