import assert from 'node:assert/strict';

const { runTrainingBenchSuite } = await import('../../src/experimental/training/suite.js');
const { probeNodeGPU } = await import('../helpers/gpu-probe.js');

const gpuProbe = await probeNodeGPU({ installFileFetchShim: true });
if (!gpuProbe.ready) {
  console.log(`training-telemetry-alerts.test: skipped (${gpuProbe.reason})`);
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
