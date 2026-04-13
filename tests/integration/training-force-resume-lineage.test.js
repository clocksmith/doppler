import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const { runTrainingBenchSuite } = await import('../../src/experimental/training/suite.js');
const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');
const { probeNodeGPU } = await import('../helpers/gpu-probe.js');

function isUnavailableNodeWebGPUError(value) {
  return /createShaderModule (is not a function|failed)|Failed to load shader|fetch failed|requires a GPUBuffer|GPUBuffer is not defined/i.test(String(value || ''));
}

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`training-force-resume-lineage.test: skipped (${gpuProbe.reason})`);
} else {
  const tempDir = mkdtempSync(join(tmpdir(), 'doppler-force-resume-lineage-'));
  try {
    const checkpointPath = join(tempDir, 'training.latest.checkpoint.json');

    try {
      await runTrainingBenchSuite({
        trainingSchemaVersion: 1,
        workloadType: 'training',
        trainingBenchSteps: 1,
        benchRun: {
          warmupRuns: 0,
          timedRuns: 1,
        },
        ulArtifactDir: tempDir,
        trainingConfig: {
          telemetry: {
            windowSize: 2,
          },
        },
      });

      const forced = await runBrowserSuite({
        suite: 'bench',
        command: 'bench',
        surface: 'node',
        trainingSchemaVersion: 1,
        workloadType: 'training',
        trainingBenchSteps: 1,
        benchRun: {
          warmupRuns: 0,
          timedRuns: 1,
        },
        ulArtifactDir: tempDir,
        resumeFrom: checkpointPath,
        forceResume: true,
        forceResumeReason: 'force-resume-lineage-test',
        trainingConfig: {
          telemetry: {
            windowSize: 3,
          },
        },
      });

      const timeline = forced?.metrics?.checkpointResumeTimeline;
      assert.ok(Array.isArray(timeline), 'forced run should include checkpoint resume timeline');
      const overrideEvent = timeline.find((entry) => entry?.type === 'resume_override_applied');
      assert.ok(overrideEvent, 'forced run should include resume_override_applied event');
      assert.ok(Array.isArray(overrideEvent.resumeAudits), 'override event should include resumeAudits array');
      assert.ok(overrideEvent.resumeAudits.length > 0, 'override event should include at least one resume audit');
      assert.equal(overrideEvent.resumeAudits[0].reason, 'force-resume-lineage-test');

      const lineageTimeline = forced?.report?.lineage?.training?.checkpointResumeTimeline;
      assert.ok(Array.isArray(lineageTimeline), 'report lineage should include checkpoint resume timeline');
      assert.ok(
        lineageTimeline.some((entry) => entry?.type === 'resume_override_applied'),
        'report lineage should include resume_override_applied event'
      );
    } catch (error) {
      if (isUnavailableNodeWebGPUError(error)) {
        console.log('training-force-resume-lineage.test: skipped (functional WebGPU runtime unavailable)');
      } else {
        throw error;
      }
    }
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

console.log('training-force-resume-lineage.test: ok');
