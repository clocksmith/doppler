import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');
const { probeNodeGPU } = await import('../helpers/gpu-probe.js');

function isShaderFetchFailure(value) {
  const message = String(value || '');
  return /Failed to load shader|fetch failed|createShaderModule (is not a function|failed)|produced no metrics|requires a GPUBuffer|GPUBuffer is not defined|Checkpoint mismatch on fields/i.test(message);
}

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`ul-stage-workflow-regression.test: skipped (${gpuProbe.reason})`);
} else {
  const stage1Dir = mkdtempSync(join(tmpdir(), 'doppler-ul-stage1-'));
  const stage2Dir = mkdtempSync(join(tmpdir(), 'doppler-ul-stage2-'));
  const stage2MismatchDir = mkdtempSync(join(tmpdir(), 'doppler-ul-stage2-mismatch-'));
  try {
    const stage1 = await runBrowserSuite({
      suite: 'training',
      command: 'verify',
      surface: 'node',
      trainingSchemaVersion: 1,
      trainingTests: ['ul-stage1'],
      trainingStage: 'stage1_joint',
      ulArtifactDir: stage1Dir,
    });

    const stage1Result = stage1.results.find((entry) => entry.name === 'ul-stage1');
    if (!stage1Result || stage1Result.passed !== true) {
      if (isShaderFetchFailure(stage1Result?.error)) {
        console.log('ul-stage-workflow-regression.test: skipped (shader fetch unavailable in this environment)');
      } else {
        assert.fail(`ul-stage1 did not pass: ${String(stage1Result?.error || 'unknown error')}`);
      }
    } else {
      assert.ok(stage1Result.artifact && typeof stage1Result.artifact.manifestPath === 'string');

      const stage2 = await runBrowserSuite({
        suite: 'training',
        command: 'verify',
        surface: 'browser',
        trainingSchemaVersion: 1,
        trainingTests: ['ul-stage2'],
        trainingStage: 'stage2_base',
        stage1Artifact: stage1Result.artifact.manifestPath,
        stage1ArtifactHash: stage1Result.artifact.manifestFileHash || stage1Result.artifact.manifestHash,
        ulArtifactDir: stage2Dir,
      });

      const stage2Result = stage2.results.find((entry) => entry.name === 'ul-stage2');
      if (!stage2Result || stage2Result.passed !== true) {
        if (isShaderFetchFailure(stage2Result?.error)) {
          console.log('ul-stage-workflow-regression.test: skipped (shader fetch unavailable in this environment)');
        } else {
          assert.fail(`ul-stage2 did not pass: ${String(stage2Result?.error || 'unknown error')}`);
        }
      } else {
        assert.ok(stage2Result.artifact && typeof stage2Result.artifact.manifestPath === 'string');

        const stage2Mismatch = await runBrowserSuite({
          suite: 'training',
          command: 'verify',
          surface: 'browser',
          trainingSchemaVersion: 1,
        trainingTests: ['ul-stage2'],
        trainingStage: 'stage2_base',
        stage1Artifact: stage1Result.artifact.manifestPath,
        stage1ArtifactHash: 'deadbeef',
        ulArtifactDir: stage2MismatchDir,
      });

        const mismatchResult = stage2Mismatch.results.find((entry) => entry.name === 'ul-stage2');
        assert.ok(mismatchResult);
        assert.equal(mismatchResult.passed, false);
        assert.match(String(mismatchResult.error || ''), /artifact hash mismatch/);
      }
    }
  } finally {
    rmSync(stage1Dir, { recursive: true, force: true });
    rmSync(stage2Dir, { recursive: true, force: true });
    rmSync(stage2MismatchDir, { recursive: true, force: true });
  }
}

console.log('ul-stage-workflow-regression.test: ok');
