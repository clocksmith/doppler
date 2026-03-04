import assert from 'node:assert/strict';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');

function isShaderFetchFailure(value) {
  const message = String(value || '');
  return /Failed to load shader|fetch failed/i.test(message);
}

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('ul-stage-workflow-regression.test: skipped (no WebGPU runtime)');
} else {
  const stage1 = await runBrowserSuite({
    suite: 'training',
    command: 'verify',
    surface: 'node',
    trainingSchemaVersion: 1,
    trainingTests: ['ul-stage1'],
    trainingStage: 'stage1_joint',
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
    });

    const stage2Result = stage2.results.find((entry) => entry.name === 'ul-stage2');
    assert.ok(stage2Result && stage2Result.passed === true);
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
    });

    const mismatchResult = stage2Mismatch.results.find((entry) => entry.name === 'ul-stage2');
    assert.ok(mismatchResult);
    assert.equal(mismatchResult.passed, false);
    assert.match(String(mismatchResult.error || ''), /artifact hash mismatch/);
  }
}

console.log('ul-stage-workflow-regression.test: ok');
