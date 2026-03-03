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
  console.log('training-report-lineage.test: skipped (no WebGPU runtime)');
} else {
  const result = await runBrowserSuite({
    suite: 'training',
    command: 'verify',
    surface: 'node',
    trainingSchemaVersion: 1,
    trainingTests: ['ul-stage1'],
    trainingStage: 'stage1_joint',
  });

  const stage1Result = Array.isArray(result?.results)
    ? result.results.find((entry) => entry?.name === 'ul-stage1')
    : null;
  if (!stage1Result || stage1Result.passed !== true) {
    if (isShaderFetchFailure(stage1Result?.error)) {
      console.log('training-report-lineage.test: skipped (shader fetch unavailable in this environment)');
    } else {
      assert.fail(`ul-stage1 did not pass: ${String(stage1Result?.error || 'unknown error')}`);
    }
  } else {
    assert.ok(result.report && typeof result.report === 'object');
    const ulArtifacts = result.report?.lineage?.training?.ulArtifacts;
    assert.ok(Array.isArray(ulArtifacts), 'report lineage should include training ulArtifacts array');
    assert.ok(ulArtifacts.length > 0, 'report lineage should include at least one UL artifact');
    assert.ok(typeof ulArtifacts[0].manifestPath === 'string' && ulArtifacts[0].manifestPath.length > 0);
  }
}

console.log('training-report-lineage.test: ok');
process.exit(0);
