import assert from 'node:assert/strict';
import { readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';

const { runTrainingSuite } = await import('../../src/training/suite.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { initDevice } = await import('../../src/gpu/device.js');

function listGeneratedTempRoots(names) {
  return names.filter((name) => name.startsWith('doppler-ul-') || name.startsWith('doppler-distill-'));
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
  console.log('suite-artifact-dir-cleanup.test: skipped (no WebGPU runtime)');
} else {
  const before = listGeneratedTempRoots(await readdir(tmpdir()));
  let runDir = null;
  try {
    const summary = await runTrainingSuite({
      trainingSchemaVersion: 1,
      trainingTests: ['ul-stage1'],
      trainingStage: 'stage1_joint',
    });

    const stage1 = summary.results.find((entry) => entry.name === 'ul-stage1');
    assert.ok(stage1 && stage1.passed === true, stage1?.error || 'ul-stage1 should pass');
    assert.ok(stage1.artifact && typeof stage1.artifact.runDir === 'string');
    runDir = resolve(process.cwd(), stage1.artifact.runDir);

    const after = listGeneratedTempRoots(await readdir(tmpdir()));
    const newTempRoots = after.filter((name) => !before.includes(name));
    assert.deepEqual(newTempRoots, []);
    assert.match(stage1.artifact.runDir, /^reports\/training\/ul\//);
  } finally {
    if (runDir) {
      await rm(runDir, { recursive: true, force: true });
    }
  }
}

console.log('suite-artifact-dir-cleanup.test: ok');
