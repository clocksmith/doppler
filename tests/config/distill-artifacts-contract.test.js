import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { sha256Hex } from '../../src/formats/sha256.js';
import { createTrainingConfig } from '../../src/config/training-defaults.js';
import {
  createDistillArtifactSession,
  resolveStageAArtifactContext,
} from '../../src/experimental/training/artifacts.js';

const TEMP_ROOT = await mkdtemp(join(tmpdir(), 'doppler-distill-artifacts-'));

function buildDistillConfig(overrides = {}) {
  return createTrainingConfig({
    training: {
      enabled: true,
      distill: {
        enabled: true,
        stage: 'stage-a',
        teacherModelId: 'translategemma-4b-it-q4k-ehf16-af32',
        studentModelId: 'translategemma-270m-it-f16-af32',
        datasetId: 'en-es',
        languagePair: 'en-es',
        ...overrides,
      },
    },
  });
}

try {
  const stageAConfig = buildDistillConfig({
    stage: 'stage-a',
    artifactDir: TEMP_ROOT,
  });
  const stageASession = await createDistillArtifactSession({
    config: stageAConfig,
    stage: 'stage-a',
    runOptions: {
      distillArtifactDir: TEMP_ROOT,
      modelId: 'toy-model',
      timestamp: '2026-03-01T00:00:00.000Z',
      batchSize: 1,
      epochs: 1,
      maxSteps: 1,
    },
  });
  assert(stageASession, 'stage-a session should be created');

  const stageAEntry = {
    schemaVersion: 1,
    step: 1,
    epoch: 0,
    batch: 1,
    objective: 'kd',
    total_loss: 0.1,
    step_time_ms: 1,
    forward_ms: 0.5,
    backward_ms: 0.4,
    distill_stage: 'stage-a',
    loss_kd: 0.07,
  };
  await stageASession.appendStep(stageAEntry);
  const stageAArtifact = await stageASession.finalize([stageAEntry]);

  const stageAManifestPath = resolve(process.cwd(), stageAArtifact.manifestPath);
  const stageAManifestRaw = await readFile(stageAManifestPath, 'utf8');
  const stageAFileHash = sha256Hex(stageAManifestRaw);

  const stageBConfig = buildDistillConfig({
    stage: 'stage-b',
    artifactDir: TEMP_ROOT,
    stageAArtifact: stageAManifestPath,
    stageAArtifactHash: stageAFileHash,
  });
  const stageAContext = await resolveStageAArtifactContext(stageBConfig);
  assert(stageAContext, 'stage-a context should resolve');
  assert.equal(stageAContext.metrics.count, 1);

  const stageBSession = await createDistillArtifactSession({
    config: stageBConfig,
    stage: 'stage-b',
    runOptions: {
      distillArtifactDir: TEMP_ROOT,
      timestamp: '2026-03-01T00:00:01.000Z',
    },
  });
  assert(stageBSession, 'stage-b session should be created');

  await assert.rejects(
    () => createDistillArtifactSession({
      config: buildDistillConfig({
        stage: 'stage-b',
        artifactDir: TEMP_ROOT,
        stageAArtifact: stageAManifestPath,
        stageAArtifactHash: 'deadbeef',
      }),
      stage: 'stage-b',
      runOptions: {
        distillArtifactDir: TEMP_ROOT,
        timestamp: '2026-03-01T00:00:02.000Z',
      },
    }),
    /artifact hash mismatch/
  );
} finally {
  await rm(TEMP_ROOT, { recursive: true, force: true });
}

console.log('distill-artifacts-contract.test: ok');
