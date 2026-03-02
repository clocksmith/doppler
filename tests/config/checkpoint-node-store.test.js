import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import { saveCheckpoint, loadCheckpoint } from '../../src/training/checkpoint.js';

const tempDir = await mkdtemp(join(tmpdir(), 'doppler-checkpoint-node-'));
const checkpointPath = join(tempDir, 'distill.latest.checkpoint.json');

try {
  await saveCheckpoint(checkpointPath, {
    trainingState: {
      schemaVersion: 1,
      progress: { step: 2, epoch: 0, batch: 2 },
    },
  }, {
    configHash: 'cfg-a',
    datasetHash: 'data-a',
  });

  const loaded = await loadCheckpoint(checkpointPath);
  assert.ok(loaded && typeof loaded === 'object');
  assert.equal(loaded.trainingState?.progress?.step, 2);
  assert.equal(loaded.metadata?.configHash, 'cfg-a');
  assert.equal(loaded.metadata?.lineage?.sequence, 1);
  assert.equal(typeof loaded.metadata?.checkpointHash, 'string');

  await saveCheckpoint(checkpointPath, {
    trainingState: {
      schemaVersion: 1,
      progress: { step: 4, epoch: 0, batch: 4 },
    },
  }, {
    configHash: 'cfg-a',
    datasetHash: 'data-a',
  });

  const loadedNext = await loadCheckpoint(checkpointPath);
  assert.equal(loadedNext.trainingState?.progress?.step, 4);
  assert.equal(loadedNext.metadata?.lineage?.sequence, 2);

  await assert.rejects(
    () => loadCheckpoint(checkpointPath, {
      expectedMetadata: {
        configHash: 'cfg-b',
      },
    }),
    /Checkpoint mismatch on fields/
  );

  const forced = await loadCheckpoint(checkpointPath, {
    expectedMetadata: {
      configHash: 'cfg-b',
    },
    forceResume: true,
    forceResumeReason: 'intentional test mismatch',
  });
  assert.ok(Array.isArray(forced.metadata?.resumeAudits));
  assert.equal(forced.metadata.resumeAudits.length, 1);
  assert.equal(forced.metadata.resumeAudits[0].reason, 'intentional test mismatch');
} finally {
  await rm(tempDir, { recursive: true, force: true });
}

console.log('checkpoint-node-store.test: ok');
