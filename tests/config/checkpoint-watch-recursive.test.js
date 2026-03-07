import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { watchFinalizedCheckpoints } from '../../src/training/checkpoint-watch.js';

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-checkpoint-watch-'));
try {
  const checkpointsDir = path.join(tempDir, 'checkpoints');
  const nestedCheckpointDir = path.join(checkpointsDir, 'stage_a', 'checkpoint-000001');
  mkdirSync(nestedCheckpointDir, { recursive: true });
  writeFileSync(
    path.join(nestedCheckpointDir, 'checkpoint.complete.json'),
    `${JSON.stringify({ checkpointId: 'checkpoint-000001', stage: 'stage_a' })}\n`,
    'utf8'
  );

  const seen = [];
  const result = await watchFinalizedCheckpoints({
    checkpointsDir,
    manifestPath: path.join(tempDir, 'watch-manifest.json'),
    stopWhenIdle: true,
    pollIntervalMs: 100,
    async onCheckpoint(markerPath) {
      seen.push(markerPath);
    },
  });

  assert.equal(result.ok, true);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].endsWith(path.join('stage_a', 'checkpoint-000001', 'checkpoint.complete.json')), true);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('checkpoint-watch-recursive.test: ok');
