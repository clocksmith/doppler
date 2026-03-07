import { join, resolve } from 'node:path';

import { watchFinalizedCheckpoints } from '../checkpoint-watch.js';
import { appendDistillationScoreboardRow } from './scoreboard.js';
import { evaluateDistillationCheckpoint, readDistillCheckpointMarker } from './eval.js';

export async function watchDistillationCheckpoints(options) {
  const layout = options.layout;
  const loadedWorkload = options.loadedWorkload;
  const checkpointsDir = resolve(options.checkpointsDir || layout.checkpoints);
  const manifestPath = resolve(options.manifestPath || join(layout.scoreboard, 'checkpoint-watch-manifest.json'));
  return watchFinalizedCheckpoints({
    checkpointsDir,
    manifestPath,
    pollIntervalMs: options.pollIntervalMs || 2000,
    stopWhenIdle: options.stopWhenIdle === true,
    signal: options.signal ?? null,
    onCheckpoint: async (markerPath) => {
      const { marker } = await readDistillCheckpointMarker(markerPath);
      const reports = await evaluateDistillationCheckpoint({
        loadedWorkload,
        checkpointPath: marker.checkpointPath,
        checkpointId: marker.checkpointId,
        checkpointStep: marker.checkpointStep,
        stageId: marker.stage,
        layout,
        stageAArtifact: marker.stageArtifact || null,
        stageAArtifactHash: marker.stageArtifactHash || null,
      });
      for (const report of reports) {
        await appendDistillationScoreboardRow(layout, String(marker.stage || 'stage'), {
          artifactType: 'training_scoreboard',
          schemaVersion: 1,
          generatedAt: new Date().toISOString(),
          stage: marker.stage,
          checkpointId: report.checkpointId,
          checkpointStep: report.checkpointStep,
          evalDatasetId: report.evalDatasetId,
          selectionMetric: report.primaryMetric,
          selectionGoal: 'max',
          primaryMetric: report.primaryMetric,
          primaryScore: report.primaryScore,
          bleu: report.bleu,
          chrf: report.chrf,
          reportPath: report.reportPath || null,
          metrics: {
            bleu: report.bleu,
            chrf: report.chrf,
            primaryScore: report.primaryScore,
          },
        }, {
          selectionMetric: report.primaryMetric,
          selectionGoal: 'max',
        });
      }
    },
  });
}
