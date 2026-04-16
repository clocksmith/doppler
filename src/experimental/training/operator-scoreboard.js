import { join } from 'node:path';

import { writeJsonArtifact, writeNdjsonRow } from './operator-artifacts.js';

function resolveComparableMetric(row, metric) {
  if (!row || typeof row !== 'object') return null;
  const direct = row[metric];
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return direct;
  }
  const metrics = row.metrics && typeof row.metrics === 'object' ? row.metrics : null;
  const nested = metrics?.[metric];
  if (typeof nested === 'number' && Number.isFinite(nested)) {
    return nested;
  }
  return null;
}

export async function appendScoreboardRow(scoreboardDir, row, options = {}) {
  const rowsPath = join(scoreboardDir, 'scoreboard.ndjson');
  await writeNdjsonRow(rowsPath, row);
  const metric = String(options.selectionMetric || row.selectionMetric || row.primaryMetric || '').trim();
  const goal = String(options.selectionGoal || row.selectionGoal || 'max').trim();
  const comparable = resolveComparableMetric(row, metric);
  const summary = {
    artifactType: 'training_scoreboard',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    selectionMetric: metric || null,
    selectionGoal: goal,
    latest: row,
    best: comparable === null
      ? row
      : {
        ...row,
        selectionMetricValue: comparable,
      },
  };
  const summaryResult = await writeJsonArtifact(join(scoreboardDir, 'latest.json'), summary);
  return {
    rowsPath,
    summaryPath: summaryResult.path,
  };
}
