import { join } from 'node:path';

import { appendScoreboardRow } from '../operator-scoreboard.js';

export async function appendDistillationScoreboardRow(layout, stageId, row, options = {}) {
  const scoreboardDir = join(layout.scoreboard, String(stageId || 'stage'));
  return appendScoreboardRow(scoreboardDir, row, options);
}
