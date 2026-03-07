import { resolve } from 'node:path';

import { runDistillationStage } from './stage-a.js';

export async function runDistillationStageB(options) {
  const priorStage = options.priorStageResult || null;
  const stageAArtifact = options.stageAArtifact
    || priorStage?.legacyArtifact?.manifestPath
    || null;
  if (!stageAArtifact) {
    throw new Error('Distillation stage-b requires a Stage A artifact path.');
  }
  return runDistillationStage({
    ...options,
    stageAArtifact: resolve(String(stageAArtifact)),
    stageAArtifactHash: options.stageAArtifactHash
      || priorStage?.legacyArtifact?.manifestHash
      || null,
  });
}
