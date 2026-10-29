export function collectTrainingArtifactsFromSuiteResult(suiteResult) {
  const ulArtifacts = [];
  const distillArtifacts = [];
  const checkpointResumeTimeline = Array.isArray(suiteResult?.metrics?.checkpointResumeTimeline)
    ? suiteResult.metrics.checkpointResumeTimeline
      .filter((entry) => entry && typeof entry === 'object')
    : [];
  const addArtifact = (artifact, source = null) => {
    if (!artifact || typeof artifact !== 'object' || typeof artifact.manifestPath !== 'string') {
      return;
    }
    const stage = String(artifact.stage || '').trim();
    const kind = String(artifact.kind || '').trim();
    if (kind === 'distill' || stage === 'stage_a' || stage === 'stage_b') {
      distillArtifacts.push(artifact);
      return;
    }
    if (kind === 'ul' || stage === 'stage1_joint' || stage === 'stage2_base' || source === 'ul') {
      ulArtifacts.push(artifact);
      return;
    }
    ulArtifacts.push(artifact);
  };

  const metricUlArtifacts = Array.isArray(suiteResult?.metrics?.ulArtifacts)
    ? suiteResult.metrics.ulArtifacts
    : [];
  for (const artifact of metricUlArtifacts) {
    addArtifact(artifact, 'ul');
  }
  const metricDistillArtifacts = Array.isArray(suiteResult?.metrics?.distillArtifacts)
    ? suiteResult.metrics.distillArtifacts
    : [];
  for (const artifact of metricDistillArtifacts) {
    addArtifact(artifact, 'distill');
  }
  const resultEntries = Array.isArray(suiteResult?.results) ? suiteResult.results : [];
  for (const entry of resultEntries) {
    addArtifact(entry?.artifact, null);
  }
  return { ulArtifacts, distillArtifacts, checkpointResumeTimeline };
}
