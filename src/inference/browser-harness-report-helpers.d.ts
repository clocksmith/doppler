export declare function collectTrainingArtifactsFromSuiteResult(
  suiteResult: Record<string, unknown>
): {
  ulArtifacts: Array<Record<string, unknown>>;
  distillArtifacts: Array<Record<string, unknown>>;
  checkpointResumeTimeline: Array<Record<string, unknown>>;
};
