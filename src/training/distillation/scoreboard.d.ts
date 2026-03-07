export declare function appendDistillationScoreboardRow(
  layout: Record<string, string>,
  stageId: string,
  row: Record<string, unknown>,
  options?: { selectionMetric?: string | null; selectionGoal?: string | null }
): Promise<{ rowsPath: string; summaryPath: string }>;
