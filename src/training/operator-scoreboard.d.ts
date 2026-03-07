export declare function appendScoreboardRow(
  scoreboardDir: string,
  row: Record<string, unknown>,
  options?: { selectionMetric?: string | null; selectionGoal?: string | null }
): Promise<{ rowsPath: string; summaryPath: string }>;
