export declare function aggregateTopDecodeTimers(
  decodeProfileSteps: Array<Record<string, unknown>> | null | undefined,
  limit?: number
): Array<{ label: string; totalMs: number }>;

export declare function buildKernelPathBuilderRuntimeOverlay(
  model: Record<string, unknown> | null | undefined,
  report: Record<string, unknown> | null | undefined
): Record<string, unknown> | null;
