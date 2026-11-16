export declare function destroyMoERouter(router: { destroy?(): void } | null | undefined): void;
export declare function roundPipelineTimingMs(value: number): number | null;
export declare function createPipelineLoadTiming(modelId?: string | null): any;
export declare function finishPipelineLoadTimingPhase(
  loadTiming: any,
  phase: string,
  startMs: number
): void;
export declare function timedPipelineLoadPhase<T>(
  loadTiming: any,
  phase: string,
  context: Record<string, unknown>,
  run: () => Promise<T>
): Promise<T>;
