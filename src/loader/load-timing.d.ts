export declare function nowMs(): number;
export declare function roundLoadTimingMs(value: number): number | null;
export declare function createLoadTiming(modelId: string, hasCustomLoader: boolean): any;
export declare function finishLoadPhase(loadTiming: any, phase: string, startMs: number): void;
export declare function finishLoadTiming(
  loadTiming: any,
  status: string,
  startMs: number,
  error?: unknown,
  failedPhase?: string | null
): void;
