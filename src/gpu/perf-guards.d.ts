export interface PerfConfig {
  allowGPUReadback: boolean;
  trackSubmitCount: boolean;
  trackAllocations: boolean;
  logExpensiveOps: boolean;
  strictMode: boolean;
}

export declare function configurePerfGuards(newConfig: Partial<PerfConfig>): void;
export declare function getPerfConfig(): Readonly<PerfConfig>;
export declare function resetPerfCounters(): void;
export declare function getPerfCounters(): Readonly<{
  submits: number;
  allocations: number;
  readbacks: number;
  startTime: number;
}>;
export declare function trackSubmit(): void;
export declare function trackAllocation(size: number, label?: string): void;
export declare function allowReadback(reason?: string, count?: number): boolean;
export declare function getPerfSummary(): string;
export declare function logPerfSummary(): void;
export declare function enableProductionMode(): void;
export declare function enableDebugMode(): void;
export declare function enableBenchmarkMode(): void;
