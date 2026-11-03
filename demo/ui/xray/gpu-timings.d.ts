export interface CapturedGpuTimings {
  phases: Array<{
    phase: string;
    steps: Array<{ step: number; kernels: Array<{ label: string; ms: number }>; totalMs: number }>;
    missingSteps: number;
    totalMs: number | null;
    reportedMs: number | null;
  }>;
  kernels: Array<{ label: string; prefill: number | null; decode: number | null }>;
}
export declare function readGpuTimings(receipt: unknown): CapturedGpuTimings;
export declare function renderGpuTimings(container: HTMLElement, receipt: unknown): void;
