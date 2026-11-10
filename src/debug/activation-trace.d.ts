export interface ActivationTraceLimits {
  activitySelection?: import('./activity-trace.js').ActivitySelection; activityRoleMaxBytes?: Record<string, number>;
  activityMaxBytes?: number; activityMaxFrameValues?: number; activityMaxStateValues?: number;
  activityTokenWindow?: number; activitySummaryBlock?: number;
  weightBlockValues?: number; weightReadValues?: number; weightRowGroupSize?: number; weightRawRows?: number; weightMaxBytes?: number;
  maxBytes: number; maxFrameBytes: number; maxFrames: number; maxGridValues: number; maxInFlight: number;
}
export interface ActivationGeometry { layers: number; width: number }
export interface ActivationFrame {
  readonly index: number; readonly forward: number; readonly atMs: number;
  readonly phase: 'prefill' | 'decode'; readonly role: 'layer-output'; readonly layer: number;
  readonly tokenStart: number; readonly tokenIds: readonly number[]; readonly shape: readonly number[];
  readonly dtype: string; readonly status: 'pending' | 'captured' | 'dropped' | 'failed';
  readonly precision: 'two-f32' | null;
  readonly values: readonly (number | string)[] | null; readonly reason: string | null;
}
export interface ActivationTraceReceipt {
  activity?: import('./activity-trace.js').ActivityReceipt;
  schema: 'doppler.activation-trace/v1';
  identity: { modelId: string; logicalModelId?: string; manifestHash: string | null; runtimeVersion: string; adapters?: unknown[] };
  geometry: ActivationGeometry; capturePolicy: ActivationTraceLimits;
  status: 'complete' | 'stopped' | 'error' | 'recording'; stoppedReason: string | null;
  recordedBytes: number; frames: ActivationFrame[];
}
export interface ActivationTrace {
  activity?: import('./activity-trace.js').ActivityTrace;
  capture(buffer: GPUBuffer | Float32Array | Float64Array, context: {
    layer: number; phase: 'prefill' | 'decode'; tokenStart: number; tokenIds: readonly number[];
    dtype: string; recorder?: { submitPartial(): void } | null; precision?: 'two-f32'; components?: 2; elementStrideBytes?: 8;
  }): void;
  drain(): Promise<void>;
  snapshot(status: ActivationTraceReceipt['status'], includeFrames?: boolean): ActivationTraceReceipt;
}
export declare function createActivationTrace(limits: ActivationTraceLimits, geometry: ActivationGeometry,
  identity: ActivationTraceReceipt['identity'], onFrame?: (frame: ActivationFrame) => void): ActivationTrace;
