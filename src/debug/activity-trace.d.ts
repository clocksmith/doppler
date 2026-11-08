import type { ActivationTraceLimits } from './activation-trace.js';
import type { TensorRepresentation } from './tensor.js';
import type { CommandRecorder } from '../gpu/command-recorder.js';

export interface ActivityLimits extends ActivationTraceLimits {
  activitySummaryBlock: number; activityTokenWindow: number; activityMaxBytes: number; activityMaxFrameValues: number; activityMaxStateValues: number;
  weightBlockValues: number; weightReadValues: number; weightRowGroupSize: number; weightRawRows: number; weightMaxBytes: number;
}
export type ActivityNumber = number | 'NaN' | 'Infinity' | '-Infinity';
export interface ActivityFrame {
  index: number; atMs: number; forward: number; phase: 'prefill' | 'decode';
  role: string; layer: number; tokenStart: number; tokenIds: number[];
  shape: number[]; fullShape: number[]; offset: number; dtype: string;
  values: ActivityNumber[] | null; status: 'captured' | 'failed' | 'dropped';
  reason: string | null; representation: 'raw' | 'reconstructed' | 'aggregated';
  summary?: { blockSize: number; observedValues: number; values: ActivityNumber[] };
  metadata: Record<string, unknown>;
}
export interface ActivityWeight {
  id: string; layer: number | null; role: string; shape: number[]; dtype: string;
  layout: string; resident: boolean; byteLength: number; status: string; reason: string | null;
  blockSize: number; rms: number[]; raw: number[]; rawOffset: number;
  error?: number[]; referenceId?: string; referenceDtype?: string;
  columnEnergy: number[]; rowGroupSize: number;
}
export interface ActivityReceipt {
  schema: 'doppler.model-activity/v1'; frames: ActivityFrame[]; weights: ActivityWeight[];
  recordedBytes: number; limits: ActivityLimits;
  unavailable: Record<string, string>; model: Record<string, unknown>;
}
export interface ActivityCaptureOptions extends TensorRepresentation {
  layer: number; shape: number[]; dtype: string; recorder?: CommandRecorder | null;
  offset?: number; tokenStart?: number; tokenIds?: readonly number[];
  metadata?: Record<string, unknown>; state?: boolean;
  representation?: ActivityFrame['representation'];
}
export interface ActivityTrace {
  begin(phase: 'prefill' | 'decode', tokenStart: number, tokenIds: readonly number[]): void;
  setReader(reader: (buffer: GPUBuffer, offset: number, size: number) => Promise<ArrayBuffer>): void;
  configure(model: Record<string, unknown>): void;
  capture(role: string, buffer: GPUBuffer | Float32Array | Float64Array, options: ActivityCaptureOptions): Promise<void>;
  probe(stage: string, buffer: GPUBuffer | Float32Array | Float64Array, options: {
    layerIdx?: number; numTokens: number; hiddenSize: number; dtype?: string;
    recorder?: CommandRecorder | null; precision?: 'two-f32'; components?: 2; elementStrideBytes?: 8;
  }): Promise<void>;
  weightAccess(layer: number, stage: string): void;
  unavailable(role: string, reason: string): void;
  setWeights(weights: ActivityWeight[]): void;
  snapshot(includeFrames?: boolean): ActivityReceipt;
  logits(logits: Float32Array): Promise<void>;
}
export function createActivityTrace(limits: ActivityLimits, started: number): ActivityTrace;
