import type { ActivationRecording } from './data.js';
import type { ActivityFrame, ActivityCoverage } from '../../../src/debug/activity-trace.js';
export type { ActivityFrame, ActivityCoverage };
export interface ActivitySelection {
  mode: string; atMs: number; position: number; layer: number; weightId: string;
  detail: string; page: number; baseline?: ActivationRecording | null;
}
export interface ActivityView {
  range: 'signed' | 'positive' | 'unit' | 'residency';
  segments: { start:number; end:number; id:string; count:number; blockSize:number }[];
  values: (number | string | undefined)[]; extent: number; offset: number;
  underlyingValues: number; quantity: string; representation: string; description: string; valuesPerScalar: number;
}
export interface PlaybackPoint { atMs: number; position: number; phase: string; layer: number; forward: number; tokenIds: readonly number[]; tokenStart: number; }
export function activityView(recording: ActivationRecording, selection: ActivitySelection): ActivityView;
export function activityModes(recording: ActivationRecording | null): { id: string; label: string; reason: string | null; clock: string; group: string; clockLabel: string }[];
export function playbackPoints(recording: ActivationRecording, mode: string, viewId?: string): PlaybackPoint[];
export function viewPositions(recording: ActivationRecording, viewId: string): Set<number> | null;
export interface MissingReason {
  kind: 'none' | 'not-selected' | 'dropped' | 'failed' | 'window' | 'fused' | 'elsewhere' | 'representation';
  text: string; nearest: number | null;
}
/** `layer` is the layer being looked at, or -1 for every layer at once. */
export function missingReason(recording: ActivationRecording, viewId: string, position: number, detail?: string, layer?: number): MissingReason;
export function omissionSummary(recording: ActivationRecording): string;
/** Capture kinds one view needs; null when it needs every kind. */
export function viewCaptureRoles(viewId: string): string[] | null;
export function assertMatchedRuns(a: ActivationRecording, b: ActivationRecording): void;
export function attentionRow(frames: ActivityFrame[], query: ActivityFrame, position: number, head: number): (number | undefined)[];
