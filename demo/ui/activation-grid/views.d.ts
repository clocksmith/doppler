import type { ActivationRecording } from './data.js';
import type { ActivityFrame } from '../../../src/debug/activity-trace.js';
export type { ActivityFrame };
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
export function assertMatchedRuns(a: ActivationRecording, b: ActivationRecording): void;
export function attentionRow(frames: ActivityFrame[], query: ActivityFrame, position: number, head: number): (number | undefined)[];
