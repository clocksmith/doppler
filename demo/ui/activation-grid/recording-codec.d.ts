import type { ActivationRecording } from './data.js';
export function encodeRecording(recording: ActivationRecording): unknown;
export function decodeRecording(input: unknown): unknown;

interface PackedValues { encoding: string; dtype: string; length: number; data: string; }
type PackedFrame<T> = Omit<T, 'values' | 'summary'> & { values: PackedValues | null; summary?: { blockSize: number; observedValues: number; values: PackedValues } };
type Weight = import('../../../src/debug/activity-trace.js').ActivityWeight;
type PackedWeight = Omit<Weight, 'rms' | 'raw' | 'columnEnergy' | 'error'> & { rms: PackedValues; raw: PackedValues; columnEnergy: PackedValues; error?: PackedValues };
export interface PackedRecording extends Omit<ActivationRecording, 'schema' | 'trace'> {
  schema: 'doppler.activation-recording/v2';
  trace: Omit<ActivationRecording['trace'], 'frames' | 'activity'> & {
    frames: PackedFrame<import('../../../src/debug/activation-trace.js').ActivationFrame>[];
    activity?: Omit<import('../../../src/debug/activity-trace.js').ActivityReceipt, 'frames' | 'weights'> & {
      frames: PackedFrame<import('../../../src/debug/activity-trace.js').ActivityFrame>[]; weights: PackedWeight[];
    };
  };
}
