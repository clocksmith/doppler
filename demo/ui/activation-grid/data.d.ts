import type { ActivationTraceReceipt, ActivationFrame } from '../../../src/debug/activation-trace.js';
export interface ActivationRecording {
  schema: 'doppler.activation-recording/v1';
  demoBuild: string;
  display: { layout: string; palette: string; scale: number };
  trace: ActivationTraceReceipt;
  fingerprint?: unknown;
  inputTokenIds?: number[] | null;
  generatedTokenIds?: number[] | null;
  outputText?: string | null;
}
export declare function validateRecording(value: unknown): ActivationRecording;
export declare function projectFrame(trace: ActivationTraceReceipt, index: number, position: number): {
  values: Array<number | string | undefined>; sources: Array<ActivationFrame | undefined>;
};
export declare function colorGrid(values: Array<number | string | undefined>, scale: number): { side: number; rgba: Uint8ClampedArray };
