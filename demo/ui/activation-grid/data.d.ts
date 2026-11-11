import type { ActivationTraceReceipt, ActivationFrame } from '../../../src/debug/activation-trace.js';
export interface ActivationRecording {
  schema: 'doppler.activation-recording/v1';
  demoBuild: string;
  display: { layout: string; palette: string; scale: number; scales?: Record<string, number> };
  trace: ActivationTraceReceipt;
  fingerprint?: unknown;
  inputTokenIds?: number[] | null;
  generatedTokenIds?: number[] | null;
  outputText?: string | null;
  prompt?: string | null;
}
export declare function validateRecording(value: unknown): ActivationRecording;
export declare function projectFrame(trace: ActivationTraceReceipt, index: number, position: number): {
  values: Array<number | string | undefined>; sources: Array<ActivationFrame | undefined>;
};
export declare function colorGrid(values: Array<number | string | undefined>, scale: number, range?: 'signed' | 'positive' | 'unit' | 'residency', dark?: boolean): { side: number; rgba: Uint8ClampedArray };

export declare function initialColorScale(values: Array<number | string | undefined>, fallback: number): number;

export declare function heatmapColor(value: number, scale: number, range?: 'signed' | 'positive' | 'unit' | 'residency', dark?: boolean): number[];

export declare function heatmapColors(range?: string, dark?: boolean): number[][];
export interface ProbabilityDistribution {
  entries: { tokenId: number; probability: number }[];
  remaining: number; count: number;
}
export declare function rankProbabilities(values: (number | string | undefined)[]): ProbabilityDistribution | null;
export declare function layerShape(values: number): { width: number; height: number };
export interface LayerHistory { positions: number[]; values: (number | string | undefined)[]; layers: number; }
export declare function layerHistory(recording: ActivationRecording, atMs: number, position: number, difference: boolean, baseline?: ActivationRecording | null): LayerHistory;
