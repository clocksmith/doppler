import type { DynamicWord, WordModel } from './whole-token-observation.js';

export type ReplayStep =
  | { kind: 'copy'; src: GPUBuffer; srcOffset: WordModel; dst: GPUBuffer; dstOffset: WordModel; size: number }
  | { kind: 'clear'; buffer: GPUBuffer; offset: number; size: number }
  | { kind: 'dispatch'; pipeline: GPUComputePipeline; groups: (GPUBindGroup | null)[]; workgroups: number[] }
  | { kind: 'indirect'; pipeline: GPUComputePipeline; groups: (GPUBindGroup | null)[]; indirect: GPUBuffer; indirectOffset: number };
export interface ReplayWrite { buffer: GPUBuffer; offset: number; bytes: Uint8Array; dynamic: DynamicWord[] }
export interface ReplayReadbackCopy { src: GPUBuffer; srcOffset: number; dstOffset: number; size: number }
export interface EncodedStep { pos: number; staging: GPUBuffer; commandBuffer: GPUCommandBuffer }
export interface BatchFeedback { staging: GPUBuffer; offset: number; feedback: GPUBuffer | null; saveFeedback: GPUBuffer }
export interface RecordedStepState {
  device: GPUDevice;
  steps: ReplayStep[];
  writes: ReplayWrite[];
  readback: ReplayReadbackCopy[];
  staging: GPUBuffer[];
  dynamicUniformWords: { index: number; model: WordModel }[];
  uniformBuffer: GPUBuffer;
  uniformMirror: ArrayBuffer;
  uniformWords: Uint32Array;
  dynamicUniformBytes: number;
}
export function encodeRecordedStep(program: RecordedStepState, pos: number, stagingIndex: number,
  batch?: BatchFeedback | null, prefix?: ((encoder: GPUCommandEncoder) => void) | null): EncodedStep;
export function writeRecordedStepInputs(program: RecordedStepState, pos: number, token: number): void;
