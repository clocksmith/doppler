import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { StepContractDeclaration } from '../../../../gpu/step-contract.js';

/** A u32 word as a function of the decode position and input token. */
export type WordModel =
  | { kind: 'const'; value: number }
  | { kind: 'token' }
  | { kind: 'affine'; base: number; slope: number; pos0: number };
export interface DynamicWord { word: number; model: WordModel }
export interface BindGroupEntryRecord { binding: number; resource: GPUBindingResource }
export interface BindGroupRecord { layout: GPUBindGroupLayout; label: string | null; entries: BindGroupEntryRecord[] }
export type BindGroupDescriptors = WeakMap<GPUBindGroup, BindGroupRecord>;
/** One bind group entry as observed in a decode step. */
export type ObservedEntry =
  | { binding: number; other: unknown }
  | { binding: number; uniform: Uint8Array | null }
  | { binding: number; buffer: GPUBuffer; offset: number; size: number | null };
/** One bind group as observed in a decode step: opaque when created outside observation. */
export type ObservedGroup =
  | { opaque: GPUBindGroup }
  | { layout: GPUBindGroupLayout; label: string | null; entries: ObservedEntry[] };
export type ObservedOp =
  | { kind: 'dispatch'; pipeline: GPUComputePipeline | null; groups: ObservedGroup[]; workgroups: number[] }
  | { kind: 'indirect'; pipeline: GPUComputePipeline | null; groups: ObservedGroup[]; buffer: GPUBuffer; offset: number }
  | { kind: 'write'; buffer: GPUBuffer; offset: number; bytes: Uint8Array; token: boolean }
  | { kind: 'copy'; src: GPUBuffer; srcOffset: number; dst: GPUBuffer; dstOffset: number; size: number }
  | { kind: 'clear'; buffer: GPUBuffer; offset: number; size: number };
/** A declared range of step positions (inclusive) and the operation that declared it. */
export interface PositionLimit { owner: string; min: number; max: number }
export interface UniformContents { uniform: Uint8Array; known: Uint8Array }
/** A method replaced on a WebGPU, pool or recorder object during observation. */
export interface SavedPatch { target: object; name: string; own: boolean; previous: Function }

export declare const U32: number;
export declare function isMapRead(buffer: GPUBuffer | null | undefined): boolean;

/** One observed decode step. Owns nothing; records references only. */
export declare class StepObservation {
  constructor(pos: number, token: number);
  pos: number;
  token: number;
  gpuTokenInput: GPUBuffer | null;
  ops: ObservedOp[];
  transient: Set<GPUBuffer>;
  uniformBytes: Map<GPUBuffer, UniformContents>;
  boundUniforms: Set<GPUBuffer>;
  gpuWrittenUniforms: Set<GPUBuffer>;
  pendingTokenInputs: { buffer: GPUBuffer; offset: number }[];
  limits: PositionLimit[];
  rangeLimit: PositionLimit | null;
  identity: readonly unknown[];
  problems: string[];
  problem(message: string): void;
  declare(declaration: StepContractDeclaration): void;
  takeTokenInput(buffer: GPUBuffer, offset: number, bytes: Uint8Array): boolean;
  writeUniform(buffer: GPUBuffer, offset: number, bytes: Uint8Array): void;
  /** Intersection of the declared position ranges; must contain this step's position. */
  range(): PositionLimit;
  uniformSnapshot(buffer: GPUBuffer, offset: number, size: number | null): Uint8Array | null;
  /** Preserve CPU initialization and GPU copy/clear ordering on the same uniform allocation. */
  preserveGpuWrittenUniform(buffer: GPUBuffer): void;
  snapshotBindGroup(bindGroup: GPUBindGroup, descriptors: BindGroupDescriptors): ObservedGroup;
}

export declare function patchMethod(
  saved: SavedPatch[], target: object, name: string, wrap: (previous: Function) => Function
): void;
export declare function restorePatches(saved: SavedPatch[]): void;
/** Install observation hooks for one recorded decode step. Returns a stop function. */
export declare function observeStep(
  device: GPUDevice, recorder: CommandRecorder, observation: StepObservation, descriptors: BindGroupDescriptors
): () => void;
