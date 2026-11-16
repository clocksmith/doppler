import type { DynamicWord, PositionLimit, StepObservation, WordModel } from './whole-token-observation.js';

/** The buffer itself, or `transient:<n>` for a per-step transient the program replaces with a private copy. */
export type BufferKey = GPUBuffer | string;
export interface UniformSlot { bytes: Uint8Array; dynamic: DynamicWord[]; offset: number }
export type TemplateEntry =
  | { binding: number; slot: UniformSlot; key?: undefined; offset?: undefined; size?: undefined }
  | { binding: number; slot?: undefined; key: BufferKey; offset: number; size: number | null };
export type TemplateGroup =
  | { opaque: GPUBindGroup; layout?: undefined; label?: undefined; entries?: undefined }
  | { opaque?: undefined; layout: GPUBindGroupLayout; label: string | null; entries: TemplateEntry[] };
export interface DispatchTemplate {
  kind: 'dispatch' | 'indirect';
  pipeline: GPUComputePipeline;
  groups: (TemplateGroup | null)[];
  workgroups?: number[];
  indirect?: BufferKey;
  indirectOffset?: number;
}
export interface WriteTemplate { kind: 'write'; key: BufferKey; offset: number; bytes: Uint8Array; dynamic: DynamicWord[] }
export interface CopyTemplate { kind: 'copy'; src: BufferKey; srcOffset: WordModel; dst: BufferKey; dstOffset: WordModel; size: number }
export interface ClearTemplate { kind: 'clear'; key: BufferKey; offset: number; size: number }
export type TemplateStep = DispatchTemplate | WriteTemplate | CopyTemplate | ClearTemplate;
export interface ReadbackCopyTemplate { src: BufferKey; srcOffset: number; dstOffset: number; size: number }
export interface ReadbackTemplate { copies: ReadbackCopyTemplate[]; dst: GPUBuffer[] }
export interface TransientInfo { size: number; usage: number }

/**
 * Classify one undeclared u32 word observed in consecutive decode steps: constant, or
 * affine in the position. The input token is never inferred; its owner declares it.
 */
export declare function classifyWord(
  values: readonly number[],
  positions: readonly number[],
  variableAdvance?: boolean
): { kind: 'const'; value: number } | { kind: 'affine'; base: number; slope: number; pos0: number } | null;
export declare function evaluateWord(model: WordModel, pos: number, token: number): number;
/** Last position (from the first observed one on) at which an affine word stays in [lo, hi]. */
export declare function affineLastPosition(model: WordModel, lo: number, hi: number): number;
/** The command template the observed steps agree on; throws when they disagree. */
export declare function buildTemplate(obs: StepObservation[]): {
  template: TemplateStep[];
  uniformSlots: UniformSlot[];
  transientTemplate: Map<string, TransientInfo>;
  readback: ReadbackTemplate;
  declared: PositionLimit;
  identity: readonly unknown[];
};
