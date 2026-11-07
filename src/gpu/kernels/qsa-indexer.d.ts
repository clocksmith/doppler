import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

/** Precomputed RoPE tables, [positions, rotaryDim / 2] f32 each. */
export interface QsaRopeTables {
  cos: GPUBuffer;
  sin: GPUBuffer;
}

export interface QsaIndexerGeometry {
  queryHeads: number;
  headDim: number;
  rotaryDim: number;
}

export interface QsaPrepareOptions extends QsaIndexerGeometry {
  numTokens: number;
  /** Cache row and position of the first token; positions are contiguous cache rows. */
  startPos: number;
  eps: number;
  /** Raw key cache capacity in tokens. */
  capacity: number;
  outputBuffer?: GPUBuffer | null;
}

export interface QsaBlockKeyOptions {
  firstBlock: number;
  blockCount: number;
  headDim: number;
  rotaryDim: number;
  compressRatio: number;
  eps: number;
  capacity: number;
}

export interface QsaSelectionGeometry {
  numTokens: number;
  startPos: number;
  compressRatio: number;
  blockTopK: number;
  /** blockTopK * compressRatio + compressRatio - 1. */
  denseVisibleLimit: number;
}

export interface QsaScoreOptions extends QsaSelectionGeometry {
  queryHeads: number;
  headDim: number;
  /** f32(sqrt(headDim)). */
  scoreDivisor: number;
}

/** Per query: selection rows [numTokens, maxSelected] u32 and meta [numTokens, 2] u32 (count, mode). */
export interface QsaSelection {
  selection: GPUBuffer;
  meta: GPUBuffer;
  maxSelected: number;
}

export interface IndexedAttentionOptions {
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  scale: number;
  /** Explicit element type of both caller-owned contiguous cache buffers. */
  kvDtype: 'f16' | 'f32';
  /** Rows present in the KV cache buffers. */
  kvRows: number;
  outputBuffer?: GPUBuffer | null;
}

/** Whether any query in the call has more visible tokens than the dense limit. */
export declare function qsaHasSparseQueries(options: { numTokens: number; startPos: number; denseVisibleLimit: number }): boolean;

export declare function runQsaPrepare(
  projection: Tensor, qNormWeight: Tensor, rope: QsaRopeTables, rawKeys: GPUBuffer, options: QsaPrepareOptions
): Promise<Tensor>;
export declare function recordQsaPrepare(
  recorder: CommandRecorder, projection: Tensor, qNormWeight: Tensor, rope: QsaRopeTables, rawKeys: GPUBuffer, options: QsaPrepareOptions
): Promise<Tensor>;

export declare function runQsaBlockKeys(
  rawKeys: GPUBuffer, kNormWeight: Tensor, rope: QsaRopeTables, blockKeys: GPUBuffer, options: QsaBlockKeyOptions
): Promise<void>;
export declare function recordQsaBlockKeys(
  recorder: CommandRecorder, rawKeys: GPUBuffer, kNormWeight: Tensor, rope: QsaRopeTables, blockKeys: GPUBuffer, options: QsaBlockKeyOptions
): Promise<void>;

/** Scores for sparse queries; null when every query takes the dense shortcut. */
export declare function runQsaBlockScores(query: Tensor, blockKeys: GPUBuffer, options: QsaScoreOptions): Promise<Tensor | null>;
export declare function recordQsaBlockScores(
  recorder: CommandRecorder, query: Tensor, blockKeys: GPUBuffer, options: QsaScoreOptions
): Promise<Tensor | null>;

export declare function runQsaSelect(scores: Tensor | null, options: QsaSelectionGeometry): Promise<QsaSelection>;
export declare function recordQsaSelect(
  recorder: CommandRecorder, scores: Tensor | null, options: QsaSelectionGeometry
): Promise<QsaSelection>;

export declare function runIndexedAttention(
  query: Tensor, keys: GPUBuffer, values: GPUBuffer, selection: QsaSelection, options: IndexedAttentionOptions
): Promise<Tensor>;
export declare function recordIndexedAttention(
  recorder: CommandRecorder, query: Tensor, keys: GPUBuffer, values: GPUBuffer, selection: QsaSelection, options: IndexedAttentionOptions
): Promise<Tensor>;
