import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { QsaRopeTables, QsaSelection } from '../../../../gpu/kernels/qsa-indexer.js';
import type { ResolvedIndexedAttention } from '../../../../config/schema/hyper-connection-decoder.schema.js';

export interface QsaIndexerCacheOptions {
  contract: ResolvedIndexedAttention;
  /** Token capacity per indexed layer; a multiple of compressRatio. */
  capacity: number;
}

export interface QsaAppendInput {
  /** index_qk_proj output [numTokens, (queryHeads + 1) * headDim]. */
  projection: Tensor;
  qNormWeight: Tensor;
  kNormWeight: Tensor;
  rope: QsaRopeTables;
  /** Must equal the layer's current length (contiguous positions). */
  startPos: number;
  numTokens: number;
}

export interface QsaCacheCheckpoint {
  readonly epoch: number;
  readonly lengths: Readonly<Record<number, number>>;
}

export interface QsaCallSelection extends QsaSelection {
  /** Score scratch, null when every query took the dense shortcut. Caller releases. */
  scores: Tensor | null;
}

export declare class QsaIndexerCache {
  constructor(options: QsaIndexerCacheOptions);
  readonly contract: ResolvedIndexedAttention;
  readonly capacity: number;
  readonly epoch: number;
  readonly destroyed: boolean;
  length(layerIdx: number): number;
  buffers(layerIdx: number): { rawKeys: GPUBuffer; blockKeys: GPUBuffer };
  /** Append raw keys and complete block keys; returns the prepared query [numTokens, queryHeads * headDim]. */
  append(target: CommandRecorder | null, layerIdx: number, input: QsaAppendInput): Promise<Tensor>;
  /** Score and select for the rows just appended. */
  select(target: CommandRecorder | null, layerIdx: number, query: Tensor, call: { startPos: number; numTokens: number }): Promise<QsaCallSelection>;
  /** Shorten every layer to at most `length` rows; invalidates earlier checkpoints. */
  truncate(length: number): void;
  reset(): void;
  checkpoint(): QsaCacheCheckpoint;
  /** Restore lengths from a checkpoint taken since the last truncate or reset. */
  restore(checkpoint: QsaCacheCheckpoint): void;
  /** Mark destroyed; release buffers after pending operations and submitted GPU work settle. */
  destroy(): Promise<void>;
}
