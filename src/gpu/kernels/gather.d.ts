/**
 * Gather (Embedding Lookup) Kernels
 *
 * Provides token embedding lookups from embedding tables.
 */

import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor, TensorDtype } from '../tensor.js';
import type { OutputBufferOptions } from './types.js';

/** Gather kernel options */
export interface GatherOptions extends OutputBufferOptions {
  useVec4?: boolean;
  embeddingDtype?: 'f16' | 'f32';
  /**
   * Output dtype. When 'f16', converts F32 embeddings to F16 output.
   * Default: 'f32'
   */
  outputDtype?: 'f16' | 'f32';
  /**
   * True if embeddings are stored as [hidden_size, vocab_size] (GGUF layout).
   * False if embeddings are stored as [vocab_size, hidden_size] (PyTorch layout).
   * Default: false (RDRR format uses PyTorch layout from SafeTensors).
   */
  transpose?: boolean;
  /** Optional index offset into the token indices buffer. */
  indexOffset?: number;
  /** Optional indirect dispatch buffer for GPU-driven workgroup counts. */
  indirectBuffer?: GPUBuffer | null;
  /** Byte offset into indirect dispatch buffer (default: 0). */
  indirectOffset?: number;
}

/**
 * Run gather/embedding lookup
 */
export declare function runGather(
  indices: GPUBuffer,
  embeddings: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options?: GatherOptions
): Promise<Tensor>;

/**
 * Record gather (batched, no submit)
 */
export declare function recordGather(
  recorder: CommandRecorder,
  indices: GPUBuffer,
  embeddings: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options?: GatherOptions
): Promise<Tensor>;
