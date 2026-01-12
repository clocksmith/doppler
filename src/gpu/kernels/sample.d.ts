/**
 * GPU-Side Sampling Kernel
 *
 * Performs sampling entirely on GPU, reducing readback from ~1MB to 4 bytes.
 */

import type { CommandRecorder } from '../command-recorder.js';

export interface SampleOptions {
  temperature?: number;
  topK?: number;
  randomSeed?: number;
  padTokenId?: number;
  logitSoftcap?: number;
  logitsDtype?: 'f16' | 'f32';
  outputBuffer?: GPUBuffer | null;
  outputIndex?: number;
}

export interface SampleResult {
  tokenId: number;
  gpuBuffer: GPUBuffer;
}

/**
 * Run GPU-side argmax (greedy decoding)
 */
export declare function runArgmax(
  logits: GPUBuffer,
  vocabSize: number,
  options?: SampleOptions
): Promise<number>;

/**
 * Run GPU-side top-k sampling
 */
export declare function runGPUSample(
  logits: GPUBuffer,
  vocabSize: number,
  options?: SampleOptions
): Promise<number>;

/**
 * Record GPU argmax (batched, no submit)
 */
export declare function recordArgmax(
  recorder: CommandRecorder,
  logits: GPUBuffer,
  vocabSize: number,
  options?: SampleOptions
): Promise<GPUBuffer>;

/**
 * Record GPU top-k sampling (batched, no submit)
 */
export declare function recordGPUSample(
  recorder: CommandRecorder,
  logits: GPUBuffer,
  vocabSize: number,
  options?: SampleOptions
): Promise<GPUBuffer>;

/**
 * Check if GPU sampling is available
 */
export declare function isGPUSamplingAvailable(): boolean;
