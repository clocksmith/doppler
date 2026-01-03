/**
 * KV Quantization Kernels
 *
 * Quantize KV tensors into packed int8/int4 with per-token+head scales.
 *
 * @module gpu/kernels/kv-quantize
 */

export interface KVQuantizeOptions {
  numKVHeads: number;
  headDim: number;
  startPos: number;
  numTokens: number;
  packedStride: number;
  mode?: 'int8' | 'int4';
}

export declare function runKVQuantize(
  keys: GPUBuffer,
  values: GPUBuffer,
  outputKeys: GPUBuffer,
  outputValues: GPUBuffer,
  scalesK: GPUBuffer,
  scalesV: GPUBuffer,
  options: KVQuantizeOptions
): Promise<void>;

export declare function recordKVQuantize(
  recorder: import('../command-recorder.js').CommandRecorder,
  keys: GPUBuffer,
  values: GPUBuffer,
  outputKeys: GPUBuffer,
  outputValues: GPUBuffer,
  scalesK: GPUBuffer,
  scalesV: GPUBuffer,
  options: KVQuantizeOptions
): Promise<void>;
