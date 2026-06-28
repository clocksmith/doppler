export interface KVCacheWriteF32ToF16Options {
  srcOffset?: number;
  dstOffset?: number;
  elementCount: number;
}

export declare function runKVCacheWriteF32ToF16(
  keys: GPUBuffer,
  values: GPUBuffer,
  outputKeys: GPUBuffer,
  outputValues: GPUBuffer,
  options: KVCacheWriteF32ToF16Options
): Promise<void>;

export declare function recordKVCacheWriteF32ToF16(
  recorder: import('../command-recorder.js').CommandRecorder,
  keys: GPUBuffer,
  values: GPUBuffer,
  outputKeys: GPUBuffer,
  outputValues: GPUBuffer,
  options: KVCacheWriteF32ToF16Options
): Promise<void>;
