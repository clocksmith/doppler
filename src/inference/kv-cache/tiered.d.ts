import type { KVCache } from './base.js';
import type { SlidingWindowKVCache } from './sliding-window.js';
import type {
  KVCacheConfig,
  KVGetResult,
  TieredGPUBuffersResult,
  MemoryStats,
  GPUContext,
} from './types.js';

/**
 * Tiered KV cache: hot ring buffer + cold paged cache.
 *
 * @module inference/kv-cache/tiered
 */
export class TieredKVCache {
  numLayers: number;
  numHeads: number;
  headDim: number;
  maxSeqLen: number;
  useGPU: boolean;
  layout: 'tiered';
  kvDtype: 'f16' | 'f32';
  bytesPerElem: number;
  kvSize: number;
  hotWindow: number;
  coldPageSize: number;
  coldDtype: 'f16' | 'f32';
  tieringMode: 'off' | 'fp16' | 'int8' | 'int4';
  coldQuantMode: 'none' | 'int8' | 'int4';
  coldPackedStride: number;
  compression: { mode: 'none' | 'int8' | 'int4'; blockSize: number };
  gating: { mode: 'auto' | 'force_on' | 'force_off'; minAluBwRatio: number };
  currentSeqLen: number;
  totalTokensSeen: number;
  memoryUsage: number;
  gpuContext: GPUContext | null;
  hotCache: SlidingWindowKVCache;
  coldCache: KVCache | null;
  coldLayers: Array<{
    keysPackedGPU: GPUBuffer;
    valuesPackedGPU: GPUBuffer;
    scalesKGPU: GPUBuffer;
    scalesVGPU: GPUBuffer;
    seqLen: number;
  }> | null;

  constructor(config: KVCacheConfig, caches?: { hotCache: SlidingWindowKVCache; coldCache: KVCache } | null);
  clear(): void;
  update(layerIdx: number, keys: Float32Array, values: Float32Array, startPos?: number): void;
  updateFromGPU(layerIdx: number, keysBuffer: GPUBuffer, valuesBuffer: GPUBuffer, startPos: number, numTokens: number): Promise<void>;
  recordUpdateFromGPU(
    recorder: import('../../gpu/kernel-selector.js').CommandRecorder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): Promise<void>;
  get(layerIdx: number): KVGetResult;
  getGPUBuffers(layerIdx: number): TieredGPUBuffersResult | null;
  hasGPUCache(): boolean;
  truncate(length: number): void;
  getMemoryStats(): MemoryStats;
  setGPUContext(gpuContext: GPUContext): void;
  destroy(): void;
  clone(): TieredKVCache;
}
