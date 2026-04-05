import { type GPUContext, type GPUBuffersResult, type MemoryStats } from './types.js';

export interface MixedGeometryKVCacheConfig {
  numLayers: number;
  numHeads: number;
  headDim: number;
  globalHeadDim?: number | null;
  maxSeqLen: number;
  useGPU: boolean;
  layout: 'contiguous';
  pageSize: number;
  kvDtype: 'f16' | 'f32';
  slidingWindow?: number | null;
  layerTypes: string[];
}

export declare class MixedGeometryKVCache {
  readonly numLayers: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly maxSeqLen: number;
  readonly layout: 'contiguous';
  readonly pageSize: number;
  readonly kvDtype: 'f16' | 'f32';
  readonly bytesPerElem: number;
  readonly kvSize: number;
  readonly windowSize?: number;
  readonly layers: Array<{
    keysGPU: GPUBuffer;
    valuesGPU: GPUBuffer;
    seqLen: number;
    kvDtype: 'f16' | 'f32';
  }>;
  readonly layerSpecs: Array<{
    layerIdx: number;
    layerType: string;
    layout: 'contiguous' | 'ring';
    headDim: number;
    numHeads: number;
    kvSize: number;
    capacityTokens: number;
    bytesPerToken: number;
    capacityBytes: number;
  }>;

  useGPU: boolean;
  currentSeqLen: number;
  totalTokensSeen: number;
  memoryUsage: number;
  gpuContext: GPUContext | null;

  constructor(config: MixedGeometryKVCacheConfig);

  update(): never;
  updateFromGPU(
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number,
    tokenIds?: number[] | null
  ): void;
  recordUpdateFromGPU(
    recorder: import('../../gpu/kernel-selector.js').CommandRecorder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number,
    tokenIds?: number[] | null
  ): void;
  get(): never;
  getKeyCache(layerIdx: number): GPUBuffer | null;
  getValueCache(layerIdx: number): GPUBuffer | null;
  getGPUBuffers(layerIdx: number): GPUBuffersResult | null;
  hasGPUCache(): boolean;
  clear(): void;
  clone(): MixedGeometryKVCache;
  truncate(length: number): void;
  getMemoryStats(): MemoryStats;
  setGPUContext(gpuContext: GPUContext): void;
  syncToCPU(): Promise<never>;
  destroy(): void;
}
