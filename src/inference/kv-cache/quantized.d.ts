import type { KVCacheConfig, MemoryStats, QuantizedGPUBuffersResult } from './types.js';

export class QuantizedKVCache {
  numLayers: number;
  numHeads: number;
  headDim: number;
  maxSeqLen: number;
  useGPU: boolean;
  layout: 'contiguous_quantized';
  kvDtype: 'f16' | 'f32';
  quantMode: 'turboquant' | 'turboquant_prod';
  bitWidth: number;
  prodMode: boolean;
  packFactor: number;
  packedStride: number;
  mseBitWidth?: number;
  msePackFactor?: number;
  msePackedStride?: number;
  residualPackedStride?: number;
  currentSeqLen: number;
  totalTokensSeen: number;
  memoryUsage: number;
  rotationMatrixBuffer: GPUBuffer | null;
  codebookCentroidsBuffer: GPUBuffer | null;
  codebookBoundariesBuffer: GPUBuffer | null;
  qjlMatrixBuffer: GPUBuffer | null;

  constructor(config: KVCacheConfig & {
    quantMode: 'turboquant' | 'turboquant_prod';
    bitWidth?: number;
    prodMode?: boolean;
  });
  setSharedBuffers(buffers: {
    rotationMatrixBuffer: GPUBuffer;
    codebookCentroidsBuffer: GPUBuffer;
    codebookBoundariesBuffer: GPUBuffer;
    qjlMatrixBuffer?: GPUBuffer | null;
    release?: (() => void) | null;
  }): void;
  updateFromGPU(layerIdx: number, keysBuffer: GPUBuffer, valuesBuffer: GPUBuffer, startPos: number, numTokens: number): Promise<void>;
  recordUpdateFromGPU(
    recorder: import('../../gpu/kernel-selector.js').CommandRecorder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): Promise<void>;
  getGPUBuffers(layerIdx: number): QuantizedGPUBuffersResult | null;
  hasGPUCache(): boolean;
  clear(): void;
  truncate(length: number): void;
  getMemoryStats(): MemoryStats;
  setGPUContext(): void;
  destroy(): void;
}
