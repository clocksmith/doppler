/**
 * Basis-Decomposed Paged KV Cache
 *
 * Experimental KV cache layout for BDPA attention.
 */

import type { KVCacheConfig, BDPAGPUBuffersResult, MemoryStats } from './types.js';
import { KVCache } from './base.js';
import type { CommandRecorder } from '../../gpu/kernel-selector.js';

export interface BasisDecomposedPagedCacheConfig extends KVCacheConfig {
  bdpaVocabSize?: number;
}

export declare class BasisDecomposedPagedCache extends KVCache {
  readonly basisVocabSize: number;
  readonly maxContextPages: number;
  readonly basisDtype: 'f16';
  readonly deltaDtype: 'int8';
  readonly tokenIds: Int32Array;
  readonly tokenIdsSet: Uint8Array;

  constructor(config: BasisDecomposedPagedCacheConfig);

  getGPUBuffers(layerIdx: number): BDPAGPUBuffersResult;
  hasGPUCache(): boolean;
  clear(): void;
  truncate(length: number): void;

  updateFromGPU(
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number,
    tokenIds?: number[] | null
  ): Promise<void>;

  recordUpdateFromGPU(
    recorder: CommandRecorder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number,
    tokenIds?: number[] | null
  ): void;

  getMemoryStats(): MemoryStats;
}
