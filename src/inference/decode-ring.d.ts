/**
 * Decode ring buffer pool for GPU batch decode.
 *
 * Preallocates fixed GPUBuffer rings for token outputs, stop flags, and staging readback.
 * Ring sizing derives from runtime batching config (batchSize × readbackInterval).
 */

export interface DecodeRingConfig {
  batchSize: number;
  tokensPerInterval: number;
  stopCheckMode: 'batch' | 'per-token';
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
}

export interface DecodeRingSlot {
  index: number;
  tokens: GPUBuffer | null;
  stop: GPUBuffer | null;
  stagingTokens: GPUBuffer | null;
  stagingStop: GPUBuffer | null;
  tokensPerInterval: number;
  zeroStopData: Uint32Array | null;
}

export declare class DecodeRing {
  ensure(config: DecodeRingConfig): void;
  acquire(): DecodeRingSlot | null;
  advance(): void;
  reset(): void;
  release(): void;
}
