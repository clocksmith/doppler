import type { InputSpan } from './prefix-embedding.js';

export interface PrefillPolicyState {
  runtimeConfig?: { inference?: { session?: {
    prefillTokenChunkSize?: number | null;
    prefillMaxInFlightChunks?: number;
  } } };
  modelConfig?: { sessionSettings?: { prefillTokenChunkSize?: number | null } } | null;
  kvCache?: { layout?: string; hasGPUCache?(): boolean } | null;
}

export declare function resolvePrefillMaxInFlightChunks(state: PrefillPolicyState): number;

export declare function resolveEffectivePrefillTokenChunkSize(
  state: PrefillPolicyState
): number | null | undefined;

export declare function resolvePrefillTokenChunkSize(state: PrefillPolicyState, numTokens: number): number | null;

export declare function releasePerLayerInputBuffer(
  buffer: GPUBuffer | null | undefined,
  recorder: { trackTemporaryBuffer(buffer: GPUBuffer): void } | null | undefined,
  decodeBuffers: { ownsBuffer(buffer: GPUBuffer): boolean } | null | undefined,
  pleCache?: { ownedBuffers?: Set<GPUBuffer> } | null
): void;

/** One _prefill call: its first position and whether its hidden output feeds a later token chunk. */
export interface PrefillCallShape {
  startPos: number;
  returnHidden: boolean;
}

/**
 * Prefill command batching is bypassed for debug, paged, multimodal-bidirectional with a GPU
 * cache, and token-chunked calls that share KV with another chunk call (startPos > 0 or a
 * non-final chunk). A prompt that fits one token chunk is one direct call and keeps batching.
 */
export declare function shouldDisablePrefillCommandBatching(
  state: PrefillPolicyState,
  opts: Record<string, unknown>,
  multimodalBidirectionalSpan: InputSpan | null,
  call: PrefillCallShape
): boolean;
