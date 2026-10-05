import type { InputSpan } from './prefix-embedding.js';

export interface PrefillPolicyState {
  runtimeConfig?: { inference?: { session?: { prefillTokenChunkSize?: number | null } } };
  modelConfig?: { sessionSettings?: { prefillTokenChunkSize?: number | null } } | null;
}

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

export declare function shouldDisablePrefillCommandBatching(
  state: Record<string, unknown>,
  opts: Record<string, unknown>,
  multimodalBidirectionalSpan: InputSpan | null
): boolean;
