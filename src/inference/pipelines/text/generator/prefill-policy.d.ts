import type { InputSpan } from './prefix-embedding.js';

export declare function resolveEffectivePrefillTokenChunkSize(
  state: Record<string, unknown>
): number | null | undefined;

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
