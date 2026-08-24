import type { LayerContext } from '../types.js';

export declare function releaseSharedAttentionState(
  sharedAttentionState: LayerContext['sharedAttentionState'],
  recorder?: { trackTemporaryBuffer(buffer: GPUBuffer): void } | null
): void;
