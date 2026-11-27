export declare function hasUnrewindableDecodeState(state: {
  linearAttentionRuntime?: { layers?: { size: number } } | null;
  convLayerStates?: { size: number } | null;
} | null | undefined): boolean;

export declare function assertShortBatchContinuationAllowed(
  state: Parameters<typeof hasUnrewindableDecodeState>[0],
  actualCount: number,
  requestedCount: number
): void;

export declare function assertBatchRetryInPlaceAllowed(
  state: Parameters<typeof hasUnrewindableDecodeState>[0],
  error: unknown
): void;
