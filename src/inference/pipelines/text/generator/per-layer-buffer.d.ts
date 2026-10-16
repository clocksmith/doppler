export declare function releasePerLayerInputBuffer(
  buffer: GPUBuffer | null | undefined,
  recorder: { trackTemporaryBuffer(buffer: GPUBuffer): void } | null | undefined,
  decodeBuffers: { ownsBuffer(buffer: GPUBuffer): boolean } | null | undefined,
  pleCache?: { ownedBuffers?: Set<GPUBuffer> } | null
): void;
