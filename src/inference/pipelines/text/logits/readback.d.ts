export declare function readBufferWithCleanup(
  buffer: GPUBuffer,
  byteLength: number,
  cleanup?: (() => void) | null,
  reader?: ((buffer: GPUBuffer, byteLength: number) => Promise<ArrayBuffer>) | null
): Promise<ArrayBuffer>;
