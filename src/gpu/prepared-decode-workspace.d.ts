/** Borrowed by serial decode recorders; owned and released by their session. */
export declare class PreparedDecodeWorkspace {
  constructor(device: GPUDevice);
  getBuffer(slot: string, bytes: number): GPUBuffer;
  ownsBuffer(buffer: GPUBuffer): boolean;
  getBufferSlot(buffer: GPUBuffer): string | null;
  getBindGroupForOutputs(operation: string, outputs: GPUBuffer[], descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  getBindGroup(slot: string, descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  getStats(): { buffers: number; bytes: number; bindings: number; hits: number; misses: number };
  release(): void;
}
