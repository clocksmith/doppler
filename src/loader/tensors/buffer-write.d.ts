export declare function alignTo4(size: number): number;
export declare function toUint8View(data: ArrayBuffer | ArrayBufferView): Uint8Array;
export declare function writeBufferAligned(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBuffer | ArrayBufferView
): void;
