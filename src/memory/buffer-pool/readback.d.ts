export interface BufferReadbackOwner {
  device: GPUDevice | null;
  createStagingBuffer(bytes: number): GPUBuffer;
  isActive(buffer: GPUBuffer): boolean;
  release(buffer: GPUBuffer, reusable: boolean): void;
}
export type RecordedReadback = Pick<import('../../gpu/command-recorder.js').CommandRecorder,
  'device' | 'getEncoder' | 'submitPartial'>;
/** The owner retains allocation/retirement accounting; this operation owns mapping and publication. */
export function readPooledBufferSlice(buffer: GPUBuffer, offset: number, size: number,
  recorder: RecordedReadback | null, owner: BufferReadbackOwner): Promise<ArrayBuffer>;
