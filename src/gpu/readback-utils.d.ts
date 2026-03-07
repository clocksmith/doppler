export interface ReadbackBufferEntry {
  buffer: GPUBuffer;
  destroy?: boolean;
  offset?: number;
  size?: number;
}

export function withMappedReadBuffer<T>(
  buffer: GPUBuffer,
  read: (range: ArrayBuffer) => T | Promise<T>,
): Promise<T>;

export function withMappedReadBuffers<T>(
  entries: ReadonlyArray<ReadbackBufferEntry>,
  read: (ranges: Array<ArrayBuffer>) => T | Promise<T>,
): Promise<T>;
