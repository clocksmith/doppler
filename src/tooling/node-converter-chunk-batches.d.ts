export interface RowChunk {
  rowStart: number;
  rowCount: number;
  offset: number;
  length: number;
}

export declare function createRowChunks(options: {
  rows: number;
  rowChunkRows: number;
  rowSourceBytes: number;
}): RowChunk[];

export declare function mapOrderedChunkBatches<T>(options: {
  chunks: RowChunk[];
  batchSize: number;
  transform(chunk: RowChunk): Promise<T>;
  consume(result: T): Promise<void>;
}): Promise<void>;
