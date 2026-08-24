export interface TensorPrimarySpan {
  shardIndex?: number;
  shard?: number;
  offset: number;
  size: number;
}

export interface TensorLocation extends TensorPrimarySpan {
  spans?: TensorPrimarySpan[];
}

export interface ShardStorageBackend {
  getFileSize?: (filename: string) => Promise<number>;
  readFile: (filename: string) => Promise<ArrayBuffer>;
}

export declare function isRequestedRangeInsideTensor(
  location: TensorLocation,
  shardIndex: number,
  offset: number,
  length: number | null
): boolean;

export declare function checkFileExistsInBackend(
  storageBackend: ShardStorageBackend,
  filename: string
): Promise<boolean>;

export declare function getFileSizeInBackend(
  storageBackend: ShardStorageBackend,
  filename: string
): Promise<number | null>;
