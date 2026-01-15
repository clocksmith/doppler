export interface CheckpointStoreOptions {
  dbName?: string;
  storeName?: string;
  version?: number;
}

export declare function saveCheckpoint(
  key: string,
  data: unknown,
  options?: CheckpointStoreOptions
): Promise<void>;

export declare function loadCheckpoint(
  key: string,
  options?: CheckpointStoreOptions
): Promise<unknown | null>;
