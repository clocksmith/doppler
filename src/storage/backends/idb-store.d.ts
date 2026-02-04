export interface IdbStoreConfig {
  dbName: string;
  shardStore: string;
  metaStore: string;
  chunkSizeBytes: number;
}

export interface IdbWriteStream {
  write(chunk: Uint8Array | ArrayBuffer): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
}

export interface IdbStore {
  init(): Promise<void>;
  openModel(modelId: string, options?: { create?: boolean }): Promise<null>;
  getCurrentModelId(): string | null;
  readFile(filename: string): Promise<ArrayBuffer>;
  readText(filename: string): Promise<string | null>;
  writeFile(filename: string, data: Uint8Array | ArrayBuffer): Promise<void>;
  createWriteStream(filename: string): Promise<IdbWriteStream>;
  deleteFile(filename: string): Promise<boolean>;
  listFiles(): Promise<string[]>;
  listModels(): Promise<string[]>;
  getModelStats(modelId: string): Promise<{
    totalBytes: number;
    fileCount: number;
    shardCount: number;
    hasManifest: boolean;
  }>;
  deleteModel(modelId: string): Promise<boolean>;
  writeManifest(text: string): Promise<void>;
  readManifest(): Promise<string | null>;
  writeTokenizer(text: string): Promise<void>;
  readTokenizer(): Promise<string | null>;
  cleanup(): Promise<void>;
}

export function createIdbStore(config: IdbStoreConfig): IdbStore;
