export interface OpfsStoreConfig {
  opfsRootDir: string;
  useSyncAccessHandle: boolean;
  maxConcurrentHandles: number;
}

export interface OpfsWriteStream {
  write(chunk: Uint8Array | ArrayBuffer): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
}

export interface OpfsWriteStreamOptions {
  append?: boolean;
  expectedOffset?: number | null;
}

export interface OpfsStore {
  init(): Promise<void>;
  openModel(modelId: string, options?: { create?: boolean }): Promise<FileSystemDirectoryHandle>;
  getCurrentModelId(): string | null;
  getFileSize(filename: string): Promise<number>;
  readFile(filename: string): Promise<ArrayBuffer>;
  readFileRange(filename: string, offset?: number, length?: number | null): Promise<ArrayBuffer>;
  readFileRangeStream(
    filename: string,
    offset?: number,
    length?: number | null,
    options?: { chunkBytes?: number }
  ): AsyncIterable<Uint8Array>;
  readText(filename: string): Promise<string | null>;
  writeFile(filename: string, data: Uint8Array | ArrayBuffer): Promise<void>;
  createWriteStream(filename: string, options?: OpfsWriteStreamOptions): Promise<OpfsWriteStream>;
  deleteFile(filename: string): Promise<boolean>;
  listFiles(): Promise<string[]>;
  listModels(): Promise<string[]>;
  deleteModel(modelId: string): Promise<boolean>;
  cleanup(): Promise<void>;
}

export function createOpfsStore(config: OpfsStoreConfig): OpfsStore;
