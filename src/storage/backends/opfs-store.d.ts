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

export interface OpfsStore {
  init(): Promise<void>;
  openModel(modelId: string, options?: { create?: boolean }): Promise<FileSystemDirectoryHandle>;
  getCurrentModelId(): string | null;
  readFile(filename: string): Promise<ArrayBuffer>;
  readText(filename: string): Promise<string | null>;
  writeFile(filename: string, data: Uint8Array | ArrayBuffer): Promise<void>;
  createWriteStream(filename: string): Promise<OpfsWriteStream>;
  deleteFile(filename: string): Promise<boolean>;
  listFiles(): Promise<string[]>;
  listModels(): Promise<string[]>;
  deleteModel(modelId: string): Promise<boolean>;
  cleanup(): Promise<void>;
}

export function createOpfsStore(config: OpfsStoreConfig): OpfsStore;
