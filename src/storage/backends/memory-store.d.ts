export interface MemoryStoreConfig {
  maxBytes: number;
}

export interface MemoryWriteStream {
  write(chunk: Uint8Array | ArrayBuffer): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
}

export interface MemoryStore {
  init(): Promise<void>;
  openModel(modelId: string, options?: { create?: boolean }): Promise<null>;
  getCurrentModelId(): string | null;
  readFile(filename: string): Promise<ArrayBuffer>;
  readText(filename: string): Promise<string | null>;
  writeFile(filename: string, data: Uint8Array | ArrayBuffer): Promise<void>;
  createWriteStream(filename: string): Promise<MemoryWriteStream>;
  deleteFile(filename: string): Promise<boolean>;
  listFiles(): Promise<string[]>;
  listModels(): Promise<string[]>;
  deleteModel(modelId: string): Promise<boolean>;
  writeManifest(text: string): Promise<void>;
  readManifest(): Promise<string | null>;
  writeTokenizer(text: string): Promise<void>;
  readTokenizer(): Promise<string | null>;
  cleanup(): Promise<void>;
}

export function createMemoryStore(config: MemoryStoreConfig): MemoryStore;
