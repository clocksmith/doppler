export interface ModelReadSession {
  readonly modelId: string;
  readFile(path: string): Promise<ArrayBuffer>;
  readRange(path: string, offset?: number, length?: number | null): Promise<ArrayBuffer>;
  readText(path: string): Promise<string | null>;
  readManifest(): Promise<string | null>;
  getFileSize(path: string): Promise<number | null>;
  streamRange(path: string, offset?: number, length?: number | null, options?: { chunkBytes?: number }): AsyncIterable<Uint8Array>;
  close(): Promise<void>;
}
export type StorageBackend = import('./backends/memory-store.js').MemoryStore
  | import('./backends/opfs-store.js').OpfsStore | import('./backends/idb-store.js').IdbStore;
export function createModelReadSession(backend: StorageBackend, modelId: string, chunkBytes: number): Promise<ModelReadSession>;
