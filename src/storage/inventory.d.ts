export type StorageBackend = 'opfs' | 'indexeddb' | 'memory' | 'unknown';

export interface StorageInventoryEntry {
  modelId: string;
  backend: StorageBackend;
  root?: string;
  totalBytes: number;
  fileCount: number;
  shardCount: number;
  hasManifest: boolean;
  kind?: 'model' | 'system';
  label?: string;
}

export interface StorageInventoryResult {
  entries: StorageInventoryEntry[];
  systemEntries: StorageInventoryEntry[];
  opfsRoots: string[];
  backendAvailability: {
    opfs: boolean;
    indexeddb: boolean;
  };
}

export function listStorageInventory(): Promise<StorageInventoryResult>;
export function deleteStorageEntry(entry: StorageInventoryEntry): Promise<boolean>;
