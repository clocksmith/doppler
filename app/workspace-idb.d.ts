export interface WorkspaceIdbStoreOptions {
  dbName?: string;
  storeName?: string;
  workspaceId?: string;
}

export interface WorkspaceFileStat {
  path: string;
  size: number;
  updated: number;
}

export interface WorkspaceIdbStore {
  init(): Promise<void>;
  readBlob(path: string): Promise<Blob | null>;
  readText(path: string): Promise<string | null>;
  writeBlob(path: string, data: Blob | ArrayBuffer | Uint8Array): Promise<void>;
  list(prefix?: string): Promise<string[]>;
  stat(path: string): Promise<WorkspaceFileStat | null>;
  exists(path: string): Promise<boolean>;
  remove(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
}

export function getWorkspaceDbName(workspaceId: string): string;
export function createWorkspaceIdbStore(options?: WorkspaceIdbStoreOptions): WorkspaceIdbStore;
