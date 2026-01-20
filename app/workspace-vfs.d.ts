export interface WorkspaceVfsOptions {
  backend?: 'opfs' | 'indexeddb' | 'auto';
  rootDirName?: string;
  workspaceId?: string;
  dbName?: string;
  storeName?: string;
  useSyncAccessHandle?: boolean;
  maxConcurrentHandles?: number;
}

export interface WorkspaceFileStat {
  path: string;
  size: number;
  updated: number;
}

export interface WorkspaceVfs {
  backendType: 'opfs' | 'indexeddb';
  workspaceId: string;
  rootDirName: string;
  readText(path: string): Promise<string | null>;
  readBlob(path: string): Promise<Blob | null>;
  writeText(path: string, content: string): Promise<void>;
  writeBlob(path: string, data: Blob | ArrayBuffer | Uint8Array): Promise<void>;
  list(prefix?: string): Promise<string[]>;
  stat(path: string): Promise<WorkspaceFileStat | null>;
  exists(path: string): Promise<boolean>;
  remove(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
}

export function createWorkspaceVfs(options?: WorkspaceVfsOptions): Promise<WorkspaceVfs>;
