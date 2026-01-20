export interface WorkspaceOpfsStoreOptions {
  rootDirName?: string;
  workspaceId?: string;
  useSyncAccessHandle?: boolean;
  maxConcurrentHandles?: number;
}

export interface WorkspaceFileStat {
  path: string;
  size: number;
  updated: number;
}

export interface WorkspaceWriteStream {
  write(chunk: Uint8Array | ArrayBuffer | Blob): Promise<void>;
  close(): Promise<void>;
  abort(): Promise<void>;
}

export interface WorkspaceOpfsStore {
  init(): Promise<void>;
  readBlob(path: string): Promise<Blob | null>;
  readText(path: string): Promise<string | null>;
  writeBlob(path: string, data: Blob | ArrayBuffer | Uint8Array): Promise<void>;
  createWriteStream(path: string): Promise<WorkspaceWriteStream | null>;
  list(prefix?: string): Promise<string[]>;
  stat(path: string): Promise<WorkspaceFileStat | null>;
  exists(path: string): Promise<boolean>;
  remove(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
}

export function createWorkspaceOpfsStore(options?: WorkspaceOpfsStoreOptions): WorkspaceOpfsStore;
