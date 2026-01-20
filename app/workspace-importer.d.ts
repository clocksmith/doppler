export interface WorkspaceImportProgress {
  completed: number;
  total: number;
  path?: string;
}

export interface WorkspaceImportResult {
  total: number;
  workspaceId: string;
  backendType: string;
}

export interface WorkspaceImporterOptions {
  backendType?: string;
  rootDirName?: string;
  excludeDirs?: string[];
  onProgress?: (progress: WorkspaceImportProgress) => void;
}

export interface WorkspaceImporter {
  importDirectory(params?: { workspaceId?: string }): Promise<WorkspaceImportResult | null>;
  terminate(): void;
}

export function createWorkspaceImporter(options?: WorkspaceImporterOptions): WorkspaceImporter;
