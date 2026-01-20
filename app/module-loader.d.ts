export interface WorkspaceModuleOptions {
  force?: boolean;
}

export interface WorkspaceModuleCacheResult {
  cleared: string;
  existed?: boolean;
}

export function loadWorkspaceModule(
  vfs: { readText: (path: string) => Promise<string | null> },
  path: string,
  options?: WorkspaceModuleOptions
): Promise<unknown>;

export function clearWorkspaceModuleCache(path?: string | null): WorkspaceModuleCacheResult;
