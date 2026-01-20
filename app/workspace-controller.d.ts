export interface WorkspaceControllerOptions {
  importButton?: HTMLElement | null;
  refreshButton?: HTMLElement | null;
  statusEl?: HTMLElement | null;
  filesEl?: HTMLElement | null;
}

export declare class WorkspaceController {
  constructor(options?: WorkspaceControllerOptions);
  init(): Promise<void>;
  refresh(): Promise<void>;
  importFolder(): Promise<void>;
}
