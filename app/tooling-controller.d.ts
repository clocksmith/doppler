export interface ToolingControllerOptions {
  toolSelect?: HTMLSelectElement | null;
  argsInput?: HTMLTextAreaElement | null;
  runButton?: HTMLButtonElement | null;
  refreshButton?: HTMLButtonElement | null;
  outputEl?: HTMLElement | null;
  statusEl?: HTMLElement | null;
}

export declare class ToolingController {
  constructor(options?: ToolingControllerOptions);
  init(): void;
  setVfs(vfs: unknown): Promise<void>;
  refresh(): Promise<void>;
  runSelected(): Promise<void>;
}
